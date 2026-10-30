"""Ingest security gates and the source-doc-id helper.

Two gates guard the only place untrusted input enters the system:
validate_local_path (path traversal / extension / size) and validate_url
(SSRF). Both must run before any content is read or fetched.
"""

import hashlib
import ipaddress
import re
import socket
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

import requests
from langchain_core.documents import Document

from ask_the_docs.config import Settings, get_settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
ALLOWED_URL_SCHEMES = {"http", "https"}
MAX_REDIRECTS = 3

_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalizes whitespace and unicode artifacts in loaded text before
    chunking. Does not attempt boilerplate/nav removal (e.g. leftover site
    chrome from URL loads) -- too fragile to do reliably; known limitation."""
    text = "\n".join(text.splitlines())
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    lines = [_HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", "\n".join(lines))
    return text.strip()


class UnsupportedFileTypeError(ValueError):
    pass


class PathSecurityError(ValueError):
    pass


class UrlSecurityError(ValueError):
    pass


def make_source_doc_id(source: str) -> str:
    """Stable short id for a source path/URL; used for bulk-delete-by-document later."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def validate_local_path(path: Path, base_dir: Path) -> Path:
    """Resolve `path` and ensure it stays inside `base_dir`, has an allowed
    extension, and is within the configured size limit."""
    settings = get_settings()
    resolved_base = base_dir.resolve()
    resolved = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()

    if not resolved.is_relative_to(resolved_base):
        raise PathSecurityError(f"Path {path} escapes base directory {base_dir}")

    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file extension: {resolved.suffix}")

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    size_mb = resolved.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise PathSecurityError(
            f"File {resolved} is {size_mb:.1f}MB, exceeds limit of {settings.max_file_size_mb}MB"
        )

    return resolved


def _is_blocked_address(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True

    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def validate_url(url: str) -> str:
    """Reject non-http(s) schemes and any URL whose host resolves to a
    private/loopback/link-local/reserved address (SSRF guard)."""
    parts = urlsplit(url)

    if parts.scheme not in ALLOWED_URL_SCHEMES:
        raise UrlSecurityError(f"Unsupported URL scheme: {parts.scheme!r}")

    if not parts.hostname:
        raise UrlSecurityError(f"URL has no host: {url}")

    if _is_blocked_address(parts.hostname):
        raise UrlSecurityError(f"URL host resolves to a disallowed address: {parts.hostname}")

    return url


def _extract_pdf_markdown(doc, source: str) -> list[Document]:
    """Converts the whole PDF to markdown (pymupdf4llm reconstructs headings
    from font size/weight) so chunk_documents' header-aware splitter can be
    reused instead of a separate PDF-specific one. A PDF with no extractable
    text anywhere (e.g. fully scanned) produces no Document -- same "don't
    index empty content" rule as before, now at whole-document granularity
    rather than per-page, since no code reads per-page metadata. Shared by
    load_pdf (local file) and load_url (PDF served directly over HTTP)."""
    import pymupdf4llm

    text = clean_text(pymupdf4llm.to_markdown(doc))
    if not text.strip():
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "source_doc_id": make_source_doc_id(source),
                "content_format": "markdown",
            },
        )
    ]


def load_pdf(path: Path) -> list[Document]:
    import pymupdf

    return _extract_pdf_markdown(pymupdf.open(str(path)), str(path))


def load_text(path: Path) -> list[Document]:
    """Loads a .txt/.md file as a single Document; empty/whitespace-only
    files produce no Document. .md files are tagged content_format="markdown"
    so chunk_documents routes them through header-aware splitting."""
    text = clean_text(path.read_text(encoding="utf-8"))
    if not text.strip():
        return []
    metadata = {"source": str(path), "source_doc_id": make_source_doc_id(str(path))}
    if path.suffix.lower() == ".md":
        metadata["content_format"] = "markdown"
    return [Document(page_content=text, metadata=metadata)]


def _docx_heading_level(style_name: str) -> int | None:
    # "Heading 1".."Heading 9" from add_heading(level=1..9) map to that same
    # level; "Title" from add_heading(level=0) is treated as a level-1
    # heading. Any other/custom style (including "Normal") is body text.
    if style_name == "Title":
        return 1
    suffix = style_name.removeprefix("Heading ")
    return int(suffix) if suffix != style_name and suffix.isdigit() else None


def load_docx(path: Path) -> list[Document]:
    """Loads a .docx file as one Document per heading section, with each
    heading rewritten as a markdown header line (content_format="markdown")
    so chunk_documents routes it through the same header-aware splitting as
    .md/PDF sources -- a chunk downstream then keeps its section heading
    even when the section itself is long enough to be sub-split."""
    import docx

    document = docx.Document(str(path))
    source = str(path)
    source_doc_id = make_source_doc_id(source)

    groups: list[list[str]] = [[]]
    for paragraph in document.paragraphs:
        level = _docx_heading_level(paragraph.style.name)
        if level is not None and groups[-1]:
            groups.append([])
        text = f"{'#' * level} {paragraph.text}" if level is not None else paragraph.text
        groups[-1].append(text)

    docs = []
    for group in groups:
        text = clean_text("\n".join(group))
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": source, "source_doc_id": source_doc_id, "content_format": "markdown"},
            )
        )
    return docs


_HTML_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _html_to_markdown(soup) -> tuple[str, bool]:
    """Walks the parsed tree in document order, rewriting heading tags as
    markdown header lines (capped at level 3 to match
    MARKDOWN_HEADERS_TO_SPLIT_ON in index.py) so chunk_documents' existing
    header-aware splitter can keep a heading with its section. Returns
    (markdown_text, found_any_heading) -- walking NavigableStrings rather
    than tag.get_text() on each heading/paragraph avoids double-counting
    text from nested inline tags (e.g. <h2><b>Policy</b></h2>)."""
    from bs4 import NavigableString, Tag

    lines = []
    has_heading = False
    for node in soup.descendants:
        if isinstance(node, Tag) and node.name in _HTML_HEADING_LEVELS:
            # separator=" ": a heading split across inline tags (e.g.
            # "<b>Refund</b> Policy") would otherwise lose the word
            # boundary between fragments (get_text(strip=True) alone
            # strips each fragment before joining them with nothing).
            heading_text = node.get_text(separator=" ", strip=True)
            if heading_text:
                level = min(_HTML_HEADING_LEVELS[node.name], 3)
                lines.append(f"{'#' * level} {heading_text}")
                has_heading = True
        elif isinstance(node, NavigableString):
            if node.find_parent(list(_HTML_HEADING_LEVELS) + ["script", "style"]) is not None:
                continue
            text = str(node).strip()
            if text:
                lines.append(text)
    return "\n".join(lines), has_heading


def load_url(url: str, _redirects_followed: int = 0) -> list[Document]:
    """Fetches a URL and extracts its content. PDFs served directly over
    HTTP (common for whitepapers, often with no .pdf suffix in the URL --
    e.g. arxiv.org/pdf/<id>) are detected by Content-Type and routed through
    the same page extraction as load_pdf, rather than decoded as HTML text.
    Each redirect hop is re-validated through validate_url (an
    attacker-controlled redirect target is exactly as dangerous as the
    original URL)."""
    import pymupdf
    from bs4 import BeautifulSoup

    settings = get_settings()
    validate_url(url)

    response = requests.get(url, timeout=settings.url_fetch_timeout_seconds, allow_redirects=False, stream=True)

    if response.is_redirect or response.is_permanent_redirect:
        if _redirects_followed >= MAX_REDIRECTS:
            raise UrlSecurityError(f"Too many redirects fetching {url}")
        location = response.headers.get("Location")
        if not location:
            raise UrlSecurityError(f"Redirect from {url} had no Location header")
        return load_url(location, _redirects_followed=_redirects_followed + 1)

    max_bytes = settings.max_url_fetch_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=8192):
        total += len(chunk)
        if total > max_bytes:
            raise UrlSecurityError(f"Response from {url} exceeded {settings.max_url_fetch_size_mb}MB limit")
        chunks.append(chunk)

    content = b"".join(chunks)
    content_type = response.headers.get("Content-Type", "")
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return _extract_pdf_markdown(pymupdf.open(stream=content, filetype="pdf"), url)

    html = content.decode(response.encoding or "utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    metadata = {"source": url, "source_doc_id": make_source_doc_id(url), "title": title}

    markdown_text, has_heading = _html_to_markdown(soup)
    if has_heading:
        text = clean_text(markdown_text)
        metadata["content_format"] = "markdown"
    else:
        text = clean_text(soup.get_text(separator="\n", strip=True))

    return [Document(page_content=text, metadata=metadata)]


def load_path(path: Path) -> list[Document]:
    """Routes an already-validated local path to the loader for its extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    raise UnsupportedFileTypeError(f"Unsupported file extension: {suffix}")


def discover_sources(docs_dir: Path) -> list[Path]:
    """Finds every file under docs_dir with an allowed extension, for bulk re-indexing."""
    return sorted(
        path for path in docs_dir.rglob("*") if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def validate_source(source: str) -> None:
    """Runs dispatch()'s security gate without loading any content -- lets
    a caller reject a bad source synchronously (e.g. before enqueueing an
    async ingest job) without doing any real I/O."""
    parts = urlsplit(source)
    if parts.scheme in ALLOWED_URL_SCHEMES:
        validate_url(source)
        return

    settings = get_settings()
    validate_local_path(Path(source), settings.docs_dir)


def dispatch(source: str) -> list[Document]:
    """Routes a user-supplied source string to the right loader, applying
    the relevant security gate first."""
    parts = urlsplit(source)
    if parts.scheme in ALLOWED_URL_SCHEMES:
        return load_url(source)

    settings = get_settings()
    resolved = validate_local_path(Path(source), settings.docs_dir)
    return load_path(resolved)


def dispatch_all_under_docs_dir(settings: Settings) -> tuple[list[Document], int]:
    """Discovers and dispatches every source under docs_dir; shared by the
    CLI's bulk `index` command and the async worker's "index" job kind."""
    paths = discover_sources(settings.docs_dir)
    docs: list[Document] = []
    for path in paths:
        docs.extend(dispatch(str(path.relative_to(settings.docs_dir))))
    return docs, len(paths)
