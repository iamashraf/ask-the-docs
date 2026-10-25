import socket
from types import SimpleNamespace

import docx as docx_module
import requests

from research_assistant.ingest import discover_sources, dispatch, load_docx, load_url

SENTINEL = [object()]


class FakeResponse:
    def __init__(self, content: bytes, encoding=None, is_redirect=False, headers=None):
        self._content = content
        self.encoding = encoding
        self.is_redirect = is_redirect
        self.is_permanent_redirect = False
        self.headers = headers or {}

    def iter_content(self, chunk_size):
        yield self._content


def _allow_public_host(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )


def _patch_url_settings(monkeypatch, max_url_fetch_size_mb: int = 10):
    monkeypatch.setattr(
        "research_assistant.ingest.get_settings",
        lambda: SimpleNamespace(
            max_url_fetch_size_mb=max_url_fetch_size_mb, url_fetch_timeout_seconds=10
        ),
    )


def test_load_docx_reads_paragraphs(tmp_path):
    docx_path = tmp_path / "paper.docx"
    document = docx_module.Document()
    document.add_paragraph("Hello docx world.")
    document.save(str(docx_path))

    docs = load_docx(docx_path)

    assert len(docs) == 1
    assert "Hello docx world." in docs[0].page_content
    assert docs[0].metadata["source"] == str(docx_path)
    assert "source_doc_id" in docs[0].metadata


def test_load_docx_returns_empty_list_for_blank_document(tmp_path):
    docx_path = tmp_path / "blank.docx"
    docx_module.Document().save(str(docx_path))

    assert load_docx(docx_path) == []


def test_load_url_fetches_and_parses_html(monkeypatch):
    _allow_public_host(monkeypatch)
    _patch_url_settings(monkeypatch)
    html = b"<html><head><title>Hi</title></head><body><p>Hello world</p></body></html>"
    monkeypatch.setattr(requests, "get", lambda url, timeout, allow_redirects, stream: FakeResponse(html))

    docs = load_url("http://example.com/page")

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["source"] == "http://example.com/page"
    assert docs[0].metadata["title"] == "Hi"


def test_load_url_follows_redirect_and_revalidates_target(monkeypatch):
    _allow_public_host(monkeypatch)
    _patch_url_settings(monkeypatch)
    responses = {
        "http://example.com/old": FakeResponse(
            b"", is_redirect=True, headers={"Location": "http://example.com/new"}
        ),
        "http://example.com/new": FakeResponse(b"<html><body>New page</body></html>"),
    }
    monkeypatch.setattr(
        requests, "get", lambda url, timeout, allow_redirects, stream: responses[url]
    )

    docs = load_url("http://example.com/old")

    assert "New page" in docs[0].page_content
    assert docs[0].metadata["source"] == "http://example.com/new"


def test_load_url_rejects_too_many_redirects(monkeypatch):
    _allow_public_host(monkeypatch)
    _patch_url_settings(monkeypatch)
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, timeout, allow_redirects, stream: FakeResponse(
            b"", is_redirect=True, headers={"Location": "http://example.com/next"}
        ),
    )

    import pytest

    from research_assistant.ingest import UrlSecurityError

    with pytest.raises(UrlSecurityError):
        load_url("http://example.com/start")


def test_load_url_enforces_size_limit(monkeypatch):
    _allow_public_host(monkeypatch)
    _patch_url_settings(monkeypatch, max_url_fetch_size_mb=0)
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, timeout, allow_redirects, stream: FakeResponse(b"more than zero bytes"),
    )

    import pytest

    from research_assistant.ingest import UrlSecurityError

    with pytest.raises(UrlSecurityError):
        load_url("http://example.com/big")


def test_dispatch_routes_http_url_to_load_url(monkeypatch):
    monkeypatch.setattr("research_assistant.ingest.load_url", lambda url: SENTINEL)
    assert dispatch("http://example.com/paper") == SENTINEL


def test_dispatch_routes_local_path_to_load_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "research_assistant.ingest.get_settings",
        lambda: SimpleNamespace(docs_dir=tmp_path, max_file_size_mb=25),
    )
    doc_path = tmp_path / "paper.pdf"
    doc_path.write_bytes(b"%PDF fake")
    monkeypatch.setattr("research_assistant.ingest.load_path", lambda path: SENTINEL)

    assert dispatch("paper.pdf") == SENTINEL


def test_discover_sources_finds_allowed_extensions_only(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"x")
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "ignore.exe").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("x")

    found = discover_sources(tmp_path)

    assert found == sorted(
        [tmp_path / "paper.pdf", tmp_path / "notes.md", sub / "nested.txt"]
    )
