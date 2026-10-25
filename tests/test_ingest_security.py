import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_assistant.ingest import (
    PathSecurityError,
    UnsupportedFileTypeError,
    UrlSecurityError,
    make_source_doc_id,
    validate_local_path,
    validate_url,
)


def _patch_settings(monkeypatch, max_file_size_mb: int = 25) -> None:
    monkeypatch.setattr(
        "research_assistant.ingest.get_settings",
        lambda: SimpleNamespace(max_file_size_mb=max_file_size_mb),
    )


def test_validate_local_path_accepts_file_within_base_dir(tmp_path, monkeypatch):
    _patch_settings(monkeypatch)
    base_dir = tmp_path / "docs"
    base_dir.mkdir()
    doc = base_dir / "paper.pdf"
    doc.write_text("content")

    assert validate_local_path(Path("paper.pdf"), base_dir) == doc.resolve()


def test_validate_local_path_rejects_path_escaping_base_dir(tmp_path, monkeypatch):
    _patch_settings(monkeypatch)
    base_dir = tmp_path / "docs"
    base_dir.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_text("content")

    with pytest.raises(PathSecurityError):
        validate_local_path(Path("../outside.pdf"), base_dir)


def test_validate_local_path_rejects_unsupported_extension(tmp_path, monkeypatch):
    _patch_settings(monkeypatch)
    base_dir = tmp_path / "docs"
    base_dir.mkdir()
    doc = base_dir / "script.exe"
    doc.write_text("content")

    with pytest.raises(UnsupportedFileTypeError):
        validate_local_path(Path("script.exe"), base_dir)


def test_validate_local_path_rejects_missing_file(tmp_path, monkeypatch):
    _patch_settings(monkeypatch)
    base_dir = tmp_path / "docs"
    base_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        validate_local_path(Path("missing.pdf"), base_dir)


def test_validate_local_path_rejects_oversize_file(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, max_file_size_mb=0)
    base_dir = tmp_path / "docs"
    base_dir.mkdir()
    doc = base_dir / "paper.pdf"
    doc.write_text("this file is bigger than a zero-mb limit")

    with pytest.raises(PathSecurityError):
        validate_local_path(Path("paper.pdf"), base_dir)


def test_validate_url_rejects_non_http_scheme():
    with pytest.raises(UrlSecurityError):
        validate_url("ftp://example.com/file")


def test_validate_url_rejects_loopback_host(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )

    with pytest.raises(UrlSecurityError):
        validate_url("http://evil.internal/")


def test_validate_url_rejects_unresolvable_host(monkeypatch):
    def fake_getaddrinfo(host, port):
        raise socket.gaierror("unresolvable")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(UrlSecurityError):
        validate_url("http://does-not-exist.invalid/")


def test_validate_url_allows_public_host(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )

    assert validate_url("http://example.com/") == "http://example.com/"


def test_make_source_doc_id_is_deterministic():
    assert make_source_doc_id("data/documents/paper.pdf") == make_source_doc_id(
        "data/documents/paper.pdf"
    )


def test_make_source_doc_id_differs_for_different_sources():
    assert make_source_doc_id("a.pdf") != make_source_doc_id("b.pdf")


def test_make_source_doc_id_is_short_hex():
    doc_id = make_source_doc_id("a.pdf")
    assert len(doc_id) == 12
    assert all(c in "0123456789abcdef" for c in doc_id)
