import pymupdf

from ask_the_docs.ingest import load_pdf, load_text


def _make_pdf_with_heading(heading: str, body: str) -> bytes:
    # fontsize 18 vs 11 is what pymupdf4llm's heading reconstruction keys
    # on (larger text -> markdown "#" heading).
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), heading, fontsize=18, fontname="helv")
    page.insert_text((72, 100), body, fontsize=11, fontname="helv")
    return doc.tobytes()


def _make_scanned_pdf() -> bytes:
    # No text objects at all -- simulates a scanned/image-only page.
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 400, 400), fill=(0, 0, 0))
    return doc.tobytes()


def test_load_pdf_keeps_heading_text_with_its_section(tmp_path):
    # Asserting the heading text is merely present isn't enough -- plain
    # text extraction would already include it unstructured. The real
    # signal is markdown heading syntax plus the content_format tag that
    # routes this document through chunk_documents' header-aware splitter.
    pdf_path = tmp_path / "policy.pdf"
    pdf_path.write_bytes(
        _make_pdf_with_heading("Refund Policy", "Customers may request a refund within 30 days.")
    )

    docs = load_pdf(pdf_path)

    assert any("# Refund Policy" in doc.page_content for doc in docs)
    assert any("Customers may request a refund within 30 days." in doc.page_content for doc in docs)
    assert all(doc.metadata["content_format"] == "markdown" for doc in docs)
    assert all(doc.metadata["source"] == str(pdf_path) for doc in docs)
    assert all("source_doc_id" in doc.metadata for doc in docs)


def test_load_pdf_returns_empty_list_for_scanned_pdf_with_no_text(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(_make_scanned_pdf())

    assert load_pdf(pdf_path) == []


def test_load_text_reads_txt_file(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello world")

    docs = load_text(txt_path)

    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == str(txt_path)
    assert "source_doc_id" in docs[0].metadata


def test_load_text_reads_markdown_file(tmp_path):
    md_path = tmp_path / "readme.md"
    md_path.write_text("# Title\n\nbody")

    docs = load_text(md_path)

    assert docs[0].page_content == "# Title\n\nbody"


def test_load_text_returns_empty_list_for_blank_file(tmp_path):
    txt_path = tmp_path / "empty.txt"
    txt_path.write_text("   \n  ")

    assert load_text(txt_path) == []


def test_load_text_cleans_messy_whitespace(tmp_path):
    txt_path = tmp_path / "messy.txt"
    txt_path.write_text("Para one.   \n\n\n\n\nPara   two.")

    docs = load_text(txt_path)

    assert docs[0].page_content == "Para one.\n\nPara two."
