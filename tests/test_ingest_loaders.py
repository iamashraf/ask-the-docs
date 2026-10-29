from ask_the_docs.ingest import load_pdf, load_text


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakePdfReader:
    def __init__(self, path):
        self.pages = [
            FakePage("Page one content."),
            FakePage("   "),
            FakePage("Page three content."),
        ]


class AllBlankPdfReader:
    def __init__(self, path):
        self.pages = [FakePage(""), FakePage("   \n\t")]


def test_load_pdf_returns_document_per_nonempty_page(tmp_path, monkeypatch):
    monkeypatch.setattr("pypdf.PdfReader", FakePdfReader)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    docs = load_pdf(pdf_path)

    assert len(docs) == 2
    assert docs[0].page_content == "Page one content."
    assert docs[0].metadata["source"] == str(pdf_path)
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 3
    assert docs[0].metadata["source_doc_id"] == docs[1].metadata["source_doc_id"]


def test_load_pdf_skips_pages_with_only_whitespace_text(tmp_path, monkeypatch):
    monkeypatch.setattr("pypdf.PdfReader", AllBlankPdfReader)
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

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
