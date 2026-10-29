from ask_the_docs.ingest import clean_text


def test_clean_text_collapses_whitespace_and_blank_lines():
    messy = "Para one.   \n\n\n\n\nPara   two.\t\tindented."

    assert clean_text(messy) == "Para one.\n\nPara two. indented."


def test_clean_text_is_noop_on_already_clean_text():
    clean = "Para one.\n\nPara two."

    assert clean_text(clean) == clean


def test_clean_text_normalizes_unicode():
    assert clean_text("ﬁnal report") == "final report"


def test_clean_text_strips_control_chars_but_keeps_newlines():
    assert clean_text("Before\x00After\nNext line") == "BeforeAfter\nNext line"


def test_clean_text_empty_string_returns_empty_string():
    assert clean_text("") == ""


def test_clean_text_whitespace_only_returns_empty_string():
    assert clean_text("   \n\t  \n  ") == ""
