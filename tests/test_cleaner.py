from ingestion.cleaner import TextCleaner


def test_removes_page_numbers():
    cleaner = TextCleaner()
    text = "Introduction\nSome content here.\n3\nMore content."
    cleaned = cleaner.clean(text)
    assert "\n3\n" not in cleaned


def test_collapses_multiple_blank_lines():
    cleaner = TextCleaner()
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    cleaned = cleaner.clean(text)
    assert "\n\n\n" not in cleaned


def test_removes_repeated_headers():
    cleaner = TextCleaner(header_footer_min_repeats=2)
    text = "RUNNING HEADER\nBody text A.\nRUNNING HEADER\nBody text B."
    cleaned = cleaner.clean(text)
    assert cleaned.count("RUNNING HEADER") == 0


def test_preserves_paragraph_content():
    cleaner = TextCleaner()
    text = "This is a normal sentence that should survive cleaning intact."
    cleaned = cleaner.clean(text)
    assert "This is a normal sentence that should survive cleaning intact." in cleaned
