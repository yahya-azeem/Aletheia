"""Tests for the epistemological quarantine validator."""
from aletheia.data.quarantine import (
    validate_date,
    validate_source,
    detect_secondary_content,
    quarantine_check,
)


def test_date_before_1930():
    assert validate_date(1850) is True
    assert validate_date("1920") is True
    assert validate_date("1930") is True


def test_date_after_1930():
    assert validate_date(1950) is False
    assert validate_date("2023") is False
    assert validate_date("2024-01-15") is False


def test_date_unknown():
    assert validate_date(None) is True


def test_approved_source():
    assert validate_source("https://www.courtlistener.com/api/v4/doc/123") is True
    assert validate_source("https://case.law/v1/cases/123") is True
    assert validate_source("https://zenodo.org/record/12345") is True


def test_unapproved_source():
    assert validate_source("https://en.wikipedia.org/wiki/Test") is False
    assert validate_source("https://www.cnn.com/article/test") is False
    assert validate_source("https://nytimes.com/2024") is False


def test_secondary_content_detection():
    clean = "The court hereby finds the defendant guilty of the charges."
    assert detect_secondary_content(clean) is False

    editorial = (
        "According to analysts, experts believe this represents a major shift. "
        "The editor's note explains that Wikipedia sources confirm the trend."
    )
    assert detect_secondary_content(editorial, threshold=3) is True


def test_quarantine_classical_pass():
    doc = {"text": "In the beginning was the Word.", "date": 1611, "source": "kjv", "lang": "en"}
    passed, reason = quarantine_check(doc, corpus_type="classical")
    assert passed is True


def test_quarantine_classical_fail_date():
    doc = {"text": "Modern synthesis.", "date": 1975, "source": "textbook", "lang": "en"}
    passed, reason = quarantine_check(doc, corpus_type="classical")
    assert passed is False
    assert "1930" in reason


def test_quarantine_modern_fail_source():
    doc = {"text": "Some text.", "date": 2020, "source": "https://cnn.com/story", "lang": "en"}
    passed, reason = quarantine_check(doc, corpus_type="modern")
    assert passed is False
    assert "not on approved list" in reason


def test_quarantine_modern_pass():
    doc = {"text": "Legal filing text.", "date": 2020, "source": "https://courtlistener.com/doc", "lang": "en"}
    passed, reason = quarantine_check(doc, corpus_type="modern")
    assert passed is True
