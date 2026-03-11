"""
Classical Corpus Extractor (§2.1)

Scrapers and processors for pre-1930 classical texts from:
  - Project Gutenberg (public domain literature)
  - Internet Archive (Sacred Books of the East, etc.)
  - Hanover Historical Texts (Presocratics, colonial records)
  - Perseus Digital Library (Greek/Latin corpus)

All output is JSONL: {"text": ..., "lang": ..., "date": ..., "source": ...}
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from aletheia.data.quarantine import quarantine_check, log_audit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Unicode NFC normalization, whitespace cleanup, diacritics preserved."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_modern_annotations(text: str) -> str:
    """Remove modern editorial footnotes, ISBNs, and boilerplate."""
    # Strip ISBN references
    text = re.sub(r"ISBN[\s:\-]*[\dX\-]+", "", text)
    # Strip [Editor's note: ...] blocks
    text = re.sub(r"\[Editor'?s?\s+[Nn]ote[:\s].*?\]", "", text, flags=re.DOTALL)
    # Strip "Project Gutenberg" boilerplate headers/footers
    for marker in [
        "*** START OF THE PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]:
        idx = text.find(marker)
        if idx != -1:
            # Find end of the marker line
            end = text.find("\n", idx)
            if marker.startswith("*** START"):
                text = text[end + 1:] if end != -1 else text
            elif marker.startswith("*** END"):
                text = text[:idx]
    return text.strip()


def detect_language(text: str) -> str:
    """Detect language of text. Falls back to 'unknown'."""
    try:
        from langdetect import detect
        return detect(text[:2000])
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Gutenberg scraper
# ---------------------------------------------------------------------------

def scrape_gutenberg(
    output_dir: str | Path,
    max_books: int = 15000,
    max_year: int = 1930,
) -> int:
    """Download pre-1930 texts from Project Gutenberg.

    Uses the Gutenberg mirror API to find books with publication dates
    at or before the cutoff year.

    Args:
        output_dir: Directory for output JSONL files.
        max_books:  Maximum number of books to download.
        max_year:   Temporal cutoff (default 1930).

    Returns:
        Number of documents successfully ingested.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gutenberg.jsonl"

    count = 0
    catalog_url = "https://gutendex.com/books/"
    params = {"copyright": "false", "languages": "en,la,el,ar,sa,fa,fr,de"}

    logger.info(f"Scraping Project Gutenberg (max {max_books} books, ≤{max_year})...")

    try:
        while catalog_url and count < max_books:
            resp = requests.get(catalog_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for book in data.get("results", []):
                if count >= max_books:
                    break

                # Extract year from author birth/death or title metadata
                authors = book.get("authors", [])
                year = None
                for author in authors:
                    death = author.get("death_year")
                    if death and death <= max_year:
                        year = death
                        break

                # Get plain text format
                formats = book.get("formats", {})
                text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")
                if not text_url:
                    continue

                try:
                    text_resp = requests.get(text_url, timeout=120)  # text downloads can be slow
                    text_resp.raise_for_status()
                    raw_text = text_resp.text
                except Exception as e:
                    logger.debug(f"Failed to download book {book.get('id')}: {e}")
                    continue

                # Clean and validate
                text = strip_modern_annotations(normalize_text(raw_text))
                if len(text) < 500:
                    continue

                lang = detect_language(text)
                doc = {
                    "text": text,
                    "lang": lang,
                    "date": year,
                    "source": f"gutenberg:{book.get('id')}",
                    "title": book.get("title", ""),
                }

                passed, reason = quarantine_check(doc, corpus_type="classical")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")

                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1

                    if count % 100 == 0:
                        logger.info(f"  Ingested {count} books...")

            # Pagination
            catalog_url = data.get("next")
            params = {}  # next URL already has params

    except Exception as e:
        logger.error(f"Gutenberg scraping error: {e}")

    logger.info(f"Gutenberg complete: {count} documents ingested.")
    return count


# ---------------------------------------------------------------------------
# Ganjoor (Persian Poetry - Rumi)
# ---------------------------------------------------------------------------

def scrape_ganjoor(
    output_dir: str | Path,
    max_verses: int = 5000,
) -> int:
    """Scrape Rumi's Mathnawi from Ganjoor.net (Original Persian)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ganjoor_rumi.jsonl"
    
    # Rumi's Mathnawi base
    base_url = "https://ganjoor.net/moulavi/masnavi"
    count = 0
    
    logger.info(f"Scraping Rumi from Ganjoor (max {max_verses} verses)...")
    
    try:
        # Iterate through the 6 Daftars
        for d in range(1, 7):
            if count >= max_verses: break
            
            # Note: We'd typically crawl the index, but here we target the first few sections
            # for the smoke test/initial ingestion.
            for s in range(1, 100):
                if count >= max_verses: break
                
                url = f"{base_url}/daftar{d}/sh{s}"
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 404: break
                    resp.raise_for_status()
                    
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Verses are in div class="b"
                    verses = soup.select("div.b")
                    if not verses: continue
                    
                    full_text = ""
                    for v in verses:
                        # misras are m1 and m2
                        m1 = v.select_one("div.m1")
                        m2 = v.select_one("div.m2")
                        if m1 and m2:
                            full_text += f"{m1.get_text(strip=True)} | {m2.get_text(strip=True)}\n"
                    
                    if not full_text: continue
                    
                    doc = {
                        "text": normalize_text(full_text),
                        "lang": "fa",
                        "date": 1273,  # Rumi died in 1273
                        "source": url,
                        "title": soup.title.get_text(strip=True) if soup.title else f"Mathnawi D{d} S{s}",
                    }
                    
                    passed, reason = quarantine_check(doc, corpus_type="classical")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += len(verses)
                        
                except Exception as e:
                    logger.debug(f"Ganjoor individual page error ({url}): {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Ganjoor scraping error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Perseus (Ancient Greek Philosophy - Socrates/Plato)
# ---------------------------------------------------------------------------

def scrape_perseus(
    output_dir: str | Path,
    max_docs: int = 10,
) -> int:
    """Scrape Socratic dialogues from Perseus (Ancient Greek TEI XML)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "perseus_greek.jsonl"
    
    # IDs for Republic, Apology, Crito (original Greek)
    work_ids = ["1999.01.0167", "1999.01.0169", "1999.01.0170"]
    count = 0
    
    logger.info(f"Scraping Perseus Ancient Greek (max {max_docs} works)...")
    
    try:
        for wid in work_ids:
            if count >= max_docs: break
            
            # Using the xmlchunk stable URL
            url = f"https://www.perseus.tufts.edu/hopper/xmlchunk?doc=Perseus:text:{wid}"
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                
                # Extract text from TEI XML
                soup = BeautifulSoup(resp.content, "xml")
                # Target the body or specific text segments
                body = soup.find("body") or soup.find("text")
                if not body: continue
                
                text = normalize_text(body.get_text())
                
                if len(text) < 500: continue
                
                doc = {
                    "text": text,
                    "lang": "el", # polytonic greek detected as greek
                    "date": -399, # Socratic era
                    "source": url,
                    "title": f"Perseus:{wid}",
                }
                
                passed, reason = quarantine_check(doc, corpus_type="classical")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                
                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                    
            except Exception as e:
                logger.debug(f"Perseus error ({wid}): {e}")
                continue
                
    except Exception as e:
        logger.error(f"Perseus scraping error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Hanover Historical Texts scraper
# ---------------------------------------------------------------------------

def scrape_hanover(
    output_dir: str | Path,
) -> int:
    """Scrape Hanover Historical Texts Collection.

    Downloads pages from history.hanover.edu/project.html and follows
    links to individual text documents.

    Args:
        output_dir: Directory for output JSONL files.

    Returns:
        Number of documents successfully ingested.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hanover.jsonl"

    base_url = "https://history.hanover.edu"
    index_url = f"{base_url}/project.html"
    count = 0

    logger.info("Scraping Hanover Historical Texts...")

    try:
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if not href.endswith((".html", ".htm", ".txt")):
                continue
            if href.startswith("http"):
                full_url = href
            else:
                full_url = f"{base_url}/{href.lstrip('/')}"

            try:
                page_resp = requests.get(full_url, timeout=30)
                page_resp.raise_for_status()
                page_soup = BeautifulSoup(page_resp.text, "html.parser")

                # Extract text content
                for tag in page_soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = normalize_text(page_soup.get_text(separator="\n"))

                if len(text) < 300:
                    continue

                lang = detect_language(text)
                doc = {
                    "text": text,
                    "lang": lang,
                    "date": None,  # dates not reliably available in metadata
                    "source": f"hanover:{href}",
                    "title": link.get_text(strip=True),
                }

                passed, reason = quarantine_check(doc, corpus_type="classical")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")

                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1

            except Exception as e:
                logger.debug(f"Failed to fetch {full_url}: {e}")
                continue

    except Exception as e:
        logger.error(f"Hanover scraping error: {e}")

    logger.info(f"Hanover complete: {count} documents ingested.")
    return count


# ---------------------------------------------------------------------------
# Yoruba Oral Literature (Tribal Traditions)
# ---------------------------------------------------------------------------

def scrape_yoruba_oral(
    output_dir: str | Path,
    max_count: int = 10,
) -> int:
    """Scrape Yoruba folktale transcriptions from yoruba-oral-literature.org."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "yoruba_oral.jsonl"
    
    # Target the transcriptions/folktales section
    url = "https://yoruba-oral-literature.org/folktales/"
    count = 0
    
    logger.info(f"Scraping Yoruba oral traditions (max {max_count} documents)...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Scrape transcriptions from tables
        # The Yoruba text is usually in the first column of the transcription tables
        rows = soup.select("table tr") or soup.find_all("tr")
        current_text = ""
        
        for row in rows:
            if count >= max_count: break
            
            cells = row.find_all("td")
            if len(cells) >= 1:
                # First cell is Yoruba
                yoruba_line = cells[0].get_text(strip=True)
                if yoruba_line:
                    current_text += yoruba_line + "\n"
                    
            # If we've collected enough text for a "document"
            if len(current_text) > 1000:
                doc = {
                    "text": normalize_text(current_text),
                    "lang": "yo", 
                    "date": None,
                    "source": url,
                    "title": "Yoruba Folktale Transcription",
                }
                
                passed, reason = quarantine_check(doc, corpus_type="classical")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                
                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                current_text = ""
                
    except Exception as e:
        logger.error(f"Yoruba scraping error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def scrape_all_classical(
    output_dir: str | Path = "data/classical",
    max_gutenberg: int = 15000,
) -> dict[str, int]:
    """Run all classical corpus scrapers.
 
    Returns dict mapping source name → number of documents ingested.
    """
    output_dir = Path(output_dir)
    results = {}
 
    results["gutenberg"] = scrape_gutenberg(output_dir, max_books=max_gutenberg)
    results["hanover"] = scrape_hanover(output_dir)
    results["ganjoor"] = scrape_ganjoor(output_dir, max_verses=1000)
    results["perseus"] = scrape_perseus(output_dir)
    results["yoruba_oral"] = scrape_yoruba_oral(output_dir)
 
    total = sum(results.values())
    logger.info(f"Total classical corpus: {total} documents across {len(results)} sources.")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    scrape_all_classical()
