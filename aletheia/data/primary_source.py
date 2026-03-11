"""
Primary-Source Modernity Ingestion (§2.2)

API consumers for post-1930 primary sources:
  - RECAP / Free Law Project (CourtListener API)
  - Harvard Caselaw Access Project (CAP API)
  - Zenodo (CERN open research repository)
  - Data.gov (federal CKAN datasets)
  - DNSA (Digital National Security Archive) — DEFERRED/STUBBED

All output is JSONL: {"text": ..., "lang": ..., "date": ..., "source": ...}
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup
import io

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import internetarchive
except ImportError:
    internetarchive = None

from aletheia.data.quarantine import quarantine_check, log_audit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CourtListener / RECAP (Free Law Project)
# ---------------------------------------------------------------------------

COURTLISTENER_TOKEN = "1d18c25d6c297611e47529994ad5002797e09a86"

def ingest_courtlistener(
    output_dir: str | Path,
    api_key: str | None = COURTLISTENER_TOKEN,
    max_docs: int = 5000,
) -> int:
    """Ingest federal court opinions from CourtListener API.

    Args:
        output_dir: Directory for output JSONL.
        api_key:    CourtListener API key (optional but recommended).
        max_docs:   Maximum documents to ingest.

    Returns:
        Number of documents ingested.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "courtlistener.jsonl"

    base_url = "https://www.courtlistener.com/api/rest/v4/opinions/"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    count = 0
    page = 1
    logger.info(f"Ingesting CourtListener opinions (max {max_docs})...")

    try:
        while count < max_docs:
            params = {
                "page": page,
                "page_size": 20,
                "order_by": "-date_created",
            }
            resp = requests.get(base_url, headers=headers, params=params, timeout=60)

            if resp.status_code == 429:
                logger.warning("Rate limited. Waiting 60s...")
                time.sleep(60)
                continue

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            if not results:
                break

            for opinion in results:
                if count >= max_docs:
                    break

                text = opinion.get("plain_text") or opinion.get("html", "")
                if not text or len(text) < 200:
                    continue

                doc = {
                    "text": text[:500000],  # cap at 500k chars
                    "lang": "en",
                    "date": opinion.get("date_created", ""),
                    "source": f"https://courtlistener.com/opinion/{opinion.get('id', '')}",
                    "case_name": opinion.get("case_name", ""),
                }

                passed, reason = quarantine_check(doc, corpus_type="modern")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")

                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1

            page += 1
            time.sleep(0.5)  # rate courtesy

    except Exception as e:
        logger.error(f"CourtListener error: {e}")

    logger.info(f"CourtListener complete: {count} opinions ingested.")
    return count


# ---------------------------------------------------------------------------
# Harvard Caselaw Access Project (CAP)
# ---------------------------------------------------------------------------

def ingest_cap(
    output_dir: str | Path,
    api_key: str | None = None,
    max_docs: int = 5000,
) -> int:
    """Ingest cases from Harvard Caselaw Access Project API.

    Args:
        output_dir: Directory for output JSONL.
        api_key:    CAP API key.
        max_docs:   Maximum documents to ingest.

    Returns:
        Number of documents ingested.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cap.jsonl"

    base_url = "https://api.case.law/v1/cases/"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    count = 0
    cursor = None
    logger.info(f"Ingesting CAP cases (max {max_docs})...")

    try:
        while count < max_docs:
            params = {"page_size": 100, "full_case": "true"}
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(base_url, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            if not results:
                break

            for case in results:
                if count >= max_docs:
                    break

                # Extract opinion text from casebody
                casebody = case.get("casebody", {})
                if isinstance(casebody, dict):
                    opinions = casebody.get("data", {}).get("opinions", [])
                    text = "\n\n".join(op.get("text", "") for op in opinions)
                else:
                    text = str(casebody)

                if len(text) < 200:
                    continue

                doc = {
                    "text": text[:500000],
                    "lang": "en",
                    "date": case.get("decision_date", ""),
                    "source": f"https://case.law/cases/{case.get('id', '')}",
                    "case_name": case.get("name", ""),
                }

                passed, reason = quarantine_check(doc, corpus_type="modern")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")

                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1

            cursor = data.get("next")
            if not cursor:
                break
            time.sleep(0.3)

    except Exception as e:
        logger.error(f"CAP error: {e}")

    logger.info(f"CAP complete: {count} cases ingested.")
    return count


# ---------------------------------------------------------------------------
# Zenodo (CERN open research)
# ---------------------------------------------------------------------------

def ingest_zenodo(
    output_dir: str | Path,
    max_docs: int = 1000,
    communities: list[str] | None = None,
) -> int:
    """Ingest research dataset metadata and descriptions from Zenodo.

    Args:
        output_dir:  Directory for output JSONL.
        max_docs:    Maximum records.
        communities: Optional Zenodo community filters.

    Returns:
        Number of records ingested.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "zenodo.jsonl"

    base_url = "https://zenodo.org/api/records"
    count = 0
    page = 1
    logger.info(f"Ingesting Zenodo records (max {max_docs})...")

    try:
        while count < max_docs:
            params = {
                "page": page,
                "size": 10,
                "q": "resource_type.type:dataset AND access_right:open",
            }
            if communities:
                params["communities"] = ",".join(communities)

            resp = requests.get(base_url, params=params, timeout=60)
            if resp.status_code == 429:
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()
            records = data.get("hits", {}).get("hits", [])

            if not records:
                break

            for record in records:
                if count >= max_docs:
                    break

                metadata = record.get("metadata", {})
                description = metadata.get("description", "")
                title = metadata.get("title", "")
                text = f"{title}\n\n{description}"

                if len(text) < 100:
                    continue

                doc = {
                    "text": text[:100000],
                    "lang": "en",
                    "date": metadata.get("publication_date", ""),
                    "source": f"https://zenodo.org/record/{record.get('id', '')}",
                    "title": title,
                }

                passed, reason = quarantine_check(doc, corpus_type="modern")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")

                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1

            page += 1
            time.sleep(1.0)

    except Exception as e:
        logger.error(f"Zenodo error: {e}")

    logger.info(f"Zenodo complete: {count} records ingested.")
    return count


# ---------------------------------------------------------------------------
# FBI Vault (vault.fbi.gov)
# ---------------------------------------------------------------------------

def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes."""
    if not pypdf:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages[:50]:  # limit to 50 pages
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.debug(f"PDF extraction error: {e}")
        return ""


def ingest_fbi_vault(
    output_dir: str | Path,
    max_docs: int = 20,
) -> int:
    """Scrape declassified records from the FBI Vault."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fbi_vault.jsonl"
    
    # Target: Jeffrey Epstein category
    url = "https://vault.fbi.gov/jeffrey-epstein"
    count = 0
    
    logger.info(f"Ingesting FBI Vault (target: Epstein, max {max_docs})...")
    
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Corrected FBI Vault selector
        links = soup.select("a.contenttype-file.url")
        for link in links:
            if count >= max_docs: break
            
            landing_url = link["href"]
            if not landing_url.startswith("http"):
                landing_url = "https://vault.fbi.gov" + landing_url
                
            # Direct PDF link transformation: /view -> /at_download/file
            pdf_url = landing_url.replace("/view", "/at_download/file")
            
            try:
                pdf_resp = requests.get(pdf_url, timeout=60)
                pdf_resp.raise_for_status()
                text = _extract_pdf_text(pdf_resp.content)
                
                if len(text) > 200:
                    doc = {
                        "text": text[:500000],
                        "lang": "en",
                        "date": None,
                        "source": pdf_url,
                        "title": link.get_text(strip=True),
                    }
                    passed, reason = quarantine_check(doc, corpus_type="modern")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
            except Exception as e:
                logger.debug(f"FBI individual file error ({pdf_url}): {e}")
                        
    except Exception as e:
        logger.error(f"FBI Vault error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# DOJ Epstein Library
# ---------------------------------------------------------------------------

def ingest_epstein_library(
    output_dir: str | Path,
    max_docs: int = 50,
) -> int:
    """Ingest exhibits from the official DOJ Epstein Library."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "epstein_library.jsonl"
    
    # Corrected DOJ URL
    url = "https://www.justice.gov/epstein/doj-disclosures/court-records-united-states-v-epstein-no-119-cr-00490-sdny-2019"
    count = 0
    
    logger.info(f"Ingesting DOJ Epstein Library (max {max_docs})...")
    
    try:
        # DOJ might require a session to bypass age gate, or direct PDF links might work
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Links to PDF exhibits
        links = soup.find_all("a", href=re.compile(r".*\.pdf$"))
        for link in links:
            if count >= max_docs: break
            
            pdf_url = link["href"]
            if not pdf_url.startswith("http"):
                pdf_url = "https://www.justice.gov" + pdf_url
                
            try:
                pdf_resp = requests.get(pdf_url, headers=headers, timeout=60)
                pdf_resp.raise_for_status()
                text = _extract_pdf_text(pdf_resp.content)
                
                if len(text) > 200:
                    doc = {
                        "text": text[:500000],
                        "lang": "en",
                        "date": "2024",
                        "source": pdf_url, 
                        "title": link.get_text(strip=True),
                    }
                    passed, reason = quarantine_check(doc, corpus_type="modern")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
            except Exception as e:
                logger.debug(f"DOJ individual file error ({pdf_url}): {e}")
                    
    except Exception as e:
        logger.error(f"DOJ Epstein error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# CIA CREST (cia.gov/readingroom)
# ---------------------------------------------------------------------------

def ingest_cia_crest(
    output_dir: str | Path,
    query: str = "Declassified",
    max_docs: int = 20,
) -> int:
    """Ingest declassified records from the CIA CREST Reading Room."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cia_crest.jsonl"
    
    # CIA CREST search URL
    search_url = f"https://www.cia.gov/readingroom/search/site/{query}"
    count = 0
    
    logger.info(f"Ingesting CIA CREST (query: {query}, max {max_docs})...")
    
    # Use a session to handle redirects and headers consistently
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    try:
        resp = session.get(search_url, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # CIA CREST fixed selector
        results = soup.select(".search-results li h3") or soup.select(".search-results .search-result")
        for res in results:
            if count >= max_docs: break
            
            link = res.find("a")
            if not link: continue
            
            doc_page_url = link["href"]
            if not doc_page_url.startswith("http"):
                doc_page_url = "https://www.cia.gov" + doc_page_url
                
            try:
                page_resp = session.get(doc_page_url, timeout=60)
                page_soup = BeautifulSoup(page_resp.text, "html.parser")
                pdf_link = page_soup.find("a", href=re.compile(r".*\.pdf$"))
                
                if pdf_link:
                    pdf_url = pdf_link["href"]
                    if not pdf_url.startswith("http"):
                        pdf_url = "https://www.cia.gov" + pdf_url
                        
                    pdf_resp = session.get(pdf_url, timeout=60)
                    pdf_resp.raise_for_status()
                    text = _extract_pdf_text(pdf_resp.content)
                    
                    if len(text) > 200:
                        doc = {
                            "text": text[:500000],
                            "lang": "en",
                            "date": None,
                            "source": pdf_url,
                            "title": link.get_text(strip=True),
                        }
                        passed, reason = quarantine_check(doc, corpus_type="modern")
                        log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                        
                        if passed:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            count += 1
            except Exception as e:
                logger.debug(f"CIA CREST individual file error ({doc_page_url}): {e}")
                
    except Exception as e:
        logger.error(f"CIA CREST error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# OSTI (Department of Energy)
# ---------------------------------------------------------------------------

def ingest_osti(
    output_dir: str | Path,
    max_docs: int = 50,
) -> int:
    """Ingest declassified DOE records from OSTI API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "osti.jsonl"
    
    # access_limitations=opn is for declassified OpenNet records
    url = "https://www.osti.gov/api/v1/records"
    count = 0
    
    logger.info(f"Ingesting OSTI (max {max_docs})...")
    
    try:
        params = {
            "access_limitations": "opn",
            "size": min(max_docs, 100),
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        records = resp.json()
        
        for record in records:
            if count >= max_docs: break
            
            # OSTI provides full-text links often as 'links' list
            pdf_url = None
            for link_obj in record.get("links", []):
                if link_obj.get("rel") == "fulltext" or ".pdf" in link_obj.get("href", ""):
                    pdf_url = link_obj["href"]
                    break
            
            if pdf_url:
                try:
                    pdf_resp = requests.get(pdf_url, timeout=60)
                    text = _extract_pdf_text(pdf_resp.content)
                    if len(text) < 200:
                        # Fallback to description if PDF fails or is short
                        text = record.get("description", "")
                except:
                    text = record.get("description", "")
            else:
                text = record.get("description", "")
                
            if len(text) > 200:
                doc = {
                    "text": text[:500000],
                    "lang": "en",
                    "date": record.get("publication_date"),
                    "source": record.get("osti_id") or pdf_url,
                    "title": record.get("title"),
                }
                passed, reason = quarantine_check(doc, corpus_type="modern")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                
                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                        
    except Exception as e:
        logger.error(f"OSTI error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# OSF (Marty Pfeiffer Nuclear Archive)
# ---------------------------------------------------------------------------

def ingest_osf_nuclear(
    output_dir: str | Path,
    max_docs: int = 50,
) -> int:
    """Ingest from Marty Pfeiffer's OSF Nuclear collection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "osf_nuclear.jsonl"
    
    # OSF node for Marty's archive
    url = "https://api.osf.io/v2/nodes/46sfd/files/osfstorage/"
    count = 0
    
    logger.info(f"Ingesting OSF Nuclear (node 46sfd, max {max_docs})...")
    
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        files = data.get("data", [])
        
        for file_obj in files:
            if count >= max_docs: break
            if file_obj.get("attributes", {}).get("kind") != "file": continue
            
            download_url = file_obj["links"]["download"]
            file_name = file_obj["attributes"]["name"]
            
            try:
                f_resp = requests.get(download_url, timeout=60)
                if file_name.endswith(".pdf"):
                    text = _extract_pdf_text(f_resp.content)
                else:
                    text = f_resp.text
                
                if len(text) > 200:
                    doc = {
                        "text": text[:500000],
                        "lang": "en",
                        "date": file_obj["attributes"].get("date_modified"),
                        "source": download_url,
                        "title": file_name,
                    }
                    passed, reason = quarantine_check(doc, corpus_type="modern")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
            except Exception as e:
                logger.debug(f"OSF individual file error: {e}")
                
    except Exception as e:
        logger.error(f"OSF error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# The Black Vault
# ---------------------------------------------------------------------------

def ingest_black_vault(
    output_dir: str | Path,
    max_docs: int = 20,
) -> int:
    """Scrape declassified documents from The Black Vault."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "black_vault.jsonl"
    
    # Corrected Black Vault URL (no hyphen)
    url = "https://www.theblackvault.com/documentarchive/category/intelligence/"
    count = 0
    
    logger.info(f"Ingesting The Black Vault (max {max_docs})...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find article links
        articles = soup.select("h3 a") or soup.select(".post-item h3 a")
        for art in articles:
            if count >= max_docs: break
            
            art_url = art["href"]
            try:
                art_resp = requests.get(art_url, headers=headers, timeout=60)
                art_soup = BeautifulSoup(art_resp.text, "html.parser")
                
                # Links to PDFs within the article
                pdf_links = art_soup.select(".entry-content a[href$='.pdf']")
                for plink in pdf_links:
                    if count >= max_docs: break
                    pdf_url = plink["href"]
                    
                    p_resp = requests.get(pdf_url, headers=headers, timeout=60)
                    text = _extract_pdf_text(p_resp.content)
                    
                    if len(text) > 200:
                        doc = {
                            "text": text[:500000],
                            "lang": "en",
                            "date": None,
                            "source": pdf_url,
                            "title": plink.get_text(strip=True) or art.get_text(strip=True),
                        }
                        passed, reason = quarantine_check(doc, corpus_type="modern")
                        log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                        
                        if passed:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            count += 1
            except Exception as e:
                logger.debug(f"Black Vault individual article error: {e}")
                
    except Exception as e:
        logger.error(f"Black Vault error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Wilson Center Digital Archive
# ---------------------------------------------------------------------------

def ingest_wilson_center(
    output_dir: str | Path,
    query: str = "intelligence",
    max_docs: int = 20,
) -> int:
    """Scrape declassified records from the Wilson Center Digital Archive."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "wilson_center.jsonl"
    
    # Wilson Center search results
    url = f"https://digitalarchive.wilsoncenter.org/search-results?q={query}"
    count = 0
    
    logger.info(f"Ingesting Wilson Center (query: {query}, max {max_docs})...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Wilson Center results are in a list of items
        items = soup.select(".search-results-item a") or soup.select(".list-view-item a")
        for item in items:
            if count >= max_docs: break
            
            doc_url = item["href"]
            if not doc_url.startswith("http"):
                doc_url = "https://digitalarchive.wilsoncenter.org" + doc_url
                
            try:
                doc_resp = requests.get(doc_url, headers=headers, timeout=60)
                doc_soup = BeautifulSoup(doc_resp.text, "html.parser")
                
                # Wilson Center often has HTML transcripts
                transcript = doc_soup.select_one(".document-transcript") or doc_soup.select_one(".transcript")
                text = transcript.get_text(separator="\n") if transcript else ""
                
                # Check for PDF download if transcript is missing or short
                if len(text) < 500:
                    pdf_link = doc_soup.select_one("a[href$='/download']")
                    if pdf_link:
                        pdf_url = pdf_link["href"]
                        if not pdf_url.startswith("http"):
                            pdf_url = "https://digitalarchive.wilsoncenter.org" + pdf_url
                        p_resp = requests.get(pdf_url, headers=headers, timeout=60)
                        text = _extract_pdf_text(p_resp.content) or text
                
                if len(text) > 200:
                    doc = {
                        "text": text[:500000],
                        "lang": "en", # Mostly English translations
                        "date": None,
                        "source": doc_url,
                        "title": item.get_text(strip=True),
                    }
                    passed, reason = quarantine_check(doc, corpus_type="modern")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
            except Exception as e:
                logger.debug(f"Wilson Center individual doc error: {e}")
                
    except Exception as e:
        logger.error(f"Wilson Center error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Israel State Archive (ISA)
# ---------------------------------------------------------------------------

def ingest_isa(
    output_dir: str | Path,
    query: str = "Mossad",
    max_docs: int = 20,
) -> int:
    """Scrape declassified records from the Israel State Archive."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "isa.jsonl"
    
    # ISA English catalog search
    url = f"https://www.archives.gov.il/en/search-results/?q={query}"
    count = 0
    
    logger.info(f"Ingesting ISA (query: {query}, max {max_docs})...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # ISA results are typically in a table or list
        links = soup.select(".search-result-item a") or soup.find_all("a", href=re.compile(r".*/publication/.*"))
        for link in links:
            if count >= max_docs: break
            
            pub_url = link["href"]
            if not pub_url.startswith("http"):
                pub_url = "https://www.archives.gov.il" + pub_url
                
            try:
                pub_resp = requests.get(pub_url, headers=headers, timeout=60)
                pub_soup = BeautifulSoup(pub_resp.text, "html.parser")
                
                # Look for direct PDF downloads or Hebrew text
                pdf_links = pub_soup.select("a[href$='.pdf']")
                for plink in pdf_links:
                    if count >= max_docs: break
                    pdf_url = plink["href"]
                    
                    p_resp = requests.get(pdf_url, headers=headers, timeout=60)
                    text = _extract_pdf_text(p_resp.content)
                    
                    if len(text) > 200:
                        doc = {
                            "text": text[:500000],
                            "lang": "he", # Primary language
                            "date": None,
                            "source": pdf_url,
                            "title": plink.get_text(strip=True) or link.get_text(strip=True),
                        }
                        passed, reason = quarantine_check(doc, corpus_type="modern")
                        log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                        
                        if passed:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            count += 1
            except Exception as e:
                logger.debug(f"ISA individual publication error: {e}")
                
    except Exception as e:
        logger.error(f"ISA error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Archive.org (Internet Archive)
# ---------------------------------------------------------------------------

def ingest_internet_archive(
    output_dir: str | Path,
    collection: str = "nationalsecurityarchive",
    max_docs: int = 50,
) -> int:
    """Ingest documents from Archive.org collections."""
    if not internetarchive:
        logger.warning("internetarchive library not installed. Skipping.")
        return 0
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ia_{collection}.jsonl"
    
    count = 0
    logger.info(f"Ingesting IA collection: {collection} (max {max_docs})...")
    
    try:
        search = internetarchive.search_items(f"collection:{collection}")
        for result in search:
            if count >= max_docs: break
            
            item_id = result["identifier"]
            item = internetarchive.get_item(item_id)
            
            # Find a text file or PDF
            text_file = None
            pdf_file = None
            for f in item.files:
                if f["name"].endswith(".txt"):
                    text_file = f["name"]
                elif f["name"].endswith(".pdf"):
                    pdf_file = f["name"]
            
            text = ""
            if text_file:
                # Use item.get_file(text_file).download() or contents
                text = item.get_file(text_file).get_content().decode("utf-8", errors="ignore")
            elif pdf_file:
                if pypdf:
                    try:
                        pdf_data = item.get_file(pdf_file).get_content()
                        text = _extract_pdf_text(pdf_data)
                    except:
                        pass
            
            if len(text) > 200:
                doc = {
                    "text": text[:500000],
                    "lang": "en",
                    "date": item.metadata.get("date"),
                    "source": f"https://archive.org/details/{item_id}",
                    "title": item.metadata.get("title"),
                }
                passed, reason = quarantine_check(doc, corpus_type="modern")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                
                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                    
    except Exception as e:
        logger.error(f"Archive.org error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Government Attic
# ---------------------------------------------------------------------------

def ingest_government_attic(
    output_dir: str | Path,
    max_docs: int = 10,
) -> int:
    """Scrape unique FOIA releases from Government Attic."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "government_attic.jsonl"
    
    url = "https://www.governmentattic.org/index.html"
    count = 0
    
    logger.info(f"Ingesting Government Attic (max {max_docs})...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Government Attic uses many internal links; scan for all .pdf links
        links = soup.find_all("a", href=True)
        for link in links:
            if count >= max_docs: break
            
            pdf_url = link["href"]
            if not pdf_url.lower().endswith(".pdf"): continue
            
            if not pdf_url.startswith("http"):
                # Handle relative links like /dir/file.pdf or ./file.pdf
                if pdf_url.startswith("/"):
                    pdf_url = "https://www.governmentattic.org" + pdf_url
                else:
                    pdf_url = "https://www.governmentattic.org/" + pdf_url
                
            try:
                p_resp = requests.get(pdf_url, headers=headers, timeout=60)
                text = _extract_pdf_text(p_resp.content)
                
                if len(text) > 200:
                    doc = {
                        "text": text[:500000],
                        "lang": "en",
                        "date": None,
                        "source": pdf_url,
                        "title": link.get_text(strip=True),
                    }
                    passed, reason = quarantine_check(doc, corpus_type="modern")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
            except Exception as e:
                logger.debug(f"Gov Attic individual file error: {e}")
                
    except Exception as e:
        logger.error(f"Government Attic error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# DTIC (Defense Technical Information Center)
# ---------------------------------------------------------------------------

def ingest_dtic(
    output_dir: str | Path,
    query: str = "intelligence",
    max_docs: int = 20,
) -> int:
    """Ingest military technical reports from DTIC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dtic.jsonl"
    
    # DTIC public technical reports search
    url = f"https://search.dtic.mil/results?q={query}"
    count = 0
    
    logger.info(f"Ingesting DTIC (max {max_docs})...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # This is a guestimate of the search result selector; DTIC often uses JS
        # Attempting basic metadata if JS is too heavy
        results = soup.select(".search-result") or soup.find_all("div", class_="result")
        for res in results:
            if count >= max_docs: break
            
            link = res.find("a")
            if not link: continue
            
            pdf_url = link["href"]
            try:
                p_resp = requests.get(pdf_url, headers=headers, timeout=60)
                text = _extract_pdf_text(p_resp.content)
                
                if len(text) > 200:
                    doc = {
                        "text": text[:500000],
                        "lang": "en",
                        "date": None,
                        "source": pdf_url,
                        "title": link.get_text(strip=True),
                    }
                    passed, reason = quarantine_check(doc, corpus_type="modern")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
            except:
                pass
                
    except Exception as e:
        logger.error(f"DTIC error: {e}")
        
    return count


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def ingest_all_modern(
    output_dir: str | Path = "data/modern",
    courtlistener_key: str | None = None,
    cap_key: str | None = None,
    max_per_source: int = 5000,
) -> dict[str, int]:
    """Run all modern primary-source ingestors.

    Returns dict mapping source → document count.
    """
    output_dir = Path(output_dir)
    results = {}

    results["courtlistener"] = ingest_courtlistener(output_dir, courtlistener_key, max_per_source)
    results["cap"] = ingest_cap(output_dir, cap_key, max_per_source)
    results["zenodo"] = ingest_zenodo(output_dir, max_docs=min(max_per_source, 1000))
    results["fbi_vault"] = ingest_fbi_vault(output_dir, max_docs=min(max_per_source, 100))
    results["epstein_library"] = ingest_epstein_library(output_dir, max_docs=min(max_per_source, 100))
    results["cia_crest"] = ingest_cia_crest(output_dir, max_docs=min(max_per_source, 100))
    results["osti"] = ingest_osti(output_dir, max_docs=min(max_per_source, 100))
    results["osf_nuclear"] = ingest_osf_nuclear(output_dir, max_docs=min(max_per_source, 100))
    results["black_vault"] = ingest_black_vault(output_dir, max_docs=min(max_per_source, 50))
    results["wilson_center"] = ingest_wilson_center(output_dir, max_docs=min(max_per_source, 50))
    results["isa"] = ingest_isa(output_dir, max_docs=min(max_per_source, 50))
    results["ia_national_security"] = ingest_internet_archive(output_dir, collection="nationalsecurityarchive", max_docs=min(max_per_source, 50))
    results["government_attic"] = ingest_government_attic(output_dir, max_docs=min(max_per_source, 20))
    results["dtic"] = ingest_dtic(output_dir, max_docs=min(max_per_source, 20))
    
    total = sum(results.values())
    logger.info(f"Total modern corpus: {total} documents across {len(results)} sources.")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    ingest_all_modern()
def run_bulk_ingestion(output_dir: str | Path = "data/primary_source"):
    """Run all primary-source ingestors at once."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # RECAP
    ingest_courtlistener(output_dir, max_docs=100)
    
    # Zenodo
    ingest_zenodo(output_dir, max_docs=50)
    
    # OSF Nuclear
    ingest_osf_nuclear(output_dir, max_docs=20)
    
    # Black Vault
    ingest_black_vault(output_dir, max_docs=10)
    
    # OSTI
    ingest_osti(output_dir, max_docs=20)
    
    logger.info("Bulk primary-source ingestion phase 1 complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bulk_ingestion()
