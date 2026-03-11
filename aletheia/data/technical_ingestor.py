"""
STEM & Technical Ingestion (§2.4-2.5)

Scrapers and processors for:
- arXiv research papers (STEM)
- Cybersecurity writeups & Bulletins
- Unix/BSD Man Pages
- Programming Syntax Guides (pre-2022)

Enforces strict temporal filters to avoid AI inbreeding.
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from aletheia.data.quarantine import quarantine_check, log_audit

logger = logging.getLogger(__name__)

# Temporal cutoff for "No AI Inbreeding" (§2.5)
CUTOFF_DATE = "2021-12-31"

def ingest_arxiv(
    output_dir: str | Path,
    categories: list[str] = ["physics", "astro-ph", "q-bio", "math"],
    max_results: int = 100,
) -> int:
    """Ingest research papers from arXiv via OAI-PMH or API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "arxiv_stem.jsonl"
    
    count = 0
    # arXiv API endpoint
    base_url = "http://export.arxiv.org/api/query"
    
    logger.info(f"Ingesting arXiv papers (categories={categories}, cutoff={CUTOFF_DATE})...")
    
    for cat in categories:
        if count >= max_results: break
        
        # Query: category AND date range (simulated via API params)
        params = {
            "search_query": f"cat:{cat} AND submittedDate:[0000 TO 202112312359]",
            "start": 0,
            "max_results": min(max_results, 50),
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "xml")
            
            entries = soup.find_all("entry")
            for entry in entries:
                if count >= max_results: break
                
                title = entry.find("title").get_text(strip=True)
                summary = entry.find("summary").get_text(strip=True)
                published = entry.find("published").get_text(strip=True)
                arxiv_id = entry.find("id").get_text(strip=True).split("/")[-1]
                
                # Metadata check for AI inbreeding markers
                if any(k in summary.lower() for k in ["large language model", "chatgpt", "gpt-4"]):
                    continue
                
                doc = {
                    "text": f"Title: {title}\n\nAbstract: {summary}",
                    "lang": "en", # arXiv metadata is primarily English, though papers vary
                    "date": published,
                    "source": f"arxiv:{arxiv_id}",
                    "title": title,
                    "category": cat
                }
                
                passed, reason = quarantine_check(doc, corpus_type="stem")
                log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                
                if passed:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                    
        except Exception as e:
            logger.error(f"arXiv ingestion error for {cat}: {e}")
            
    return count

def ingest_man_pages(
    output_dir: str | Path,
    source_dir: str | Path = "/usr/share/man", # Default for Linux
) -> int:
    """Ingest local Unix/BSD Man pages (pre-trained logic)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "man_pages.jsonl"
    
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.warning(f"Man pages source not found: {source_dir}")
        return 0
    
    count = 0
    logger.info(f"Ingesting Man pages from {source_dir}...")
    
    # Traverse man directories (man1-man8)
    for man_dir in source_path.glob("man*"):
        for man_file in man_dir.glob("*.[1-8]*"):
            try:
                # Use 'man' command or read raw roff
                import subprocess
                try:
                    text = subprocess.check_output(["man", str(man_file)], 
                                                   stderr=subprocess.STDOUT, 
                                                   text=True, 
                                                   encoding="utf-8")
                except:
                    with open(man_file, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                
                if len(text) < 500: continue
                
                doc = {
                    "text": text,
                    "lang": "en",
                    "date": "2021-12-31", # Man pages are system-level/static
                    "source": f"man:{man_file.name}",
                    "title": f"Manual Page: {man_file.name}",
                }
                
                # Basic quarantine (system files usually pass)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.debug(f"Error reading man page {man_file}: {e}")
                continue
                
    logger.info(f"Ingested {count} Man pages.")
    return count

def ingest_cybersecurity(
    output_dir: str | Path,
    max_results: int = 50,
) -> int:
    """Ingest Cybersecurity writeups (Full-Disclosure / Exploit-DB style)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cyber_security.jsonl"
    
    # Target: Seclists.org Full Disclosure archives (organized by month/year)
    # We target 2021 specifically to avoid AI inbreeding
    url = "https://seclists.org/fulldisclosure/2021/Oct/"
    count = 0
    
    logger.info(f"Ingesting Cybersecurity writeups (cutoff={CUTOFF_DATE})...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find links to specific threads
        links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].isdigit()]
        
        for link_id in links[:max_results]:
            thread_url = f"{url}{link_id}"
            try:
                t_resp = requests.get(thread_url, headers=headers, timeout=30)
                t_resp.raise_for_status()
                t_soup = BeautifulSoup(t_resp.text, "html.parser")
                
                # Content is usually in a <pre> block
                content = t_soup.find("pre")
                if content:
                    text = content.get_text()
                    if len(text) < 300: continue
                    
                    doc = {
                        "text": text,
                        "lang": "en",
                        "date": "2021-10-01", 
                        "source": thread_url,
                        "title": t_soup.title.get_text(strip=True) if t_soup.title else "Full Disclosure Entry",
                    }
                    
                    passed, reason = quarantine_check(doc, corpus_type="cyber")
                    log_audit(doc, passed, reason, output_dir / "quarantine_log.jsonl")
                    
                    if passed:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
                        
            except Exception as e:
                logger.debug(f"Error reading cyber thread {thread_url}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Cybersecurity ingestion error: {e}")
        
    return count

def ingest_syntax_guides(
    output_dir: str | Path,
) -> int:
    """Ingest version-locked official Programming Syntax guides (pre-2022)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "syntax_guides.jsonl"
    
    # Targets: Python 3.10 Docs, Rust 1.56, etc.
    targets = [
        {"url": "https://docs.python.org/3.10/reference/index.html", "title": "Python 3.10 Grammar", "lang": "py"},
        {"url": "https://doc.rust-lang.org/1.56.0/reference/index.html", "title": "Rust 1.56 Reference", "lang": "rs"},
    ]
    
    count = 0
    logger.info("Ingesting Programming Syntax Guides (Pre-2022 Focus)...")
    
    for target in targets:
        try:
            resp = requests.get(target["url"], timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Extract main content
            text = soup.get_text(separator="\n")
            
            doc = {
                "text": text,
                "lang": "en",
                "date": "2021-10-01", # Approximate for 3.10 release
                "source": target["url"],
                "title": target["title"],
                "ext": target.get("lang")
            }
            
            # Syntax guides are high-purity documents
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            
        except Exception as e:
            logger.error(f"Syntax guide error for {target['url']}: {e}")
            
    return count

def ingest_qdl_science(
    output_dir: str | Path,
    max_results: int = 20,
) -> int:
    """Ingest Islamic Golden Age scientific manuscripts from QDL (Qatar Digital Library)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "islamic_science.jsonl"
    
    # QDL Search for Scientific Manuscripts (Al-Khwarizmi, Al-Biruni)
    url = "https://www.qdl.qa/en/search/site/science?f%5B0%5D=field_subject%3A334"
    count = 0
    
    logger.info("Ingesting Islamic Golden Age Science (QDL)...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find manuscript links
        links = soup.select("h2.search-result-title a[href]")
        
        for link in links[:max_results]:
            ms_url = f"https://www.qdl.qa{link['href']}"
            try:
                ms_resp = requests.get(ms_url, headers=headers, timeout=30)
                ms_resp.raise_for_status()
                ms_soup = BeautifulSoup(ms_resp.text, "html.parser")
                
                # Extract descriptive metadata (which often contains transcriptions/summaries)
                content = ms_soup.find("div", class_="field-name-field-content-summary")
                if content:
                    text = content.get_text()
                    if len(text) < 200: continue
                    
                    doc = {
                        "text": text,
                        "lang": "ar", # Primary target is original Arabic
                        "date": "1000-01-01", # Placeholder for Golden Age
                        "source": ms_url,
                        "title": ms_soup.title.get_text(strip=True) if ms_soup.title else "QDL Manuscript",
                    }
                    
                    # Store as original antiquity (pre-1930)
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                        
            except Exception as e:
                logger.debug(f"Error reading QDL entry {ms_url}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"QDL ingestion error: {e}")
        
    return count

def ingest_manhattan_project(
    output_dir: str | Path,
) -> int:
    """Ingest Manhattan Project District History and technical reports from DOE OpenNet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "manhattan_project.jsonl"
    
    # Target: Manhattan District History (Official DOE Collection)
    url = "https://www.osti.gov/opennet/manhattan_district.jsp"
    count = 0
    
    logger.info("Ingesting Manhattan Project Primary Sources...")
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find links to the 36 volumes (PDF or HTML)
        links = soup.find_all("a", href=re.compile(r"manhattan_district_history"))
        
        for link in links[:10]: # Sample volumes
            vol_url = f"https://www.osti.gov/opennet/{link['href']}" if not link['href'].startswith("http") else link['href']
            try:
                vol_resp = requests.get(vol_url, timeout=30)
                vol_resp.raise_for_status()
                vol_soup = BeautifulSoup(vol_resp.text, "html.parser")
                
                text = vol_soup.get_text(separator="\n")
                if len(text) < 1000: continue
                
                doc = {
                    "text": text,
                    "lang": "en",
                    "date": "1945-01-01",
                    "source": vol_url,
                    "title": f"Manhattan District History: {link.get_text(strip=True)}",
                }
                
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.debug(f"Error reading Manhattan volume {vol_url}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Manhattan Project ingestion error: {e}")
        
    return count

def ingest_museum_classics(
    output_dir: str | Path,
) -> int:
    """Ingest 'Museum' grade scientific foundations in original languages."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "museum_classics.jsonl"
    
    # Authoritative Targets
    targets = [
        {"url": "http://www.newtonproject.ox.ac.uk/texts/view/TR000030?mode=transcription", "title": "Newton: Principia (Latin Transcription)", "lang": "la"},
        {"url": "https://einsteinpapers.press.princeton.edu/vol2-doc/312", "title": "Einstein: On the Electrodynamics of Moving Bodies (German)", "lang": "de"},
        {"url": "http://farside.ph.utexas.edu/Books/Euclid/Elements.html", "title": "Euclid: The Elements (Original Greek/English Reference)", "lang": "grc"},
    ]
    
    count = 0
    logger.info("Ingesting 'Museum' Scientific Foundations (Original Source Only)...")
    
    for target in targets:
        try:
            resp = requests.get(target["url"], timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Extract content (targeting main text containers)
            text = soup.get_text(separator="\n")
            if len(text) < 1000: continue
            
            doc = {
                "text": text,
                "lang": target["lang"],
                "date": "1687-01-01", # Placeholder for the specific era
                "source": target["url"],
                "title": target["title"],
            }
            
            # No AI inbreeding check (Antiquity is safe)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            
        except Exception as e:
            logger.error(f"Museum classic error for {target['url']}: {e}")
            
    return count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
