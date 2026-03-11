"""
Data Cleaning & Deduplication (§5.1)

Provides MinHashLSH deduplication and language-strict filtering to ensure
pure antiquity subsets without modern translations or duplicates.
"""

from __future__ import annotations
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Set

logger = logging.getLogger(__name__)

def minhash_fingerprint(text: str, num_perm: int = 128) -> list[int]:
    """Simple MinHash fingerprint for deduplication."""
    import hashlib
    tokens = set(text.lower().split())
    if not tokens: return [0] * num_perm
    
    fingerprint = []
    for i in range(num_perm):
        min_hash = float('inf')
        for t in tokens:
            h = int(hashlib.md5(f"{t}_{i}".encode()).hexdigest(), 16)
            if h < min_hash:
                min_hash = h
        fingerprint.append(min_hash)
    return fingerprint

def is_ai_generated(text: str) -> bool:
    """Detect common AI-generated text markers (Anti-AI Inbreeding)."""
    markers = [
        r"as an ai language model",
        r"i don't have personal feelings",
        r"my knowledge cutoff",
        r"cannot fulfill this request",
        r"as of my last update",
        r"openai",
        r"chatgpt",
        r"language model developed by",
    ]
    text_lower = text.lower()
    for marker in markers:
        if re.search(marker, text_lower):
            return True
    return False

def is_temporal_safe(date_str: str) -> bool:
    """Verify data is pre-2022 to avoid training on AI-generated web-slop."""
    if not date_str: return True # Default safe if date unknown (e.g. antiquity)
    try:
        # Extract year
        match = re.search(r"(\d{4})", date_str)
        if match:
            year = int(match.group(1))
            return year <= 2021
    except:
        pass
    return True

def is_strict_original(text: str, lang: str) -> bool:
    """Validate that text is strictly in the original language (no English/Latin mix)."""
    if not text: return False
    
    # Remove common punctuation/whitespace
    clean_text = re.sub(r'[\s\d\p{P}]+', '', text)
    if not clean_text: return False
    
    if lang == "fa": # Persian (Arabic script)
        # Range for Arabic/Persian characters
        persian_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        return (persian_chars / len(clean_text)) > 0.95
    
    if lang == "el": # Ancient Greek
        greek_chars = len(re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]', text))
        return (greek_chars / len(clean_text)) > 0.90
    
    if lang == "yo": # Yoruba (Latin with heavy diacritics)
        # Check for characteristic Yoruba diacritics (sub-dots, tonemarks)
        # ẹ, ọ, ṣ, etc.
        yoruba_markers = len(re.findall(r'[ẹọṣẸỌṢ]', text))
        return yoruba_markers > 0 or len(re.findall(r'[a-zA-Z]', text)) > 0.8 # Broad check
        
    return True

def clean_corpus(input_file: str | Path, output_file: str | Path, lang: str):
    """Clean and deduplicate a JSONL corpus file."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    seen_hashes: Set[tuple[int, ...]] = set()
    
    count = 0
    removed_dupes = 0
    removed_mixed = 0
    removed_ai = 0
    removed_temporal = 0
    
    logger.info(f"Cleaning {input_file} (lang={lang})...")
    
    with open(input_file, "r", encoding="utf-8") as f, \
         open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                date = str(doc.get("date", ""))
                
                # 1. Strict language check
                if not is_strict_original(text, lang):
                    removed_mixed += 1
                    continue
                
                # 2. AI Inbreeding check
                if is_ai_generated(text):
                    removed_ai += 1
                    continue
                
                # 3. Temporal Safety check
                if not is_temporal_safe(date):
                    removed_temporal += 1
                    continue
                
                # 4. Deduplication check (MinHash)
                h = tuple(minhash_fingerprint(text[:1000], num_perm=8))
                if h in seen_hashes:
                    removed_dupes += 1
                    continue
                seen_hashes.add(h)
                
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.error(f"Error cleaning line: {e}")
                
    logger.info(f"Cleaned {input_file}: Ingested={count}, Dupes={removed_dupes}, AI={removed_ai}, Late={removed_temporal}, Mixed={removed_mixed}")
    return count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    # clean_corpus("data/classical/ganjoor_rumi.jsonl", "data/classical/ganjoor_rumi_clean.jsonl", "fa")
