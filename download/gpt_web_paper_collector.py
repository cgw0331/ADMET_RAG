#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Europe PMC collector (REST-first, no Entrez, no GPT)

- Uses Europe PMC REST API to search: organoid AND (ADMET indicators)
- Prefers PMCID hits; falls back to direct pdfUrl if present
- Downloads PDFs to raws_v1/PMCxxxx/article.pdf or raws_v1/AUTO_xxx/article.pdf
- If PMCID available, also downloads supplementary files from PMC article page
"""

import os
import re
import time
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

OUT_DIR = Path("./raws_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADMET_KEYWORDS = {
    # Absorption
    "caco2_permeability": '"Caco-2 permeability" OR Papp OR "apparent permeability"',
    "mdck_permeability": 'MDCK OR "Madin-Darby canine kidney"',
    "pampa": 'PAMPA OR "parallel artificial membrane permeability"',
    "lipinski_rule_of_five": '"Lipinski rule" OR "rule of five"',
    "logd_logs_pka": 'logD OR logS OR pKa OR "partition coefficient" OR solubility',
    # Distribution
    "ppb": '"plasma protein binding" OR PPB OR "serum protein binding"',
    "bbb": '"blood-brain barrier" OR BBB OR "brain penetration"',
    "vd": '"volume of distribution" OR Vd OR "distribution volume"',
    # Metabolism
    "cyp1a2": 'CYP1A2',
    "cyp2c9": 'CYP2C9',
    "cyp2c19": 'CYP2C19',
    "cyp2d6": 'CYP2D6',
    "cyp3a4": 'CYP3A4',
    "cyp_inhibition": '"CYP inhibition" OR "P450 inhibition"',
    # Excretion
    "cl": 'clearance OR "renal clearance" OR "hepatic clearance"',
    "t_half": '"half-life" OR t1/2 OR "terminal half-life"',
    # Toxicity
    "herg": 'hERG OR "QT prolongation"',
    "dili": 'DILI OR hepatotoxicity OR "Drug-Induced Liver Injury"',
    "ames_test": '"Ames test" OR mutagenicity',
    "carcinogenicity": 'carcinogenic* OR genotoxic* OR "cancer risk"',
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (ADMET-Organoid-Collector; +https://europepmc.org/)",
    "Accept": "*/*",
}
EPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PMC_BASE_WWW = "https://www.ncbi.nlm.nih.gov/pmc/articles"
PMC_BASE_CDN = "https://pmc.ncbi.nlm.nih.gov/articles"

SUPP_EXTS = (".pdf", ".zip", ".xlsx", ".xls", ".csv", ".tsv", ".doc", ".docx", ".ppt", ".pptx")

# ---------- HTTP Session with Retry ----------
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=False  # retry on all
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s

SESSION = make_session()


# ---------- Query Builder ----------
def build_queries(max_blocks: Optional[int] = None) -> List[str]:
    """Build Europe PMC queries: organoid AND (<block>), default uses ALL blocks."""
    blocks = list(ADMET_KEYWORDS.values())
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    return [f"organoid AND ({b})" for b in blocks]


# ---------- REST Search ----------
def epmc_rest_search(query: str, page: int = 1, page_size: int = 25) -> List[Dict]:
    """
    Europe PMC REST:
      - params: query=<q>, page=<page>, pageSize=<n>, format=json
      - fields of interest: pmcid, pdfUrl, ext_id(PMID/DOI), title
    """
    params = {
        "query": query,
        "page": str(page),
        "pageSize": str(page_size),
        "format": "json",
    }
    r = SESSION.get(EPMC_API, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    hits = j.get("resultList", {}).get("result", []) or []
    items = []
    for h in hits:
        pmcid = h.get("pmcid")  # e.g., "PMC7878295"
        pdf_url = h.get("pdfUrl")  # sometimes direct
        title = h.get("title", "")
        ext_id = h.get("ext_id")  # PMID or DOI
        items.append({"pmcid": pmcid, "pdf_url": pdf_url, "title": title, "ext_id": ext_id})
    return items


# ---------- Utils ----------
def auto_folder_id(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"AUTO_{h}"

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._\\-]+", "_", s).strip("_")
    return s[:200] if len(s) > 200 else s

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def looks_like_pdf(resp: requests.Response, first_chunk: bytes) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ct:
        return True
    # fallback: PDF header
    return first_chunk.startswith(b"%PDF-")

# ---------- Downloaders ----------
def stream_download(url: str, out_path: Path, referer: Optional[str] = None, accept_pdf: bool = False) -> bool:
    headers = {}
    if referer:
        headers["Referer"] = referer
    if accept_pdf:
        headers["Accept"] = "application/pdf"
    with SESSION.get(url, headers=headers, timeout=90, stream=True, allow_redirects=True) as r:
        if r.status_code != 200:
            return False
        # Peek first bytes to validate PDF if needed
        first = b""
        for chunk in r.iter_content(8192):
            if chunk:
                first = chunk
                break
        if accept_pdf and not looks_like_pdf(r, first):
            return False
        ensure_dir(out_path.parent)
        with open(out_path, "wb") as f:
            if first:
                f.write(first)
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    return True

def try_download_pmc_pdf(pmcid: str, out_pdf: Path) -> bool:
    """
    Try multiple canonical PDF endpoints for a given PMCID.
    """
    candidates = [
        f"{PMC_BASE_WWW}/{pmcid}/pdf",
        f"{PMC_BASE_WWW}/{pmcid}/pdf/",
        f"{PMC_BASE_WWW}/{pmcid}/pdf/?download=1",
        f"{PMC_BASE_WWW}/{pmcid}/pdf/{pmcid}.pdf",
        f"{PMC_BASE_CDN}/{pmcid}/pdf",
        f"{PMC_BASE_CDN}/{pmcid}/pdf/",
        f"{PMC_BASE_CDN}/{pmcid}/pdf/?download=1",
        f"{PMC_BASE_CDN}/{pmcid}/pdf/{pmcid}.pdf",
        f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
    ]
    for url in candidates:
        ok = stream_download(url, out_pdf, referer=url, accept_pdf=True)
        if ok:
            return True
    return False

def download_pmc_supplements(pmcid: str, pmc_dir: Path, max_files: int = 10) -> int:
    """
    Parse PMC HTML to find /bin/ and common supp-file links.
    """
    saved = 0
    for base in (PMC_BASE_WWW, PMC_BASE_CDN):
        article_url = f"{base}/{pmcid}/"
        try:
            r = SESSION.get(article_url, timeout=30)
            if r.status_code != 200 or not r.text:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            sdir = pmc_dir / "supplements"
            ensure_dir(sdir)

            # Collect candidate links
            found = []
            for a in soup.select("a[href]"):
                href = a.get("href", "").strip()
                if not href:
                    continue
                abs_url = href if href.startswith("http") else urllib.parse.urljoin(article_url, href)
                low = abs_url.lower()
                cond_path = ("/bin/" in low) or ("download" in low) or low.endswith(SUPP_EXTS)
                if cond_path:
                    found.append(abs_url)

            uniq = []
            for u in found:
                if u not in uniq:
                    uniq.append(u)

            for u in uniq:
                if saved >= max_files:
                    break
                name = sanitize_filename(u.split("/")[-1].split("?")[0]) or f"supp_{saved+1}"
                out = sdir / name
                if stream_download(u, out, referer=article_url, accept_pdf=False):
                    saved += 1
            if saved > 0:
                return saved
        except Exception:
            continue
    return saved

def download_from_pmcid(pmcid: str, title: str, query: str) -> Optional[Path]:
    pmc_dir = OUT_DIR / pmcid
    pdf_path = pmc_dir / "article.pdf"
    if pdf_path.exists():
        return pdf_path
    # 1) PDF
    ok = try_download_pmc_pdf(pmcid, pdf_path)
    if not ok:
        return None
    # 2) Metadata
    meta = {"pmcid": pmcid, "title": title, "query": query, "source": "EuropePMC"}
    ensure_dir(pmc_dir)
    with open(pmc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # 3) Supplements
    supp_count = download_pmc_supplements(pmcid, pmc_dir, max_files=10)
    print(f"[OK] {pmcid} → {pdf_path} (supp: {supp_count})")
    return pdf_path

def download_from_direct_pdf(url: str, title: str, query: str) -> Optional[Path]:
    folder = auto_folder_id(url)
    fdir = OUT_DIR / folder
    pdf_path = fdir / "article.pdf"
    if pdf_path.exists():
        return pdf_path
    ok = stream_download(url, pdf_path, referer=url, accept_pdf=True)
    if not ok:
        return None
    meta = {"direct_pdf": url, "title": title, "query": query, "source": "EuropePMC"}
    ensure_dir(fdir)
    with open(fdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] {folder} (direct) → {pdf_path}")
    return pdf_path


# ---------- Collector ----------
def collect(target: int = 3, max_blocks: Optional[int] = None, pages_per_query: int = 2, page_size: int = 25, delay: float = 0.25, skip_existing: bool = True):
    saved = 0
    queries = build_queries(max_blocks=max_blocks)
    # Pre-scan existing PMCID folders to skip already collected
    existing_pmcids = set()
    if skip_existing:
        for p in OUT_DIR.iterdir():
            if p.is_dir() and p.name.startswith("PMC"):
                existing_pmcids.add(p.name)
        if existing_pmcids:
            print(f"[SKIP] existing PMCID folders: {len(existing_pmcids)}")
    for q in queries:
        if saved >= target:
            break
        for page in range(1, pages_per_query + 1):
            if saved >= target:
                break
            try:
                items = epmc_rest_search(q, page=page, page_size=page_size)
            except Exception as e:
                print(f"[REST FAIL] {q} page={page}: {e}")
                continue

            for it in items:
                if saved >= target:
                    break
                pmcid = it.get("pmcid")
                pdf_url = it.get("pdf_url")
                title = it.get("title", "")
                # Prefer PMCID path (stable OA)
                if pmcid:
                    if skip_existing and pmcid in existing_pmcids:
                        continue
                    if download_from_pmcid(pmcid, title, q):
                        saved += 1
                        time.sleep(delay)
                        continue
                # Fallback: direct pdfUrl if exists
                if pdf_url:
                    if download_from_direct_pdf(pdf_url, title, q):
                        saved += 1
                        time.sleep(delay)
                        continue
            time.sleep(delay)
    print(f"[DONE] saved={saved}/{target} → {OUT_DIR.resolve()}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--max_blocks", type=int, default=None, help="Limit number of ADMET keyword blocks (default: use all)")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--page_size", type=int, default=25)
    args = ap.parse_args()

    collect(
        target=args.target,
        max_blocks=args.max_blocks,
        pages_per_query=args.pages,
        page_size=args.page_size,
    )

if __name__ == "__main__":
    main()
