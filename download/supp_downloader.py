#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
supp_downloader.py
Fetch supplementary material links for a given PMC ID and download them.
Priority: PMC -> Publisher page (via DOI) -> Elsevier CDN heuristic -> (optional) GPT assist.
Usage:
  python supp_downloader.py PMC7878295 -o out
  python supp_downloader.py PMC7878295 --max-probe 12 --exts xlsx,csv
Env:
  .env with OPEN_API_KEY=sk-... (optional; only used if all heuristics fail)
"""
import argparse
import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional GPT assist
OPENAI_AVAILABLE = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPEN_API_KEY", "").strip()
    if OPENAI_API_KEY:
        OPENAI_AVAILABLE = True
        import json
        import textwrap
        # Use requests to call OpenAI responses API (avoid extra deps)
        def ask_gpt_for_links(context):
            try:
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "gpt-5.1-mini",
                    "input": textwrap.dedent(context)[:12000],  # keep payload safe-ish size
                }
                resp = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "")
                    # Extract URLs from text
                    urls = re.findall(r'https?://[^\s<>"\'\)\]]+', text)
                    return list(dict.fromkeys(urls)), text
            except Exception as e:
                print(f"[GPT] error: {e}", file=sys.stderr)
            return [], ""
    else:
        OPENAI_AVAILABLE = False
except Exception:
    OPENAI_AVAILABLE = False

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
SESS = requests.Session()
SESS.headers.update({
    "User-Agent": UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
})

# Add resilient retries for transient failures and rate limits
_retry = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "HEAD"]),
    raise_on_status=False,
)
SESS.mount("https://", HTTPAdapter(max_retries=_retry))
SESS.mount("http://", HTTPAdapter(max_retries=_retry))

# Map extensions to Accept headers to nudge correct content-type
ACCEPT_BY_EXT = {
    "pdf": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream;q=0.9,*/*;q=0.8",
    "xls": "application/vnd.ms-excel,application/octet-stream;q=0.9,*/*;q=0.8",
    "csv": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
    "tsv": "text/tab-separated-values,application/octet-stream;q=0.9,*/*;q=0.8",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/octet-stream;q=0.9,*/*;q=0.8",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation,application/octet-stream;q=0.9,*/*;q=0.8",
}

def http_get(url, **kw):
    r = SESS.get(url, timeout=kw.pop("timeout", 30), allow_redirects=True, **kw)
    r.raise_for_status()
    return r

def http_head(url, **kw):
    r = SESS.head(url, timeout=kw.pop("timeout", 15), allow_redirects=True, **kw)
    # Some CDNs don't support HEAD well; treat 200 as good, others we ignore
    return r

SUP_EXTS_DEFAULT = ["xlsx","xls","csv","tsv","pdf","docx","pptx","xml","txt"]

def find_pmc_links(pmcid, exts):
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    out = []
    try:
        res = http_get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        # Direct /bin/ links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/bin/" in href or "supplementary" in href.lower():
                absu = urljoin(url, href)
                if any(absu.lower().endswith("." + e) for e in exts):
                    out.append(absu)
        # Fallback: anything labeled Supplementary
        sup_sections = soup.find_all(lambda tag: tag.name in ["section","div"] and "supplement" in (tag.get("id","") + " " + " ".join(tag.get("class",[]))).lower())
        for sec in sup_sections:
            for a in sec.find_all("a", href=True):
                absu = urljoin(url, a["href"])
                if any(absu.lower().endswith("." + e) for e in exts):
                    out.append(absu)
        out = list(dict.fromkeys(out))
        # Extract DOI for later - try multiple methods
        doi = None
        # Method 1: meta tag
        meta = soup.find("meta", {"name":"citation_doi"})
        if meta and meta.get("content"):
            doi = meta["content"].strip()
        # Method 2: Look for DOI in text/links
        if not doi:
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "doi.org" in href:
                    m = re.search(r'10\.\d+/[^\s<>"\']+', href)
                    if m:
                        doi = m.group(0)
                        break
        # Method 3: Look for DOI text pattern
        if not doi:
            text = soup.get_text()
            m = re.search(r'10\.\d+/[^\s<>"\'\(\)]+', text)
            if m:
                doi = m.group(0)
        return out, doi, res.text  # Return HTML too
    except Exception as e:
        print(f"[PMC] fetch error: {e}", file=sys.stderr)
        return [], None, None

def resolve_doi(doi):
    if not doi: 
        return None, None
    try:
        # Try with different referers to avoid 403
        referers = [
            "https://pmc.ncbi.nlm.nih.gov/",
            "https://www.ncbi.nlm.nih.gov/",
            "https://doi.org/",
        ]
        for ref in referers:
            try:
                headers = {"Referer": ref, "User-Agent": UA}
                r = SESS.get(f"https://doi.org/{doi}", headers=headers, timeout=30, allow_redirects=True)
                r.raise_for_status()
                final = r.url
                return final, r.text
            except Exception:
                continue
        # If all referers fail, try without referer
        r = SESS.get(f"https://doi.org/{doi}", timeout=30, allow_redirects=True)
        r.raise_for_status()
        final = r.url
        return final, r.text
    except Exception as e:
        print(f"[DOI] resolve error: {e}", file=sys.stderr)
        return None, None

def extract_links_from_html(base_url, html, exts, pmc_filenames=None):
    """
    Extract supplementary material links from publisher HTML.
    pmc_filenames: list of filenames found on PMC (to match similar files on publisher)
    """
    out = []
    if not html: 
        return out
    soup = BeautifulSoup(html, "html.parser")
    
    # direct anchors with supplement keywords
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = (a.get_text() or "").lower()
        if any(k in (href.lower() + " " + label) for k in ["supp", "supplement", "mmc", "supporting", "suppl", "additional"]):
            absu = urljoin(base_url, href)
            if any(absu.lower().split("?")[0].endswith("." + e) for e in exts):
                out.append(absu)
    
    # Also look for files matching PMC filenames (even without supplement keywords)
    if pmc_filenames:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            absu = urljoin(base_url, href)
            # Extract filename from URL
            url_filename = os.path.basename(urlparse(absu).path).lower()
            # Check if this matches any PMC filename
            for pmc_fn in pmc_filenames:
                pmc_base = os.path.splitext(os.path.basename(pmc_fn))[0].lower()
                if pmc_base in url_filename or url_filename in pmc_base:
                    if any(absu.lower().split("?")[0].endswith("." + e) for e in exts):
                        out.append(absu)
    
    # look into script tags (some sites embed JSON with asset URLs)
    text = soup.get_text(" ")
    # quick regex for urls in HTML
    urls = re.findall(r'https?://[^\s<>"\'\)\]]+', html)
    for u in urls:
        if any(k in u.lower() for k in ["supp", "supplement", "mmc", "mediaobjects", "suppl", "additional"]):
            if any(u.lower().split("?")[0].endswith("." + e) for e in exts):
                out.append(u)
    
    return list(dict.fromkeys(out))

def guess_elsevier_cdn(landing_url, html, exts, max_probe=10):
    """
    If publisher is Elsevier (sciencedirect/linkinghub), try ars.els-cdn PII-based pattern.
    """
    if not landing_url:
        return []
    host = urlparse(landing_url).hostname or ""
    if "sciencedirect.com" not in host and "linkinghub.elsevier.com" not in host:
        return []
    # PII extraction
    PII = None
    # From URL
    m = re.search(r'(S\d{16})', landing_url)
    if m:
        PII = m.group(1)
    # From HTML
    if not PII and html:
        m = re.search(r'(S\d{16})', html)
        if m:
            PII = m.group(1)
    if not PII:
        return []
    cdn_base = f"https://ars.els-cdn.com/content/image/1-s2.0-{PII}-mmc{{n}}"
    # common extensions to probe first
    probe_exts = ["xlsx","pdf","csv","xls"]
    out = []
    for i in range(1, max_probe+1):
        for ext in probe_exts + [e for e in exts if e not in probe_exts]:
            url = cdn_base.format(n=i) + f".{ext}"
            try:
                r = http_head(url)
                if r.status_code == 200 and int(r.headers.get("Content-Length","0")) > 0:
                    out.append(url)
                    # Avoid duplicates, but continue probing for others
            except Exception:
                pass
    return list(dict.fromkeys(out))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def safe_name(s):
    s = re.sub(r'[^a-zA-Z0-9._-]+', '_', s)
    return s.strip("_")

def is_valid_file(filepath, expected_ext=None):
    """
    Check if downloaded file is valid (not HTML, not too small, matches extension).
    Returns (is_valid, reason)
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    size = os.path.getsize(filepath)
    
    # Check file size (too small = likely HTML/POW page)
    if size < 5000:  # Less than 5KB is suspicious
        # Read first bytes to check if it's HTML
        try:
            with open(filepath, "rb") as f:
                first_bytes = f.read(512)
                text = first_bytes.decode("utf-8", errors="ignore").lower()
                if "<html" in text or "preparing to download" in text or "pow" in text:
                    return False, f"HTML/POW page detected (size: {size} bytes)"
        except Exception:
            pass
        # Even if not HTML, too small files are suspicious
        if size < 1024:  # Less than 1KB is definitely wrong
            return False, f"File too small (size: {size} bytes)"
    
    # Check if PDF file actually starts with %PDF
    if expected_ext == "pdf" or filepath.lower().endswith(".pdf"):
        try:
            with open(filepath, "rb") as f:
                header = f.read(4)
                if not header.startswith(b"%PDF"):
                    # Check if it's HTML
                    f.seek(0)
                    text = f.read(512).decode("utf-8", errors="ignore").lower()
                    if "<html" in text:
                        return False, "PDF file contains HTML content"
        except Exception:
            pass
    
    # Check if ZIP file actually starts with ZIP signature
    if expected_ext in ["docx", "xlsx", "pptx"] or any(filepath.lower().endswith(f".{e}") for e in ["docx", "xlsx", "pptx"]):
        try:
            with open(filepath, "rb") as f:
                header = f.read(4)
                # ZIP signatures: PK\x03\x04 (ZIP), PK\x05\x06 (empty ZIP), PK\x07\x08 (spanned)
                # DOCX/XLSX/PPTX are also ZIP archives
                if not header.startswith(b"PK"):
                    # Check if it's HTML
                    f.seek(0)
                    text = f.read(512).decode("utf-8", errors="ignore").lower()
                    if "<html" in text:
                        return False, "ZIP/DOCX/XLSX file contains HTML content"
        except Exception:
            pass
    
    return True, "Valid file"

def download_file(url, outdir, referer=None, expected_ext=None):
    """
    Download a single file with referer header support.
    Returns (success, filepath, reason)
    """
    try:
        fname = os.path.basename(urlparse(url).path)
        path = os.path.join(outdir, safe_name(fname))
        
        headers = {}
        if referer:
            headers["Referer"] = referer
        # Add Accept based on ext
        if expected_ext and expected_ext in ACCEPT_BY_EXT:
            headers["Accept"] = ACCEPT_BY_EXT[expected_ext]
        # Add Origin for stricter CDNs
        try:
            from urllib.parse import urlparse as _urlparse
            _u = _urlparse(url)
            headers.setdefault("Origin", f"{_u.scheme}://{_u.netloc}")
        except Exception:
            pass
        # Always include baseline browser-like headers
        headers.setdefault("User-Agent", UA)
        headers.setdefault("Accept-Language", "en-US,en;q=0.9")
        headers.setdefault("Connection", "keep-alive")
        
        r = SESS.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)
        r.raise_for_status()
        
        # Check Content-Type header
        content_type = r.headers.get("Content-Type", "").lower()
        if "text/html" in content_type and expected_ext != "html":
            return False, None, f"Server returned HTML instead of {expected_ext}"
        
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<15):
                if chunk:
                    f.write(chunk)
        
        size = os.path.getsize(path)
        
        # Validate downloaded file
        is_valid, reason = is_valid_file(path, expected_ext)
        if not is_valid:
            os.remove(path)  # Remove invalid file
            return False, None, reason
        
        return True, path, f"{size} bytes"
    except Exception as e:
        # Provide more diagnostic info where possible
        return False, None, getattr(e, "args", [str(e)])[0] if e else "unknown error"

def guess_referer_for_url(url, landing_url, pmcid=None):
    """Pick best Referer for a given asset URL."""
    # Prefer publisher landing page if available
    if landing_url:
        return landing_url
    # Else, fall back to PMC article page
    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    # Else, origin of the URL
    try:
        u = urlparse(url)
        return f"{u.scheme}://{u.netloc}/"
    except Exception:
        return None

def download_all(urls, outdir, pmcid=None, landing_url=None, landing_html=None, doi=None, exts=None):
    """
    Download files with validation and fallback to publisher page if PMC links fail.
    """
    dl = []
    failed_urls = []
    ensure_dir(outdir)
    
    for u in urls:
        # Extract expected extension
        expected_ext = None
        if exts:
            for ext in exts:
                if u.lower().endswith(f".{ext}"):
                    expected_ext = ext
                    break
        
        print(f"    Attempting: {u}")
        
        # First attempt: pick best referer
        referer = guess_referer_for_url(u, landing_url, pmcid)
        success, filepath, reason = download_file(u, outdir, referer=referer, expected_ext=expected_ext)
        
        if success:
            print(f"    [OK] {u} -> {filepath} ({reason})")
            dl.append(filepath)
        else:
            print(f"    [FAIL] {u}: {reason}")
            failed_urls.append((u, expected_ext))
    
    # Retry failed URLs from publisher page if landing_url is available
    # Also try to find alternative links on publisher page
    if failed_urls and landing_url and landing_html:
        print(f"\n[RETRY] Attempting {len(failed_urls)} failed URL(s) from publisher page...")
        
        # First, try the same URLs with publisher referer
        still_failed = []
        for u, expected_ext in failed_urls:
            success, filepath, reason = download_file(u, outdir, referer=landing_url, expected_ext=expected_ext)
            if success:
                print(f"    [OK] {u} -> {filepath} ({reason})")
                dl.append(filepath)
            else:
                still_failed.append((u, expected_ext))
        
        # If still failing, try to find alternative links on publisher page
        if still_failed:
            print(f"\n[SEARCH] Searching for alternative links on publisher page...")
            # Extract filenames from failed URLs to find matching files
            failed_filenames = [os.path.basename(urlparse(u).path) for u, _ in still_failed]
            alt_links = extract_links_from_html(landing_url, landing_html, exts, pmc_filenames=failed_filenames)
            # Remove links we already tried
            alt_links = [u for u in alt_links if u not in [f[0] for f in failed_urls]]
            
            if alt_links:
                print(f"    Found {len(alt_links)} alternative link(s) on publisher page.")
                for u in alt_links:
                    expected_ext = None
                    for ext in exts:
                        if u.lower().endswith(f".{ext}"):
                            expected_ext = ext
                            break
                    success, filepath, reason = download_file(u, outdir, referer=landing_url, expected_ext=expected_ext)
                    if success:
                        print(f"    [OK] {u} -> {filepath} ({reason})")
                        dl.append(filepath)
                    else:
                        print(f"    [FAIL] {u}: {reason}")
    
    return dl

def main():
    ap = argparse.ArgumentParser(description="Download supplementary files for a PMC article.")
    ap.add_argument("pmcid", help="e.g., PMC7878295")
    ap.add_argument("-o","--outdir", default="supp_out", help="Output directory (will create pmcid subfolder)")
    ap.add_argument("--exts", default=",".join(SUP_EXTS_DEFAULT), help="Comma-separated whitelist of extensions")
    ap.add_argument("--max-probe", type=int, default=10, help="Max mmcN probes for Elsevier CDN heuristic")
    ap.add_argument("--no-gpt", action="store_true", help="Disable GPT fallback even if OPEN_API_KEY is present")
    args = ap.parse_args()

    pmcid = args.pmcid.strip()
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]

    # Create PMC-specific output directory
    outdir = os.path.join(args.outdir, pmcid)
    ensure_dir(outdir)

    all_links = []

    print(f"[1/5] Checking PMC page for {pmcid} ...")
    pmc_links, doi, pmc_html = find_pmc_links(pmcid, exts)
    if pmc_links:
        print(f"    Found {len(pmc_links)} link(s) on PMC.")
        all_links.extend(pmc_links)
    else:
        print("    No direct PMC supplementary links found.")

    landing_url = None
    landing_html = None
    
    print(f"[2/5] Resolving DOI ...")
    if doi:
        landing_url, landing_html = resolve_doi(doi)
        if landing_url:
            print(f"    DOI resolved -> {landing_url}")
            # Extract PMC filenames to help match on publisher page
            pmc_filenames = [os.path.basename(urlparse(u).path) for u in pmc_links] if pmc_links else None
            pub_links = extract_links_from_html(landing_url, landing_html, exts, pmc_filenames=pmc_filenames)
            if pub_links:
                print(f"    Found {len(pub_links)} link(s) on publisher page.")
                all_links.extend(pub_links)
            else:
                print("    No direct links extracted from publisher HTML.")

            print(f"[3/5] Elsevier CDN heuristic (if applicable) ...")
            cdn_links = guess_elsevier_cdn(landing_url, landing_html, exts, max_probe=args.max_probe)
            if cdn_links:
                print(f"    Found {len(cdn_links)} link(s) via ars.els-cdn heuristic.")
                all_links.extend(cdn_links)
            else:
                print("    No ars.els-cdn matches (or not Elsevier).")
        else:
            print(f"    DOI resolution failed for {doi}.")
            # Try to construct publisher URL from DOI as fallback
            pub_url = None
            if doi:
                # MDPI: 10.3390/pharmaceutics15071980 -> https://www.mdpi.com/1999-4923/15/7/1980
                if "10.3390" in doi:
                    # Try to extract journal and article ID from DOI
                    parts = doi.replace("10.3390/", "").split("/")
                    if len(parts) >= 2:
                        journal = parts[0]
                        article_id = parts[1]
                        # Construct MDPI URL (common pattern)
                        pub_url = f"https://www.mdpi.com/article/10.3390/{doi.replace('10.3390/', '')}"
            
            # Try to extract publisher URL from PMC HTML as additional fallback
            if not pub_url and pmc_html:
                soup = BeautifulSoup(pmc_html, "html.parser")
                # Look for citation_fulltext_url or citation_public_url (not PDF)
                for meta_name in ["citation_fulltext_url", "citation_public_url"]:
                    meta = soup.find("meta", {"name": meta_name})
                    if meta and meta.get("content"):
                        url_candidate = meta.get("content").strip()
                        # Skip PDF URLs
                        if not url_candidate.endswith(".pdf"):
                            pub_url = url_candidate
                            break
                # Look for publisher links in HTML
                if not pub_url:
                    for a in soup.find_all("a", href=True):
                        href = a.get("href", "")
                        # Check for common publisher domains (not PMC, not PDFs)
                        if any(domain in href for domain in ["mdpi.com", "plos.org", "nature.com", "springer.com", "wiley.com", "sciencedirect.com", "hindawi.com"]):
                            if not href.endswith(".pdf") and "pmc.ncbi.nlm.nih.gov" not in href:
                                pub_url = href
                                break
                
                if pub_url:
                    try:
                        headers = {"Referer": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/", "User-Agent": UA}
                        r = SESS.get(pub_url, headers=headers, timeout=30, allow_redirects=True)
                        if r.status_code == 200:
                            landing_url = r.url
                            landing_html = r.text
                            print(f"    Found publisher page -> {landing_url}")
                            pmc_filenames = [os.path.basename(urlparse(u).path) for u in pmc_links] if pmc_links else None
                            pub_links = extract_links_from_html(landing_url, landing_html, exts, pmc_filenames=pmc_filenames)
                            if pub_links:
                                print(f"    Found {len(pub_links)} link(s) on publisher page.")
                                all_links.extend(pub_links)
                        else:
                            print(f"    Publisher page returned {r.status_code}")
                    except Exception as e:
                        print(f"    Could not fetch publisher page: {e}")
    else:
        print("    No DOI found on PMC page.")

    all_links = list(dict.fromkeys(all_links))

    if not all_links and OPENAI_AVAILABLE and not args.no_gpt:
        print(f"[4/5] GPT assist (last resort) ...")
        # Build context with PMC HTML (truncated) and DOI
        pmc_html_snippet = (pmc_html or "")
        if len(pmc_html_snippet) > 10000:
            pmc_html_snippet = pmc_html_snippet[:10000]
        context = f"""
You are given a PubMed Central article context. Identify the likely PUBLISHER DOMAIN and propose direct, downloadable supplementary file URLs (xlsx/csv/xls/tsv/pdf only; no zip) for this article.

Rules:
- Return only actual file URLs from publisher CDNs; avoid HTML landing pages.
- Examples: ars.els-cdn.com (Elsevier); static-content.springer.com, media.springernature.com (Springer/Nature);
  onlinelibrary.wiley.com, wiley.s3.amazonaws.com (Wiley); static-content.production-cdn.plos.org (PLOS);
  embopress.org or "sourceData" asset links (EMBO); mdpi.com (MDPI).
- Prefer links that resemble mmcN, suppl, supporting, mediaobjects, source-data, or similar patterns.
- Do NOT output zip links.

Input:
- PMC ID: {pmcid}
- DOI (may be empty): {doi or '(none)'}
- PMC HTML (truncated):
{pmc_html_snippet}

Output:
- One URL per line. Only direct file links with these extensions: .xlsx, .xls, .csv, .tsv, .pdf
"""
        gpt_urls, gpt_text = ask_gpt_for_links(context)
        # Log GPT prompt/response summary
        print("    [GPT] prompt (first 200 chars):", context[:200].replace("\n", " "))
        print("    [GPT] response (first 200 chars):", (gpt_text or "")[:200].replace("\n", " "))
        if gpt_urls:
            # filter by allowed extensions defensively
            allowed_exts = {"xlsx","xls","csv","tsv","pdf"}
            gpt_urls = [u for u in gpt_urls if any(u.lower().split("?")[0].endswith("."+e) for e in allowed_exts)]
            if gpt_urls:
                print(f"    GPT suggested {len(gpt_urls)} URL(s).")
                all_links.extend(gpt_urls)
            else:
                print("    GPT suggested URLs but none matched allowed extensions.")
        else:
            print("    GPT could not suggest URLs.")
        all_links = list(dict.fromkeys(all_links))

    if not all_links:
        print("[X] No supplementary links found.", file=sys.stderr)
        sys.exit(2)

    print(f"[5/5] Downloading {len(all_links)} file(s) to {outdir}...")
    downloaded = download_all(all_links, outdir, pmcid=pmcid, landing_url=landing_url, landing_html=landing_html, doi=doi, exts=exts)
    if not downloaded:
        print("[X] All downloads failed.", file=sys.stderr)
        sys.exit(3)
    print(f"\nDone. Downloaded {len(downloaded)} valid file(s).")
    for p in downloaded:
        print(f"  {p}")

if __name__ == "__main__":
    main()
