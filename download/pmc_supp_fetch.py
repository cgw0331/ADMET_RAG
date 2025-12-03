#!/usr/bin/env python3
import os, re, time, pathlib, urllib.parse, json, sys
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

UA = "Mozilla/5.0 (ADMET Supplement Fetcher; +https://www.ncbi.nlm.nih.gov/pmc/)"
BASE = "https://www.ncbi.nlm.nih.gov"
OUT = pathlib.Path("Supplements"); OUT.mkdir(exist_ok=True)

ALLOWED_EXTS = (".pdf", ".xlsx", ".xls", ".csv", ".tsv", ".doc", ".docx")

class FetchError(Exception):
    pass

def _session(referer=None):
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    if referer:
        s.headers.update({"Referer": referer})
    return s

@retry(reraise=True, stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(FetchError))
def _get(s, url, stream=False):
    r = s.get(url, allow_redirects=True, timeout=30, stream=stream)
    if r.status_code >= 400:
        raise FetchError(f"{r.status_code} {url}")
    return r

def pmc_article_url(pmcid: str) -> str:
    pmcid = pmcid.strip().upper().replace("PMC", "")
    return f"{BASE}/pmc/articles/PMC{pmcid}/"

def parse_pmc_supp_urls(pmcid: str):
    """Extract candidate supplement links from PMC article page."""
    art = pmc_article_url(pmcid)
    s = _session(referer=art)
    html = _get(s, art).text
    soup = BeautifulSoup(html, "html.parser")

    # Try DOI for publisher hop
    doi = None
    m = soup.find("meta", attrs={"name": re.compile("citation_doi|dc.identifier", re.I)})
    if m and m.get("content"):
        c = m.get("content").strip()
        mdoi = re.search(r"10\.\d{4,9}/\S+", c)
        doi = mdoi.group(0) if mdoi else (c if c.lower().startswith("10.") else None)

    candidates = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not href:
            continue
        text = (a.get_text() or "").strip()
        u = urllib.parse.urljoin(art, href)
        lo = u.lower()
        # retain only allowed direct types
        if any(lo.endswith(ext) for ext in ALLOWED_EXTS):
            candidates.append((u, text))
        # some publishers need download=1
        if lo.endswith(".pdf") and "download=1" not in lo and ("?pdf=" in lo or "/pdf" in lo):
            candidates.append((u + ("&" if "?" in u else "?") + "download=1", text))

    # dedupe by URL without fragment
    uniq = []
    seen = set()
    for u, t in candidates:
        key = u.split("#")[0]
        if key not in seen:
            seen.add(key)
            uniq.append((u, t))
    return uniq, art, doi

def parse_publisher_links_from_doi(doi: str):
    """Follow DOI to publisher, parse page for allowed resources."""
    if not doi:
        return [] , None
    doi_url = f"https://doi.org/{doi}"
    s = _session(referer=doi_url)
    try:
        r = _get(s, doi_url)
        pub_url = r.url
        pub_html = r.text
        soup = BeautifulSoup(pub_html, "html.parser")
        outs = []
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            u = urllib.parse.urljoin(pub_url, href)
            if any(u.lower().endswith(ext) for ext in ALLOWED_EXTS):
                outs.append((u, a.get_text(strip=True) or ""))
        # dedupe
        seen = set(); uniq = []
        for u, t in outs:
            k = u.split('#')[0]
            if k not in seen:
                seen.add(k); uniq.append((u, t))
        return uniq, pub_url
    except Exception:
        return [] , None

def _filename_from_cd(r, fallback):
    cd = r.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\'|\")?([^";]+)', cd, flags=re.I)
    if m:
        name = urllib.parse.unquote(m.group(1))
    else:
        name = pathlib.Path(urllib.parse.urlparse(fallback).path).name or "download.bin"
    name = re.sub(r"[^\w.\-()+\[\] ]+", "_", name).strip("._ ")
    return name[:200] or "file.bin"

def _looks_html(r, first_chunk: bytes) -> bool:
    ct = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in ct:
        return True
    if first_chunk.strip().lower().startswith(b"<html"):
        return True
    if b"cloudpmc-viewer-pow" in first_chunk:
        return True
    return False

def download_pmc_supplements(pmcid: str, max_files: int = 12):
    urls, referer, doi = parse_pmc_supp_urls(pmcid)
    if not urls:
        # Try publisher via DOI
        pub_links, pub_ref = parse_publisher_links_from_doi(doi)
        if pub_links:
            urls = pub_links
            referer = pub_ref or referer
        else:
            print(f"[{pmcid}] 보충자료 링크를 찾지 못했어요.")
            return []

    s = _session(referer=referer)
    saved = []
    outdir = OUT / pmcid
    outdir.mkdir(exist_ok=True, parents=True)
    for i, (u, label) in enumerate(urls, 1):
        if i > max_files: break
        try:
            r = _get(s, u, stream=True)
            # peek first chunk
            first = b""
            for chunk in r.iter_content(8192):
                if chunk:
                    first = chunk
                    break
            # HTML guard
            if _looks_html(r, first):
                # try download=1 param once
                if "download=1" not in u:
                    u2 = u + ("&" if "?" in u else "?") + "download=1"
                    r = _get(s, u2, stream=True)
                    first = b""
                    for chunk in r.iter_content(8192):
                        if chunk:
                            first = chunk
                            break
                    if _looks_html(r, first):
                        print(f"  - HTML/차단 응답: {u}")
                        continue
                else:
                    print(f"  - HTML/차단 응답: {u}")
                    continue

            name = _filename_from_cd(r, u)
            # enforce allowed extensions
            ext = pathlib.Path(name).suffix.lower()
            if ext not in ALLOWED_EXTS:
                # try infer from content-type
                ct = (r.headers.get("Content-Type") or "").lower()
                if "pdf" in ct:
                    name += ".pdf"; ext = ".pdf"
                elif "excel" in ct or "spreadsheet" in ct:
                    name += ".xlsx"; ext = ".xlsx"
                elif "csv" in ct:
                    name += ".csv"; ext = ".csv"
                elif "tsv" in ct:
                    name += ".tsv"; ext = ".tsv"
                elif "msword" in ct or "doc" in ct:
                    name += ".doc"; ext = ".doc"
            if ext not in ALLOWED_EXTS:
                print(f"  - 허용되지 않은 확장자 스킵: {name}")
                continue

            out = outdir / name
            with open(out, "wb") as f:
                if first:
                    f.write(first)
                for chunk in r.iter_content(1024*256):
                    if chunk: f.write(chunk)
            saved.append(str(out))
            print(f"  ✓ {name}")
            time.sleep(0.2)
        except Exception as e:
            print(f"  × 실패: {u} -> {e}")
            continue
    return saved

if __name__ == "__main__":
    pmc = sys.argv[1] if len(sys.argv) > 1 else "PMC7878295"
    files = download_pmc_supplements(pmc)
    print(json.dumps({"pmc": pmc, "files": files}, ensure_ascii=False, indent=2))


