#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, argparse, pathlib, urllib.parse, hashlib
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")  # required for Unpaywall API
NCBI_TOOL = os.getenv("NCBI_TOOL", "knu_admet_downloader")  # polite identification
HEADERS = {"User-Agent": "Mozilla/5.0 (ADMET Downloader; +https://www.ncbi.nlm.nih.gov/books/NBK25500/)"}
OUTDIR = pathlib.Path("raws_v1"); OUTDIR.mkdir(exist_ok=True)

class DownloadError(Exception):
    pass

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), retry=retry_if_exception_type(DownloadError))
def _http_get(url, stream=False, timeout=20, headers=None, params=None):
    h = dict(HEADERS); h.update(headers or {})
    r = requests.get(url, headers=h, timeout=timeout, stream=stream, allow_redirects=True, params=params)
    if r.status_code >= 400:
        raise DownloadError(f"GET {url} -> {r.status_code}")
    return r

def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._\-]+", "_", s).strip("_")
    return s[:200] if len(s) > 200 else s

def _write_response_pdf(r: requests.Response, out_path: pathlib.Path) -> bool:
    head = r.raw.read(5, decode_content=True)
    # allow writing even if content-type is wrong as long as signature is PDF
    if head != b"%PDF-":
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(head)
        for chunk in r.iter_content(chunk_size=1024*256):
            if chunk:
                f.write(chunk)
    return True

def _download_pdf_once(url: str, out_path: pathlib.Path, referer: str | None = None) -> bool:
    h = {"Accept": "application/pdf"}
    if referer:
        h["Referer"] = referer
    r = _http_get(url, stream=True, headers=h)
    ok = _write_response_pdf(r, out_path)
    if ok:
        return True
    return False

# ===== Save PDF with multi-candidate fallback =====
def save_pdf(url: str, suggested_name: str, referer: str | None = None):
    fname = _sanitize_filename(suggested_name) or "paper"
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    # folder decision
    folder = OUTDIR / fname.split(".")[0]
    if fname.upper().startswith("PMC"):
        folder = OUTDIR / fname.split(".")[0]
    elif fname.upper().startswith("DOI_"):
        folder = OUTDIR / fname.split(".")[0]
    else:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        folder = OUTDIR / f"AUTO_{h}"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "article.pdf"

    # Build candidate list
    candidates: list[str] = [url]
    if url.endswith("/"):
        candidates.append(url + "?download=1")
    elif "/pdf/" in url and "?" not in url:
        candidates.append(url + "?download=1")

    # PMC fallback by pmcid
    pmc_match = re.search(r"PMC\d+", url, re.I)
    pmcid = pmc_match.group(0) if pmc_match else None
    if pmcid:
        candidates.extend([
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
            f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
        ])

    tried = set()
    for c in candidates:
        if c in tried:
            continue
        tried.add(c)
        try:
            if _download_pdf_once(c, out, referer=referer):
                return str(out)
        except Exception:
            continue
    raise DownloadError("All PDF candidates failed")

# ===== 1) PubMed PMID -> PMCID -> PMC PDF =====
def pmid_to_pmcid(pmid: str) -> str | None:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "json",
        "tool": NCBI_TOOL,
    }
    j = requests.get(base, params=params, headers={"Accept": "application/json"}, timeout=20).json()
    try:
        linksets = j["linksets"][0]
        for link in linksets.get("linksetdbs", []):
            if link.get("dbto") == "pmc":
                pmcid = link["links"][0]
                return pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
    except Exception:
        return None
    return None

def find_pmc_pdf_url(pmcid: str) -> str | None:
    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    html = _http_get(article_url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", {"name": "citation_pdf_url"})
    if meta and meta.get("content"):
        return meta["content"]
    candidates = []
    for a in soup.select("a[href]"):
        href = a["href"]
        text = (a.get_text() or "").strip().lower()
        if "pdf" in text or href.lower().endswith(".pdf") or "/pdf" in href.lower():
            candidates.append(urllib.parse.urljoin(article_url, href))
    seen = []
    for c in candidates:
        u = c.split("#")[0]
        if u not in seen:
            seen.append(u)
    if seen:
        return seen[0]
    # Europe PMC render fallback
    return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"

def download_from_pmid(pmid: str):
    pmcid = pmid_to_pmcid(pmid)
    if not pmcid:
        print(f"[PMID {pmid}] PMC 오픈액세스 본문이 없거나 PMCID 매핑 실패.")
        return None
    pdf_url = find_pmc_pdf_url(pmcid)
    if not pdf_url:
        print(f"[{pmcid}] PDF 링크를 찾지 못함.")
        return None
    out = save_pdf(pdf_url, f"{pmcid}")
    print(f"[PMID {pmid}] -> {pmcid} PDF saved: {out}")
    return out

# ===== 2) DOI -> Unpaywall -> best_oa_location.url_for_pdf =====
def doi_to_oa_pdf(doi: str) -> str | None:
    if not UNPAYWALL_EMAIL:
        print("UNPAYWALL_EMAIL 환경변수가 필요합니다.")
        return None
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}"
    j = requests.get(url, params={"email": UNPAYWALL_EMAIL}, headers={"Accept": "application/json"}, timeout=20).json()
    best = j.get("best_oa_location") or {}
    pdf = (best.get("url_for_pdf") or "").strip()
    if pdf:
        return pdf
    for loc in j.get("oa_locations", []):
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None

def download_from_doi(doi: str):
    pdf_url = doi_to_oa_pdf(doi)
    if not pdf_url:
        print(f"[DOI {doi}] OA PDF 링크를 찾지 못함(페이월 가능).")
        return None
    out = save_pdf(pdf_url, f"DOI_{_sanitize_filename(doi)}")
    print(f"[DOI {doi}] PDF saved: {out}")
    return out

# ===== 3) Generic URL -> PDF link(s) =====
def extract_pdf_links_naive(page_url: str) -> list[str]:
    html = _http_get(page_url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        abs_url = urllib.parse.urljoin(page_url, href)
        if abs_url.lower().endswith(".pdf") or "/pdf" in abs_url.lower():
            links.append(abs_url)
    uniq = []
    for u in links:
        if u not in uniq:
            uniq.append(u)
    return uniq

def guess_filename_from_url(u: str) -> str:
    parsed = urllib.parse.urlparse(u)
    name = pathlib.Path(parsed.path).name or "download"
    if ".pdf" not in name.lower():
        name += ".pdf"
    return _sanitize_filename(name)

# ===== 3-b) (Optional) GPT API로 PDF 링크 추출 보조 =====
def gpt_suggest_pdf_links(page_url: str, html: str, top_k: int = 3) -> list[str]:
    if not OPENAI_API_KEY:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"Extract up to {top_k} direct PDF links from this page. URL: {page_url}. Return JSON list only. HTML (truncated):\n{html[:15000]}"
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"\[.*\]", txt, flags=re.S)
        if not m:
            return []
        urls = json.loads(m.group(0))
        return [urllib.parse.urljoin(page_url, u) for u in urls][:top_k]
    except Exception as e:
        print(f"[GPT] link suggestion failed: {e}")
        return []

def download_from_url(url: str, allow_gpt=True):
    # If it's a PMC article page, derive referer for subsequent PDF fetches
    referer = url if "/pmc/articles/" in url else None
    pdfs = extract_pdf_links_naive(url)
    if not pdfs and allow_gpt and OPENAI_API_KEY:
        html = _http_get(url, timeout=20).text
        pdfs = gpt_suggest_pdf_links(url, html)
    # If still nothing and it's a PMC page, try standard patterns by detecting pmcid
    if not pdfs and "/pmc/articles/" in url:
        m = re.search(r"PMC\d+", url, re.I)
        if m:
            pmcid = m.group(0)
            pdfs = [
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1",
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
                f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
            ]
    if not pdfs:
        print(f"[URL] PDF 링크 발견 못함: {url}")
        return None
    first = pdfs[0]
    name = guess_filename_from_url(first)
    out = save_pdf(first, name, referer=referer)
    print(f"[URL] PDF saved: {out}")
    return out

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser(description="Automatic PDF downloader for PubMed/PMC, DOI via Unpaywall, and arbitrary pages (GPT-assisted).")
    ap.add_argument("--pmids", nargs="*", default=[], help="One or more PubMed IDs")
    ap.add_argument("--dois", nargs="*", default=[], help="One or more DOIs")
    ap.add_argument("--urls", nargs="*", default=[], help="One or more generic URLs")
    args = ap.parse_args()

    for pmid in args.pmids:
        try:
            download_from_pmid(str(pmid).strip())
        except Exception as e:
            print(f"[PMID {pmid}] error: {e}")

    for doi in args.dois:
        try:
            download_from_doi(doi.strip())
        except Exception as e:
            print(f"[DOI {doi}] error: {e}")

    for u in args.urls:
        try:
            download_from_url(u.strip())
        except Exception as e:
            print(f"[URL {u}] error: {e}")

if __name__ == "__main__":
    main()
