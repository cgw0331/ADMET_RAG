#!/usr/bin/env python3
"""
Download supplements using hybrid strategy (PMC heuristics + optional GPT hints)

Inputs:
  - Base: raws_v1/PMCxxxx with article.pdf present
  - Optional: text_extracted/PMCxxxx/extracted_text.txt

Strategy:
  1) Parse PMC article page to collect likely supplement links (/bin/, download, known extensions)
  2) If enabled and still low yield, ask GPT (title/DOI/PMCID + brief snippet) to suggest candidate URLs
  3) Validate by HEAD/GET and save under raws_v1/PMCxxxx/supplements/

Usage examples:
  python download_supplements.py --limit 5 --use_gpt
  python download_supplements.py --pmc PMC12345678 --use_gpt --dry_run
"""

import os
import re
import json
import time
import argparse
import urllib.parse
from pathlib import Path
from typing import List, Optional, Dict

import requests
from bs4 import BeautifulSoup

OUT_BASE = Path("./raws_v1")
TEXT_BASE = Path("./text_extracted")
DEFAULT_SUPP_ROOT = Path("./Supplements")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (ADMET-Organoid-Collector)",
    "Accept": "*/*",
}

PMC_BASE_WWW = "https://www.ncbi.nlm.nih.gov/pmc/articles"
PMC_BASE_CDN = "https://pmc.ncbi.nlm.nih.gov/articles"

# Only download these types
SUPP_EXTS = (".pdf", ".xlsx", ".xls", ".csv", ".tsv", ".doc", ".docx")


def session_with_retry() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retries = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


SESSION = session_with_retry()


def list_pmc_ids(base: Path) -> List[str]:
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("PMC")])


def get_title_doi_from_pmc(pmcid: str) -> Dict[str, Optional[str]]:
    for base in (PMC_BASE_WWW, PMC_BASE_CDN):
        url = f"{base}/{pmcid}/"
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code != 200 or not r.text:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            title = None
            doi = None

            # meta tags commonly used
            mt = soup.find("meta", attrs={"name": re.compile("citation_title|dc\.title", re.I)})
            if mt and mt.get("content"):
                title = mt["content"].strip()
            mt2 = soup.find("meta", attrs={"name": re.compile("citation_doi|dc\.identifier", re.I)})
            if mt2 and mt2.get("content"):
                c = mt2["content"].strip()
                # dc.identifier may be like doi:10.x/...
                m = re.search(r"10\.\d{4,9}/\S+", c)
                doi = m.group(0) if m else c if c.lower().startswith("10.") else None

            if not title:
                h1 = soup.find("h1")
                if h1:
                    title = h1.get_text(" ", strip=True)

            return {"title": title, "doi": doi, "article_url": url}
        except Exception:
            continue
    return {"title": None, "doi": None, "article_url": None}


def find_supp_links_from_pmc(pmcid: str) -> List[str]:
    found: List[str] = []
    for base in (PMC_BASE_WWW, PMC_BASE_CDN):
        article_url = f"{base}/{pmcid}/"
        try:
            r = SESSION.get(article_url, timeout=30)
            if r.status_code != 200 or not r.text:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.select("a[href]"):
                href = a.get("href", "").strip()
                if not href:
                    continue
                abs_url = href if href.startswith("http") else urllib.parse.urljoin(article_url, href)
                low = abs_url.lower()
                # Only accept direct links with desired extensions
                if low.endswith(SUPP_EXTS):
                    found.append(abs_url)
        except Exception:
            continue
    # dedupe preserving order
    uniq = []
    for u in found:
        if u not in uniq:
            uniq.append(u)
    return uniq


def read_text_snippet(pmcid: str, max_chars: int = 1200) -> Optional[str]:
    p = TEXT_BASE / pmcid / "extracted_text.txt"
    if not p.exists():
        return None
    try:
        s = p.read_text(encoding="utf-8", errors="ignore")
        return s[:max_chars]
    except Exception:
        return None


def gpt_suggest_urls(pmcid: str, title: Optional[str], doi: Optional[str], snippet: Optional[str], model: str = "gpt-4o", max_urls: int = 6, debug: bool = False) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = {
            "pmcid": pmcid,
            "title": title,
            "doi": doi,
            "hint": "Find direct downloadable supplementary files if they exist.",
            "rules": [
                "Only return direct downloadable URLs if reasonably certain",
                "Prefer PMC, publisher pages, figshare, zenodo, dryad, supporting info",
                "Return JSON: {\"urls\": [<url>...]}"
            ],
            "snippet": snippet or "",
        }
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ],
            temperature=0.1,
            max_tokens=400,
        )
        txt = resp.choices[0].message.content or ""
        txt = txt.strip()
        if debug:
            print(f"[HINT OPENAI pmc={pmcid}] prompt=", json.dumps(prompt, ensure_ascii=False)[:500])
            print(f"[HINT OPENAI pmc={pmcid}] raw=", txt[:500])
        if txt.startswith("```json"):
            txt = txt[7:]
        if txt.endswith("```"):
            txt = txt[:-3]
        data = json.loads(txt)
        urls = data.get("urls", []) if isinstance(data, dict) else []
        urls = [u for u in urls if isinstance(u, str)]
        return urls[:max_urls]
    except Exception:
        return []


def is_valid_supp_url(url: str) -> bool:
    low = url.lower()
    # Strictly allow only desired extensions
    return any(low.endswith(ext) for ext in SUPP_EXTS)


def download_file(url: str, out_path: Path, referer: Optional[str] = None) -> bool:
    headers = {}
    if referer:
        headers["Referer"] = referer
    try:
        with SESSION.get(url, headers=headers, timeout=90, stream=True, allow_redirects=True) as r:
            if r.status_code != 200:
                return False
            # HTML guard by Content-Type
            ctype = (r.headers.get("Content-Type") or "").lower()
            # Peek first chunk
            first_chunk = b""
            for chunk in r.iter_content(8192):
                if chunk:
                    first_chunk = chunk
                    break
            # Detect PMC POW/HTML placeholders
            if ("text/html" in ctype) or (first_chunk.strip().lower().startswith(b"<html")) or (b"cloudpmc-viewer-pow" in first_chunk):
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                if first_chunk:
                    f.write(first_chunk)
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def save_urls(urls: List[str], supp_pmc_dir: Path):
    if not urls:
        return
    supp_pmc_dir.mkdir(parents=True, exist_ok=True)
    with open(supp_pmc_dir / "supplement_urls.json", "w", encoding="utf-8") as f:
        json.dump({"urls": urls}, f, ensure_ascii=False, indent=2)


def write_summary(pmcid: str, supp_pmc_dir: Path, urls: List[str], saved_files: List[Path], ext_stats: Dict[str, int] | None = None):
    summary = supp_pmc_dir / "supplements_summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"PMCID: {pmcid}\n")
        f.write(f"Total URLs found: {len(urls)}\n")
        f.write(f"Saved files: {len(saved_files)}\n")
        if ext_stats:
            f.write("\nExtension stats (all candidates):\n")
            for k, v in sorted(ext_stats.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"  .{k}: {v}\n")
        f.write("\nSaved list:\n")
        for p in saved_files:
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            f.write(f"  - {p.name} ({size} bytes)\n")
        if urls:
            f.write("\nAll candidate URLs:\n")
            for u in urls:
                f.write(f"  - {u}\n")


def llm_compound_relevance(provider: str, openai_model: str, ollama_model: str, context: Dict[str, Optional[str]], debug: bool = False) -> bool:
    """Return True if LLM judges the supplement likely contains compound/assay data.
    Context keys: pmcid, title, doi, url, filename, anchor_text, snippet
    """
    question = {
        "task": "Decide if a supplementary file likely contains compound-related experimental data",
        "criteria": [
            "Contains assay results, IC50/EC50/Ki/Km, Dose/Response, concentration tables",
            "Contains chemical identifiers (SMILES, InChI, CAS), compound names, structures",
            "Contains ADMET metrics (Caco-2, MDCK, PAMPA, BBB, PPB, Vd, CL, t1/2, CYP, hERG, DILI, Ames, carcinogenicity)",
            "If it's only forms, cover letters, licenses, figure captions only -> NO"
        ],
        "return_format": "JSON {\"relevant\": true|false, \"reason\": <short>}"
    }
    prompt = {
        "question": question,
        "context": {k: v for k, v in context.items() if v}
    }
    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return False
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "Answer in JSON only."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
                ],
                temperature=0.1,
                max_tokens=200,
            )
            txt = (resp.choices[0].message.content or "").strip()
            if debug:
                print(f"[REL OPENAI] ctx=", json.dumps(context, ensure_ascii=False)[:500])
                print(f"[REL OPENAI] raw=", txt[:500])
        else:
            # ollama local
            import ollama
            client = ollama.Client()
            response = client.chat(
                model=ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=False) + "\nRespond JSON: {\"relevant\": true|false, \"reason\": \"\"}"
                    }
                ],
                options={"temperature": 0.1}
            )
            txt = (response.get("message", {}).get("content", "") or "").strip()
            if debug:
                print(f"[REL OLLAMA] ctx=", json.dumps(context, ensure_ascii=False)[:500])
                print(f"[REL OLLAMA] raw=", txt[:500])
        if txt.startswith("```json"):
            txt = txt[7:]
        if txt.endswith("```"):
            txt = txt[:-3]
        data = json.loads(txt)
        return bool(data.get("relevant", False))
    except Exception:
        return False


def process_one(pmcid: str, use_gpt: bool, dry_run: bool, max_downloads: int = 12, sleep_sec: float = 0.2,
                filter_compound: bool = False, llm_provider: str = "openai",
                openai_model: str = "gpt-4o-mini", ollama_model: str = "llama3.1",
                supp_root: Path = DEFAULT_SUPP_ROOT, debug: bool = False) -> int:
    pmc_dir = OUT_BASE / pmcid
    supp_pmc_dir = supp_root / pmcid
    supp_pmc_dir.mkdir(parents=True, exist_ok=True)

    # 1) Deterministic from PMC page
    links = find_supp_links_from_pmc(pmcid)

    # 2) GPT hints (optional)
    if use_gpt and len(links) < 2:
        meta = get_title_doi_from_pmc(pmcid)
        snippet = read_text_snippet(pmcid)
        hints = gpt_suggest_urls(pmcid, meta.get("title"), meta.get("doi"), snippet, debug=debug)
        # keep only plausible urls
        hints = [u for u in hints if is_valid_supp_url(u)]
        for u in hints:
            if u not in links:
                links.append(u)

    # save urls record
    save_urls(links, supp_pmc_dir)

    # 3) Download
    saved = 0
    saved_paths: List[Path] = []
    # track extension stats across all candidate links
    ext_stats: Dict[str, int] = {}
    total_links = len(links)
    print(f"  [{pmcid}] {total_links}개 링크 확인 중...")
    for idx, u in enumerate(links, 1):
        if saved >= max_downloads:
            break
        name = urllib.parse.unquote(u.split("/")[-1].split("?")[0]) or f"supp_{saved+1}"
        name = re.sub(r"[^a-zA-Z0-9._\-]+", "_", name)
        out = supp_pmc_dir / name
        # update extension stats
        ext = (Path(name).suffix or "").lower().lstrip(".")
        if ext:
            ext_stats[ext] = ext_stats.get(ext, 0) + 1
        if out.exists():
            print(f"    [{idx}/{total_links}] {name} 이미 존재 - 건너뜀")
            continue
        # optional LLM-based relevance check (pre-download)
        if filter_compound:
            print(f"    [{idx}/{total_links}] {name} 화합물 관련성 확인 중...", end="\r", flush=True)
            snippet = read_text_snippet(pmcid, max_chars=1200)
            meta = get_title_doi_from_pmc(pmcid)
            anchor_text = None  # not tracked here; kept for future enhancement
            ctx = {
                "pmcid": pmcid,
                "title": meta.get("title"),
                "doi": meta.get("doi"),
                "url": u,
                "filename": name,
                "anchor_text": anchor_text,
                "snippet": snippet,
            }
            is_rel = llm_compound_relevance(llm_provider, openai_model, ollama_model, ctx, debug=debug)
            if not is_rel:
                print(f"    [{idx}/{total_links}] {name} 관련 없음 - 건너뜀")
                continue
            print(f"    [{idx}/{total_links}] {name} 관련 있음 - 다운로드 시도...")
        else:
            print(f"    [{idx}/{total_links}] {name} 다운로드 시도...", end="\r", flush=True)
        if dry_run:
            print(f"    [{idx}/{total_links}] [DRY] {pmcid} -> {u}")
            saved += 1
            continue
        if download_file(u, out, referer=str(pmc_dir)):
            saved += 1
            saved_paths.append(out)
            file_size = out.stat().st_size if out.exists() else 0
            print(f"    [{idx}/{total_links}] ✓ {name} 저장됨 ({file_size} bytes)")
            time.sleep(sleep_sec)
        else:
            print(f"    [{idx}/{total_links}] ✗ {name} 다운로드 실패")
    # write summary (with extension stats)
    write_summary(pmcid, supp_pmc_dir, links, saved_paths, ext_stats)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmc", type=str, default=None, help="Single PMCID (e.g., PMC12345678)")
    ap.add_argument("--limit", type=int, default=0, help="Process first N PMC folders (0=all)")
    ap.add_argument("--use_gpt", action="store_true", help="Enable GPT hints for supplement URLs")
    ap.add_argument("--dry_run", action="store_true", help="Do not download, only list")
    ap.add_argument("--max_per_pmc", type=int, default=12)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--filter_compound", action="store_true", help="Use LLM to download only compound-related supplements")
    ap.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "ollama"], help="LLM provider")
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--ollama_model", type=str, default="llama4")
    ap.add_argument("--supp_root", type=str, default=str(DEFAULT_SUPP_ROOT), help="Root folder to save supplements (default: ./Supplements)")
    args = ap.parse_args()
    debug = False
    if hasattr(args, "debug"):
        debug = getattr(args, "debug")

    targets: List[str]
    if args.pmc:
        targets = [args.pmc]
    else:
        targets = list_pmc_ids(OUT_BASE)
        if args.limit and args.limit > 0:
            targets = targets[:args.limit]

    total_saved = 0
    for i, pmcid in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {pmcid} supplements...")
        saved = process_one(
            pmcid,
            use_gpt=args.use_gpt,
            dry_run=args.dry_run,
            max_downloads=args.max_per_pmc,
            sleep_sec=args.sleep,
            filter_compound=args.filter_compound,
            llm_provider=args.llm_provider,
            openai_model=args.openai_model,
            ollama_model=args.ollama_model,
            supp_root=Path(args.supp_root),
            debug=debug,
        )
        print(f"  -> saved {saved}")
        total_saved += saved

    print(f"[DONE] total_saved={total_saved}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
보충자료 다운로드 스크립트
- PMC 논문의 보충자료 자동 다운로드
- LLM 검증 + web scraping
"""

import os
import re
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import ollama

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_pmc_id_from_text(text_file_path: Path) -> str:
    """
    텍스트 파일에서 PMC ID 추출
    
    Args:
        text_file_path: 텍스트 파일 경로
        
    Returns:
        PMC ID (예: "PMC7878295")
    """
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # PMC ID 패턴 찾기
    pmc_pattern = r'PMC\d+'
    match = re.search(pmc_pattern, text)
    
    if match:
        return match.group()
    
    # 파일명이나 폴더명에서 추출
    if "PMC" in str(text_file_path):
        parts = str(text_file_path).split('/')
        for part in parts:
            if "PMC" in part:
                return part
    
    return None


def check_supplementary_existence(pmc_id: str) -> Dict[str, Any]:
    """
    PMC에서 보충자료 존재 여부 확인
    
    Args:
        pmc_id: PMC ID
        
    Returns:
        보충자료 정보 딕셔너리
    """
    logger.info(f"보충자료 확인 중: {pmc_id}")
    
    try:
        # PMC 페이지 접근
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 보충자료 섹션 찾기
        supp_links = []
        
        # 보충자료 다운로드 섹션 찾기
        # PMC 보충자료는 보통 "Additional files" 또는 "Download" 섹션에 있음
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            text = link.get_text().lower().strip()
            
            # 보충자료 관련 링크 찾기
            if any(keyword in text for keyword in ['download', 'supplement', 'additional', 'pdf', 'excel', 'table', 'data']):
                # 실제 파일 다운로드 링크인지 확인
                if any(ext in href.lower() for ext in ['.pdf', '.xls', '.xlsx', '.doc', '.docx', '.zip', '.csv']):
                    supp_links.append(link)
                elif 'download' in href.lower() or 'suppl' in href.lower():
                    supp_links.append(link)
        
        # PMC의 보충자료 다운로드 표준 URL 시도
        # PMC 보충자료는 보통 https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/bin/suppl/ 형식
        
        # PMC 표준 보충자료 URL 시도
        suppl_base_urls = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/bin/suppl/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/supplement/",
        ]
        
        # 표준 보충자료 파일 시도
        standard_files = []
        for base_url in suppl_base_urls:
            try:
                # SD1, SD2, SD3 등 시도
                for i in range(1, 21):  # 최대 20개 파일
                    test_url = f"{base_url}#SD{i}"
                    # HTML이 아닌 실제 파일 다운로드 URL 확인 필요
            except:
                pass
        
        # 실제 파일 링크만 추출
        real_file_links = []
        for link in supp_links:
            href = link.get('href', '')
            if href and not href.startswith('#'):
                real_file_links.append(link)
        
        result = {
            "pmc_id": pmc_id,
            "has_supplementary": len(supp_links) > 0,
            "supplement_count": len(supp_links),
            "links": [link.get('href', '') for link in supp_links[:10]],  # 처음 10개만
            "url": url
        }
        
        logger.info(f"보충자료 발견: {result['has_supplementary']} ({result['supplement_count']}개)")
        return result
        
    except Exception as e:
        logger.error(f"보충자료 확인 실패: {e}")
        return {"pmc_id": pmc_id, "has_supplementary": False, "error": str(e)}


def download_supplements(pmc_id: str, output_dir: Path, supplement_info: Dict[str, Any]) -> bool:
    """
    보충자료 다운로드
    
    Args:
        pmc_id: PMC ID
        output_dir: 저장 디렉토리
        supplement_info: 보충자료 정보
        
    Returns:
        성공 여부
    """
    logger.info(f"보충자료 다운로드 시작: {pmc_id}")
    
    if not supplement_info.get('has_supplementary', False):
        logger.warning("보충자료가 없습니다.")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    
    for i, href in enumerate(supplement_info.get('links', []), 1):
        try:
            # 전체 URL 생성
            if href.startswith('/'):
                full_url = f"https://www.ncbi.nlm.nih.gov{href}"
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/{href}"
            
            # 파일명 추출
            filename = href.split('/')[-1]
            if not filename or filename == '#' or filename == '':
                filename = f"supplement_{i}.pdf"
            else:
                # 확장자 확인 및 추가
                if '.' not in filename:
                    filename = f"{filename}_{i}"
                else:
                    # 파일명이 이미 있는 경우 인덱스 추가
                    name, ext = filename.rsplit('.', 1)
                    filename = f"{name}_{i}.{ext}"
            
            file_path = output_dir / filename
            
            # 다운로드
            logger.info(f"다운로드 중 ({i}/{len(supplement_info['links'])}): {filename}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(full_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"다운로드 완료: {file_path}")
            downloaded += 1
            
        except Exception as e:
            logger.error(f"다운로드 실패: {e}")
            continue
    
    if downloaded > 0:
        logger.info(f"총 {downloaded}개 파일 다운로드 완료")
        return True
    else:
        logger.warning("다운로드된 파일이 없습니다.")
        return False


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="보충자료 다운로드")
    parser.add_argument("--text_file", required=True, help="텍스트 파일 경로 (extracted_text.txt)")
    parser.add_argument("--output_dir", default="./Supplements", help="보충자료 저장 디렉토리")
    parser.add_argument("--pmc_id", help="PMC ID (자동 추출 시 생략 가능)")
    
    args = parser.parse_args()
    
    # PMC ID 추출
    text_file = Path(args.text_file)
    if args.pmc_id:
        pmc_id = args.pmc_id
    else:
        pmc_id = extract_pmc_id_from_text(text_file)
    
    if not pmc_id:
        logger.error("PMC ID를 찾을 수 없습니다.")
        return
    
    logger.info(f"PMC ID: {pmc_id}")
    
    # 보충자료 확인
    supplement_info = check_supplementary_existence(pmc_id)
    
    # 결과 저장
    result_file = Path(args.output_dir) / pmc_id / "supplement_info.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(supplement_info, f, ensure_ascii=False, indent=2)
    logger.info(f"보충자료 정보 저장: {result_file}")
    
    # 보충자료 다운로드
    if supplement_info.get('has_supplementary', False):
        output_dir = Path(args.output_dir) / pmc_id
        success = download_supplements(pmc_id, output_dir, supplement_info)
        
        if success:
            print(f"✅ 보충자료 다운로드 완료!")
            print(f"  저장 위치: {output_dir}")
        else:
            print("❌ 보충자료 다운로드 실패!")
    else:
        print("보충자료가 없습니다.")


if __name__ == "__main__":
    main()

