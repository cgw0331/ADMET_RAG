#!/usr/bin/env python3
"""
효율적인 GPT 기반 논문 수집 시스템
- 키워드 사전 필터링으로 GPT 호출 최소화
- 배치 검증으로 토큰 절약
"""

import os
import re
import json
import csv
import time
import math
import shutil
import zipfile
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from lxml import etree
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
from dateutil import parser as dateparser
import openai
from dotenv import load_dotenv

from Bio import Entrez

# 환경변수 로드
load_dotenv()

EMAIL = os.getenv("NCBI_EMAIL", "chlrjs3@kangwon.ac.kr")
API_KEY = os.getenv("NCBI_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")

# OpenAI 클라이언트 초기화
openai.api_key = OPENAI_API_KEY

# 기간 제한 없음 - 모든 연도의 논문 수집
DATE_FROM = "2010/01/01"  # 매우 넓은 범위
DATE_TO = datetime.now().strftime("%Y/%m/%d")

# Organoid 기본 키워드
BASE_ORGANOID = '('
BASE_ORGANOID += 'organoid[tiab] OR organoids[tiab] OR organoid*[tiab] '
BASE_ORGANOID += 'OR Organoids[mh] OR organoid-derived[tiab]'
BASE_ORGANOID += ')'

# PMC 자유원문 필터
PMC_SUBSET = '"pubmed pmc"[sb]'

OUT_DIR = Path("./raws_gpt")
LOG_DIR = Path("./logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# NCBI rate limit
RATE_SLEEP = 0.34 if API_KEY else 0.5
HTTP_TIMEOUT = 30
RETRY = 3

def sleep():
    time.sleep(RATE_SLEEP)

def http_get(url, params=None, stream=False):
    for i in range(RETRY):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT, stream=stream, headers={"User-Agent":"Mozilla/5.0"})
            if 200 <= r.status_code < 300:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2**i, 10))
                continue
            r.raise_for_status()
        except Exception:
            if i == RETRY-1:
                raise
            time.sleep(min(2**i, 10))
    raise RuntimeError("GET failed: " + url)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_bin(content: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(content)

def sanitize_filename(name: str, maxlen=150) -> str:
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
    if len(name) > maxlen:
        name = name[:maxlen]
    return name

def has_admet_keywords(text: str) -> bool:
    """
    텍스트에서 ADMET 키워드가 있는지 간단히 확인
    """
    text_lower = text.lower()
    
    # 주요 ADMET 키워드들
    admet_keywords = [
        'caco-2', 'permeability', 'papp', 'mdck', 'pampa',
        'lipinski', 'rule of five', 'logd', 'logs', 'pka',
        'protein binding', 'ppb', 'blood-brain barrier', 'bbb',
        'volume of distribution', 'vd', 'cyp1a2', 'cyp2c9', 
        'cyp2c19', 'cyp2d6', 'cyp3a4', 'cyp inhibition',
        'clearance', 'half-life', 't1/2', 'herg', 'qt prolongation',
        'dili', 'hepatotoxicity', 'liver injury', 'ames test',
        'mutagenicity', 'carcinogenic', 'genotoxic'
    ]
    
    # 키워드가 2개 이상 있으면 True
    keyword_count = sum(1 for keyword in admet_keywords if keyword in text_lower)
    return keyword_count >= 2

def validate_papers_batch(titles_abstracts: List[Dict]) -> List[Dict]:
    """
    여러 논문을 한 번에 GPT로 검증 (토큰 절약)
    """
    if not titles_abstracts:
        return []
    
    try:
        # 배치로 검증할 논문들 정보 구성
        papers_info = []
        for i, paper in enumerate(titles_abstracts):
            papers_info.append(f"""
논문 {i+1}:
제목: {paper['title']}
초록: {paper['abstract'][:500]}...  # 초록은 500자로 제한
DOI: {paper.get('doi', 'N/A')}
""")
        
        prompt = f"""
다음 논문들이 실제 연구 논문이고 ADMET 관련 내용이 포함되어 있는지 판단해주세요.

{''.join(papers_info)}

각 논문에 대해 다음 형식으로 간단히 답변해주세요:
논문1: 연구논문(예/아니오), ADMET관련(예/아니오), 신뢰도(0-1)
논문2: 연구논문(예/아니오), ADMET관련(예/아니오), 신뢰도(0-1)
...

예시:
논문1: 연구논문(예), ADMET관련(예), 신뢰도(0.9)
논문2: 연구논문(아니오), ADMET관련(아니오), 신뢰도(0.2)
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 결과 파싱
        results = []
        lines = result_text.split('\n')
        for i, line in enumerate(lines):
            if f"논문{i+1}:" in line:
                is_research = "연구논문(예)" in line
                has_admet = "ADMET관련(예)" in line
                # 신뢰도 추출
                confidence_match = re.search(r'신뢰도\(([0-9.]+)\)', line)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                
                results.append({
                    "is_research_paper": is_research,
                    "has_admet_content": has_admet,
                    "is_organoid_related": True,  # 이미 organoid로 검색했으므로
                    "confidence_score": confidence,
                    "reasoning": f"배치 검증 결과: {line.strip()}"
                })
            else:
                # 파싱 실패 시 기본값
                results.append({
                    "is_research_paper": False,
                    "has_admet_content": False,
                    "is_organoid_related": True,
                    "confidence_score": 0.0,
                    "reasoning": "파싱 실패"
                })
        
        return results
        
    except Exception as e:
        print(f"[GPT-BATCH-VALIDATION] Error: {e}")
        # 에러 시 모든 논문을 거부
        return [{
            "is_research_paper": False,
            "has_admet_content": False,
            "is_organoid_related": True,
            "confidence_score": 0.0,
            "reasoning": f"GPT 배치 검증 실패: {str(e)}"
        } for _ in titles_abstracts]

# Entrez 초기화
Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

def build_organoid_query() -> str:
    """Organoid만으로 검색 (PMC 자유원문 포함)"""
    return f"({BASE_ORGANOID}) AND {PMC_SUBSET}"

def esearch_pmids(query: str, retmax=500, retstart=0) -> Tuple[List[str], int]:
    """PubMed ESearch: 페이징 지원 (날짜 필터 없음)"""
    sleep()
    h = Entrez.esearch(db="pubmed",
                       term=query,
                       retmax=retmax,
                       retstart=retstart)
    rec = Entrez.read(h)
    h.close()
    ids = rec.get("IdList", [])
    total = int(rec.get("Count", "0"))
    return ids, total

def esummary(pmids: List[str]) -> Dict[str, dict]:
    """PubMed ESummary로 PMID들에 대한 정보를 Batch로 가져옴"""
    out = {}
    if not pmids:
        return out
    B = 200
    for i in range(0, len(pmids), B):
        chunk = pmids[i:i+B]
        sleep()
        h = Entrez.esummary(db="pubmed", id=",".join(chunk), retmode="json")
        summ = json.load(h)
        h.close()
        result = summ.get("result", {})
        for pid in chunk:
            if pid in result:
                out[pid] = result[pid]
    return out

def elink_pubmed_to_pmc(pmids: List[str]) -> Dict[str, Optional[str]]:
    """PubMed의 PMID를 PMC의 PMCID로 맵핑"""
    pmc = {pid: None for pid in pmids}
    if not pmids:
        return pmc
    B = 200
    for i in range(0, len(pmids), B):
        ids = pmids[i:i+B]
        sleep()
        h = Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(ids))
        rec = Entrez.read(h)
        h.close()
        for linkset in rec:
            pid = linkset["IdList"][0]
            links = linkset.get("LinkSetDb", [])
            found = None
            for ldb in links:
                if ldb.get("DbTo") == "pmc":
                    for link in ldb.get("Link", []):
                        found = f"PMC{link['Id']}"
                        break
            pmc[pid] = found
    return pmc

def pmc_try_main_pdf(pmcid: str) -> Optional[bytes]:
    """PMC에서 PDF 다운로드"""
    def _is_pdf(resp) -> bool:
        ctype = resp.headers.get("Content-Type", "").lower()
        return ctype.startswith("application/pdf") or resp.content[:4] == b"%PDF"

    # 표준 패턴들 시도
    bases = [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1",
    ]
    
    for url in bases:
        try:
            r = http_get(url)
            if 200 <= r.status_code < 300 and _is_pdf(r):
                return r.content
        except Exception:
            pass

    # OA 서비스 시도
    try:
        r = http_get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
                     params={"id": pmcid})
        if not (200 <= r.status_code < 300):
            return None
        text = r.text
        pdf_urls = re.findall(r'href="([^"]+\.pdf[^"]*)"', text, flags=re.IGNORECASE)
        tried = set()
        base = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        for href in pdf_urls:
            url = href if href.startswith("http") else urljoin(base, href)
            if url in tried: 
                continue
            tried.add(url)
            resp = http_get(url)
            ctype = resp.headers.get("Content-Type","").lower()
            if ctype.startswith("application/pdf") or resp.content[:4] == b"%PDF":
                return resp.content
    except Exception:
        pass

    return None

def harvest_efficient(target_pdfs=1000, query: Optional[str]=None, page_size=200, max_pages=100):
    """
    효율적인 논문 수집 (키워드 사전 필터링 + 배치 GPT 검증)
    """
    q = query or build_organoid_query()
    print("[ORGANOID QUERY]\n", q, "\n")

    saved = 0
    retstart = 0
    page = 0
    validation_stats = {
        "total_checked": 0,
        "keyword_filtered": 0,
        "gpt_validated": 0,
        "research_papers": 0,
        "admet_related": 0,
        "organoid_related": 0
    }

    while saved < target_pdfs and page < max_pages:
        pmids, total = esearch_pmids(q, retmax=page_size, retstart=retstart)
        if not pmids:
            print(f"[INFO] 더 이상 PMID가 없습니다. total={total}, retstart={retstart}")
            break

        print(f"[ESearch] page={page+1} retstart={retstart} -> got {len(pmids)} / total={total}")

        # PMID -> PMCID 매핑 + 메타
        pmc_map = elink_pubmed_to_pmc(pmids)
        meta = esummary(pmids)

        # 배치 처리를 위한 후보 논문들 수집
        batch_candidates = []
        batch_indices = []
        
        for i, pid in enumerate(pmids):
            if saved >= target_pdfs:
                break

            pmcid = pmc_map.get(pid)
            if not pmcid:
                continue

            # 메타데이터에서 제목과 초록 추출
            if pid not in meta:
                continue
                
            paper_meta = meta[pid]
            title = paper_meta.get("Title", "")
            abstract = paper_meta.get("Abstract", "")
            doi = paper_meta.get("DOI", "")

            if not title or not abstract:
                continue

            # 1단계: 키워드 사전 필터링
            combined_text = f"{title} {abstract}"
            if has_admet_keywords(combined_text):
                validation_stats["keyword_filtered"] += 1
                batch_candidates.append({
                    "pid": pid,
                    "pmcid": pmcid,
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "meta": paper_meta
                })
                batch_indices.append(i)

        # 2단계: 배치 GPT 검증 (5개씩)
        batch_size = 5
        for batch_start in range(0, len(batch_candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_candidates))
            current_batch = batch_candidates[batch_start:batch_end]
            
            if not current_batch:
                continue
                
            print(f"[GPT-BATCH-VALIDATION] {len(current_batch)}개 논문 배치 검증 중...")
            
            # 배치 검증용 데이터 준비
            batch_data = []
            for candidate in current_batch:
                batch_data.append({
                    "title": candidate["title"],
                    "abstract": candidate["abstract"],
                    "doi": candidate["doi"]
                })
            
            # GPT 배치 검증
            validation_results = validate_papers_batch(batch_data)
            validation_stats["total_checked"] += len(current_batch)
            
            # 검증 결과 처리
            for i, (candidate, validation_result) in enumerate(zip(current_batch, validation_results)):
                if validation_result["is_research_paper"]:
                    validation_stats["research_papers"] += 1
                if validation_result["has_admet_content"]:
                    validation_stats["admet_related"] += 1
                if validation_result["is_organoid_related"]:
                    validation_stats["organoid_related"] += 1

                # 검증 통과 조건
                if (validation_result["is_research_paper"] and 
                    validation_result["has_admet_content"] and 
                    validation_result["is_organoid_related"] and
                    validation_result["confidence_score"] >= 0.7):
                    
                    print(f"[GPT-OK] {candidate['pmcid']}: 검증 통과 (신뢰도: {validation_result['confidence_score']:.2f})")
                    validation_stats["gpt_validated"] += 1
                    
                    # PDF 다운로드
                    try:
                        pdf_bytes = pmc_try_main_pdf(candidate['pmcid'])
                    except Exception as e:
                        print(f"[WARN] PDF fetch error for {candidate['pmcid']}: {e}")
                        pdf_bytes = None

                    if not pdf_bytes:
                        print(f"[MISS] no PDF for {candidate['pmcid']}")
                        continue

                    # 폴더 생성 및 저장
                    folder = OUT_DIR / f"{candidate['pid']}_{candidate['pmcid']}"
                    folder.mkdir(parents=True, exist_ok=True)
                    save_bin(pdf_bytes, folder / "article.pdf")
                    
                    # 메타데이터 저장
                    candidate['meta']["gpt_validation"] = validation_result
                    save_json(candidate['meta'], folder / "pubmed_summary.json")
                    
                    print(f"[OK] saved PDF for {candidate['pmcid']}")
                    saved += 1

                else:
                    print(f"[GPT-REJECT] {candidate['pmcid']}: 검증 실패 - {validation_result['reasoning']}")

        # 다음 페이지로
        retstart += len(pmids)
        page += 1
        
        # 진행 상황 출력
        print(f"\n[PROGRESS] 저장된 논문: {saved}/{target_pdfs}")
        print(f"[STATS] 검증 통계: {validation_stats}")
        print("-" * 50)

    # 최종 통계 저장
    final_stats = {
        "target_pdfs": target_pdfs,
        "saved_pdfs": saved,
        "validation_stats": validation_stats,
        "query": q,
        "timestamp": datetime.now().isoformat()
    }
    save_json(final_stats, LOG_DIR / f"efficient_collection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    print(f"\n[Done] 효율적인 GPT 검증 완료!")
    print(f"저장된 논문: {saved}/{target_pdfs}")
    print(f"검증 통계: {validation_stats}")
    print(f"결과 저장 위치: {OUT_DIR.resolve()}")

def main():
    import argparse
    global RATE_SLEEP, OUT_DIR

    parser = argparse.ArgumentParser(
        description="효율적인 GPT 기반 ADMET 논문 수집기"
    )
    parser.add_argument("--target_pdfs", type=int, default=1000,
                        help="수집할 논문 목표 개수")
    parser.add_argument("--query", type=str, default=None,
                        help="사용자 정의 쿼리")
    parser.add_argument("--email", type=str, default=EMAIL,
                        help="NCBI 이메일")
    parser.add_argument("--api_key", type=str, default=API_KEY,
                        help="NCBI API 키")
    parser.add_argument("--sleep", type=float, default=RATE_SLEEP,
                       help="레이트리밋(초)")
    parser.add_argument("--output_dir", type=str, default="./raws_gpt",
                       help="논문 저장 디렉토리")

    args = parser.parse_args()

    # 설정 적용
    if args.email:
        Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
    if args.sleep is not None:
        RATE_SLEEP = max(0.2, float(args.sleep))
    
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Output directory: {OUT_DIR.resolve()}")
    print(f"[INFO] OpenAI API Key: {'설정됨' if OPENAI_API_KEY else '설정되지 않음'}")

    try:
        harvest_efficient(
            target_pdfs=args.target_pdfs,
            query=args.query
        )
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 작업을 취소했습니다.")
    except Exception as e:
        print(f"[오류] 수집 중 예외 발생: {e}")

if __name__ == "__main__":
    main()


