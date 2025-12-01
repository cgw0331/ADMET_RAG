#!/usr/bin/env python3
"""
PMC 직접 검색으로 논문 수집
"""

import os
import re
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
import openai
from dotenv import load_dotenv

from Bio import Entrez

load_dotenv()

EMAIL = os.getenv("NCBI_EMAIL", "chlrjs3@kangwon.ac.kr")
API_KEY = os.getenv("NCBI_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")

openai.api_key = OPENAI_API_KEY

Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

OUT_DIR = Path("./raws_gpt")
LOG_DIR = Path("./logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

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

def has_admet_keywords(text: str) -> bool:
    """ADMET 키워드 확인"""
    text_lower = text.lower()
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
    keyword_count = sum(1 for keyword in admet_keywords if keyword in text_lower)
    return keyword_count >= 2

def validate_with_gpt(title: str, abstract: str) -> Dict[str, any]:
    """GPT로 논문 검증"""
    try:
        prompt = f"""
다음 논문이 실제 연구 논문이고 ADMET 관련 내용이 포함되어 있는지 간단히 판단해주세요.

제목: {title}
초록: {abstract[:500]}...

다음 형식으로 답변해주세요:
연구논문(예/아니오), ADMET관련(예/아니오), 신뢰도(0-1)
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        
        is_research = "연구논문(예)" in result_text
        has_admet = "ADMET관련(예)" in result_text
        confidence_match = re.search(r'신뢰도\(([0-9.]+)\)', result_text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        return {
            "is_research_paper": is_research,
            "has_admet_content": has_admet,
            "is_organoid_related": True,
            "confidence_score": confidence,
            "reasoning": result_text
        }
        
    except Exception as e:
        return {
            "is_research_paper": False,
            "has_admet_content": False,
            "is_organoid_related": True,
            "confidence_score": 0.0,
            "reasoning": f"GPT 검증 실패: {str(e)}"
        }

def search_pmc_direct():
    """PMC 데이터베이스에서 직접 검색"""
    
    # PMC에서 organoid 검색
    organoid_query = 'organoid[tiab] OR organoids[tiab] OR organoid*[tiab] OR Organoids[mh]'
    
    print(f"PMC 검색 쿼리: {organoid_query}")
    
    # PMC에서 직접 검색
    h = Entrez.esearch(db="pmc", term=organoid_query, retmax=1000)
    rec = Entrez.read(h)
    h.close()
    
    pmcids = rec.get("IdList", [])
    total = int(rec.get("Count", "0"))
    
    print(f"PMC에서 찾은 논문 수: {total}")
    print(f"처리할 PMCID: {len(pmcids)}")
    
    return pmcids

def get_pmc_metadata(pmcid: str) -> Optional[Dict]:
    """PMC에서 메타데이터 가져오기"""
    try:
        h = Entrez.esummary(db="pmc", id=pmcid, retmode="json")
        summ = json.load(h)
        h.close()
        
        result = summ.get("result", {}).get(pmcid, {})
        if result:
            return {
                "pmcid": pmcid,
                "title": result.get("Title", ""),
                "abstract": result.get("Abstract", ""),
                "authors": result.get("AuthorList", []),
                "journal": result.get("Source", ""),
                "pub_date": result.get("PubDate", ""),
                "doi": result.get("DOI", "")
            }
    except Exception as e:
        print(f"[WARN] 메타데이터 가져오기 실패 {pmcid}: {e}")
    
    return None

def download_pmc_pdf(pmcid: str) -> Optional[bytes]:
    """PMC에서 PDF 다운로드"""
    try:
        # 표준 PDF URL들 시도
        pdf_urls = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1",
        ]
        
        for url in pdf_urls:
            try:
                r = http_get(url)
                if r.content.startswith(b'%PDF'):
                    return r.content
            except Exception:
                continue
                
    except Exception as e:
        print(f"[WARN] PDF 다운로드 실패 {pmcid}: {e}")
    
    return None

def collect_from_pmc(target_pdfs=100):
    """PMC에서 직접 논문 수집"""
    
    pmcids = search_pmc_direct()
    
    saved = 0
    stats = {
        "total_processed": 0,
        "metadata_found": 0,
        "keyword_filtered": 0,
        "gpt_validated": 0,
        "pdf_downloaded": 0
    }
    
    for i, pmcid in enumerate(pmcids):
        if saved >= target_pdfs:
            break
            
        print(f"[{i+1}/{len(pmcids)}] 처리 중: {pmcid}")
        stats["total_processed"] += 1
        
        # 메타데이터 가져오기
        metadata = get_pmc_metadata(pmcid)
        if not metadata:
            print(f"  메타데이터 없음")
            continue
            
        stats["metadata_found"] += 1
        title = metadata["title"]
        abstract = metadata["abstract"]
        
        if not title or not abstract:
            print(f"  제목 또는 초록 없음")
            continue
        
        # 키워드 필터링
        combined_text = f"{title} {abstract}"
        if not has_admet_keywords(combined_text):
            print(f"  ADMET 키워드 없음")
            continue
            
        stats["keyword_filtered"] += 1
        
        # GPT 검증
        print(f"  GPT 검증 중...")
        validation = validate_with_gpt(title, abstract)
        
        if (validation["is_research_paper"] and 
            validation["has_admet_content"] and 
            validation["confidence_score"] >= 0.7):
            
            print(f"  ✓ 검증 통과 (신뢰도: {validation['confidence_score']:.2f})")
            stats["gpt_validated"] += 1
            
            # PDF 다운로드
            pdf_bytes = download_pmc_pdf(pmcid)
            if pdf_bytes:
                # 저장
                folder = OUT_DIR / pmcid
                folder.mkdir(parents=True, exist_ok=True)
                save_bin(pdf_bytes, folder / "article.pdf")
                
                # 메타데이터 저장
                metadata["gpt_validation"] = validation
                save_json(metadata, folder / "metadata.json")
                
                print(f"  ✓ 저장 완료")
                saved += 1
                stats["pdf_downloaded"] += 1
            else:
                print(f"  PDF 다운로드 실패")
        else:
            print(f"  ✗ 검증 실패: {validation['reasoning']}")
        
        print(f"  진행: {saved}/{target_pdfs}")
        print("-" * 50)
    
    print(f"\n[Done] PMC 수집 완료!")
    print(f"저장된 논문: {saved}/{target_pdfs}")
    print(f"통계: {stats}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PMC 직접 논문 수집")
    parser.add_argument("--target_pdfs", type=int, default=100, help="수집할 논문 수")
    parser.add_argument("--output_dir", type=str, default="./raws_gpt", help="저장 디렉토리")
    
    args = parser.parse_args()
    
    global OUT_DIR
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        collect_from_pmc(target_pdfs=args.target_pdfs)
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 작업을 취소했습니다.")
    except Exception as e:
        print(f"[오류] 수집 중 예외 발생: {e}")

if __name__ == "__main__":
    main()


