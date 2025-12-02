#!/usr/bin/env python3
"""
GPT 기반 논문 수집 시스템
- ADMET 19종 지표 키워드로 정확한 논문 검색
- GPT로 논문 품질 검증 및 필터링
- PMC에서 PDF 및 보충자료 다운로드
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

# ADMET 19종 지표 키워드 (확장된 버전)
ADMET_KEYWORDS = {
    # Absorption (흡수)
    "caco2_permeability": '(Caco-2[tiab] OR "Caco-2 permeability"[tiab] OR Papp[tiab] OR "apparent permeability"[tiab])',
    "mdck_permeability": '(MDCK[tiab] OR "MDCK permeability"[tiab] OR "Madin-Darby canine kidney"[tiab])',
    "pampa": '(PAMPA[tiab] OR "parallel artificial membrane permeability"[tiab])',
    "lipinski_rule": '("Lipinski rule"[tiab] OR "rule of five"[tiab] OR "Lipinski\'s rule"[tiab])',
    "logd_logs_pka": '(logD[tiab] OR logS[tiab] OR pKa[tiab] OR "partition coefficient"[tiab] OR "solubility"[tiab])',
    
    # Distribution (분포)
    "ppb": '("plasma protein binding"[tiab] OR PPB[tiab] OR "protein binding"[tiab] OR "serum protein binding"[tiab])',
    "bbb": '("blood-brain barrier"[tiab] OR BBB[tiab] OR "brain penetration"[tiab] OR "BBB permeability"[tiab])',
    "vd": '("volume of distribution"[tiab] OR Vd[tiab] OR "distribution volume"[tiab])',
    
    # Metabolism (대사)
    "cyp1a2": '(CYP1A2[tiab] OR "CYP1A2 inhibition"[tiab] OR "CYP1A2 inhibitor"[tiab])',
    "cyp2c9": '(CYP2C9[tiab] OR "CYP2C9 inhibition"[tiab] OR "CYP2C9 inhibitor"[tiab])',
    "cyp2c19": '(CYP2C19[tiab] OR "CYP2C19 inhibition"[tiab] OR "CYP2C19 inhibitor"[tiab])',
    "cyp2d6": '(CYP2D6[tiab] OR "CYP2D6 inhibition"[tiab] OR "CYP2D6 inhibitor"[tiab])',
    "cyp3a4": '(CYP3A4[tiab] OR "CYP3A4 inhibition"[tiab] OR "CYP3A4 inhibitor"[tiab])',
    "cyp_inhibition": '("CYP inhibition"[tiab] OR "cytochrome P450 inhibition"[tiab] OR "P450 inhibition"[tiab])',
    
    # Excretion (배설)
    "clearance": '(clearance[tiab] OR CL[tiab] OR "renal clearance"[tiab] OR "hepatic clearance"[tiab])',
    "half_life": '("half-life"[tiab] OR "t1/2"[tiab] OR "elimination half life"[tiab] OR "terminal half-life"[tiab])',
    
    # Toxicity (독성)
    "herg": '(hERG[tiab] OR "hERG blockers"[tiab] OR "QT prolongation"[tiab] OR "cardiac toxicity"[tiab])',
    "dili": '("Drug-Induced Liver Injury"[mh] OR DILI[tiab] OR "hepatotoxicity"[tiab] OR "liver injury"[tiab])',
    "ames_test": '("Ames test"[tiab] OR "bacterial mutagenicity"[tiab] OR "mutagenicity"[tiab])',
    "carcinogenicity": '(carcinogenic*[tiab] OR genotoxic*[tiab] OR "cancer risk"[tiab])',
}

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

def check_supplements_with_gpt(title: str, abstract: str, doi: str = None) -> Dict[str, any]:
    """
    GPT를 사용하여 논문에 보충자료가 있는지 판단
    """
    try:
        prompt = f"""
다음 논문 정보를 보고 이 논문에 보충자료(Supplementary materials)가 있는지 판단해주세요.

제목: {title}
초록: {abstract}
DOI: {doi or "N/A"}

다음 기준으로 판단해주세요:
1. 논문에 보충자료(Supplementary materials, Supplementary data, Additional files) 언급이 있는가?
2. 보충자료의 종류는 무엇인가? (Excel, PDF, 이미지, 데이터 파일 등)
3. 보충자료의 내용은 무엇인가? (ADMET 데이터, 실험 결과, 표, 그래프 등)

다음 형식으로 JSON 응답해주세요:
{{
    "has_supplements": true/false,
    "supplement_types": ["Excel", "PDF", "이미지"] 또는 [],
    "supplement_content": "보충자료 내용 설명",
    "confidence_score": 0.0-1.0,
    "reasoning": "판단 근거 설명"
}}

주의사항:
- 명시적으로 보충자료 언급이 있어야 함
- "Additional data", "Supplementary information", "Supporting materials" 등도 포함
- 단순히 "자세한 방법은 보충자료 참조" 같은 언급은 제외
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트에서 정보 추출
            has_supplements = "true" in result_text.lower() and "has_supplements" in result_text.lower()
            
            return {
                "has_supplements": has_supplements,
                "supplement_types": [],
                "supplement_content": "GPT 응답 파싱 실패",
                "confidence_score": 0.5,
                "reasoning": "GPT 응답 파싱 실패, 기본값 사용"
            }
            
    except Exception as e:
        print(f"[GPT-SUPPLEMENT-CHECK] Error: {e}")
        return {
            "has_supplements": False,
            "supplement_types": [],
            "supplement_content": "",
            "confidence_score": 0.0,
            "reasoning": f"GPT 보충자료 검사 실패: {str(e)}"
        }
    """
    GPT를 사용하여 논문이 실제 연구 논문인지 검증
    """
    try:
        prompt = f"""
다음은 PubMed에서 검색된 논문 정보입니다. 이 논문이 실제 연구 논문인지 판단해주세요.

제목: {title}
초록: {abstract}
DOI: {doi or "N/A"}

다음 기준으로 판단해주세요:
1. 실제 연구 논문인가? (예: 실험, 임상시험, 데이터 분석 등)
2. ADMET 관련 내용이 포함되어 있는가?
3. Organoid와 관련된 연구인가?

다음 형식으로 JSON 응답해주세요:
{{
    "is_research_paper": true/false,
    "has_admet_content": true/false,
    "is_organoid_related": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "판단 근거 설명"
}}

주의사항:
- 리뷰 논문, 메타분석, 편집자 논평은 제외
- 가이드라인, 프로토콜, 매뉴얼은 제외
- 튜토리얼, 교재, 책은 제외
- 실제 실험 데이터나 분석 결과가 있는 논문만 포함
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트에서 정보 추출
            is_research = "true" in result_text.lower() and "is_research_paper" in result_text.lower()
            has_admet = "true" in result_text.lower() and "has_admet_content" in result_text.lower()
            is_organoid = "true" in result_text.lower() and "is_organoid_related" in result_text.lower()
            
            return {
                "is_research_paper": is_research,
                "has_admet_content": has_admet,
                "is_organoid_related": is_organoid,
                "confidence_score": 0.5,
                "reasoning": "GPT 응답 파싱 실패, 기본값 사용"
            }
            
    except Exception as e:
        print(f"[GPT-VALIDATION] Error: {e}")
        return {
            "is_research_paper": False,
            "has_admet_content": False,
            "is_organoid_related": False,
            "confidence_score": 0.0,
            "reasoning": f"GPT 검증 실패: {str(e)}"
        }

# Entrez 초기화
Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

def build_organoid_query() -> str:
    """Organoid만으로 검색 (PMC 자유원문 포함)"""
    return f"({BASE_ORGANOID}) AND {PMC_SUBSET}"

def build_admet_query() -> str:
    """ADMET 19종 지표 키워드로 쿼리 구성 (PMC 필터 완화)"""
    # 모든 ADMET 키워드를 OR로 연결
    all_admet = "(" + " OR ".join(ADMET_KEYWORDS.values()) + ")"
    
    # PMC 필터 완화 - PMC 자유원문이 있으면 좋지만 필수는 아님
    return f"({BASE_ORGANOID}) AND {all_admet}"

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

def pmc_fetch_jats(pmcid: str) -> Optional[str]:
    """PMC에서 JATS XML 가져오기"""
    params = {"db": "pmc", "id": pmcid, "rettype": "xml"}
    try:
        r = http_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        pass
    return None

def pmc_list_supplements_from_jats(pmcid: str) -> List[Dict[str,str]]:
    """JATS XML에서 보충자료 URL 수집"""
    jats = pmc_fetch_jats(pmcid)
    supps = []
    if not jats:
        return supps

    try:
        xml = etree.fromstring(jats.encode("utf-8"))
    except Exception:
        return supps

    XLINK = "{http://www.w3.org/1999/xlink}"
    base_article = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    base_bin     = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/"

    # media xlink:href
    for media in xml.findall(".//media"):
        href = media.get(XLINK+"href") or media.get("xlink:href")
        if href:
            url = href if href.startswith("http") else urljoin(base_bin, href)
            supps.append({"source":"JATS-media", "url":url, "desc":media.get("mimetype","")})

    # supplementary-material 내부 ext-link / self-uri
    for supp in xml.findall(".//supplementary-material"):
        for s in supp.findall(".//self-uri"):
            href = s.get(XLINK+"href") or s.get("xlink:href")
            if href:
                url = href if href.startswith("http") else urljoin(base_article, href)
                supps.append({"source":"JATS-selfuri", "url":url, "desc":s.get("content-type","")})
        for e in supp.findall(".//ext-link"):
            href = e.get(XLINK+"href") or e.get("xlink:href")
            if href:
                url = href if href.startswith("http") else urljoin(base_article, href)
                supps.append({"source":"JATS-extlink", "url":url, "desc":e.get("ext-link-type","")})

    # dedup
    uniq = {}
    for s in supps:
        uniq[s["url"]] = s
    return list(uniq.values())

def download_to(path: Path, url: str) -> bool:
    """URL에서 파일 다운로드"""
    try:
        r = http_get(url, stream=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if not path.exists() or path.stat().st_size == 0:
            return False
        return True
    except Exception as e:
        print(f"[DL-ERR] {url} -> {e}")
        return False

def harvest_with_gpt_validation(target_pdfs=10, query: Optional[str]=None, page_size=200, max_pages=100):
    """
    GPT 검증을 포함한 논문 수집 (Organoid 우선 검색)
    """
    # Organoid만으로 먼저 검색 (더 많은 논문 확보)
    q = query or build_organoid_query()
    print("[ORGANOID QUERY]\n", q, "\n")

    saved = 0
    retstart = 0
    page = 0
    validation_stats = {
        "total_checked": 0,
        "gpt_validated": 0,
        "research_papers": 0,
        "admet_related": 0,
        "organoid_related": 0,
        "supplements_checked": 0,
        "supplements_found": 0,
        "supplements_downloaded": 0
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

        for pid in pmids:
            if saved >= target_pdfs:
                break

            pmcid = pmc_map.get(pid)
            print(f"PID {pid} → PMCID {pmcid}")

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
                print(f"[SKIP] {pmcid}: 제목 또는 초록 없음")
                continue

            # GPT로 논문 검증 (ADMET 관련성 강화)
            print(f"[GPT-VALIDATION] {pmcid}: 검증 중...")
            validation_result = validate_paper_with_gpt(title, abstract, doi)
            validation_stats["total_checked"] += 1
            
            if validation_result["is_research_paper"]:
                validation_stats["research_papers"] += 1
            if validation_result["has_admet_content"]:
                validation_stats["admet_related"] += 1
            if validation_result["is_organoid_related"]:
                validation_stats["organoid_related"] += 1

            # 검증 통과 조건: 연구 논문이고 ADMET 관련이면서 신뢰도가 높은 경우 (조건 강화)
            if (validation_result["is_research_paper"] and 
                validation_result["has_admet_content"] and 
                validation_result["is_organoid_related"] and
                validation_result["confidence_score"] >= 0.8):  # 신뢰도 기준 상향
                
                print(f"[GPT-OK] {pmcid}: 검증 통과 (신뢰도: {validation_result['confidence_score']:.2f})")
                validation_stats["gpt_validated"] += 1
                
                # PDF 다운로드
                try:
                    pdf_bytes = pmc_try_main_pdf(pmcid)
                except Exception as e:
                    print(f"[WARN] PDF fetch error for {pmcid}: {e}")
                    pdf_bytes = None

                if not pdf_bytes:
                    print(f"[MISS] no PDF for {pmcid}")
                    continue

                # 폴더 생성 및 저장
                folder = OUT_DIR / f"{pid}_{pmcid}"
                folder.mkdir(parents=True, exist_ok=True)
                save_bin(pdf_bytes, folder / "article.pdf")
                
                # 메타데이터 저장
                paper_meta["gpt_validation"] = validation_result
                save_json(paper_meta, folder / "pubmed_summary.json")
                
                print(f"[OK] saved PDF for {pmcid}")
                saved += 1

                # GPT로 보충자료 존재 여부 먼저 확인
                print(f"[GPT-SUPPLEMENT-CHECK] {pmcid}: 보충자료 존재 여부 확인 중...")
                supplement_check = check_supplements_with_gpt(title, abstract, doi)
                validation_stats["supplements_checked"] += 1
                
                # 보충자료가 있다고 판단된 경우에만 다운로드
                if (supplement_check["has_supplements"] and 
                    supplement_check["confidence_score"] >= 0.6):
                    validation_stats["supplements_found"] += 1
                    
                    print(f"[GPT-SUPPLEMENT-OK] {pmcid}: 보충자료 존재 확인됨 - {supplement_check['reasoning']}")
                    
                    try:
                        supps = pmc_list_supplements_from_jats(pmcid)
                        if supps:
                            supp_dir = folder / "supplements"
                            supp_dir.mkdir(parents=True, exist_ok=True)
                            index_rows = []
                            saved_cnt = 0
                            
                            for s in supps[:10]:  # 최대 10개로 제한
                                url = s["url"]
                                raw_name = urlparse(url).path.split("/")[-1] or "supp"
                                raw_name = raw_name.split("?")[0]
                                fname = sanitize_filename(raw_name if raw_name else "supp")
                                out = supp_dir / fname

                                ok = False
                                if url.startswith("http"):
                                    ok = download_to(out, url)

                                index_rows.append({**s, "saved_as": (str(out.name) if ok else "")})
                                if ok:
                                    saved_cnt += 1

                            p = folder / "supp_index.csv"
                            with p.open("w", newline="", encoding="utf-8") as f:
                                w = csv.DictWriter(f, fieldnames=["source","url","desc","saved_as"])
                                w.writeheader()
                                w.writerows(index_rows)

                            print(f"[SUPP] {pmcid}: found={len(supps)} saved={saved_cnt}")
                            if saved_cnt > 0:
                                validation_stats["supplements_downloaded"] += 1
                        else:
                            print(f"[SUPP] {pmcid}: GPT는 보충자료 있다고 했지만 실제로는 없음")
                            
                    except Exception as e:
                        print(f"[WARN] supplement error for {pmcid}: {e}")
                else:
                    print(f"[GPT-SUPPLEMENT-SKIP] {pmcid}: 보충자료 없음으로 판단 - {supplement_check['reasoning']}")
                
                # 보충자료 검사 결과도 메타데이터에 저장
                paper_meta["supplement_check"] = supplement_check

            else:
                print(f"[GPT-REJECT] {pmcid}: 검증 실패 - {validation_result['reasoning']}")

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
    save_json(final_stats, LOG_DIR / f"gpt_collection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    print(f"\n[Done] GPT 검증 완료!")
    print(f"저장된 논문: {saved}/{target_pdfs}")
    print(f"검증 통계: {validation_stats}")
    print(f"결과 저장 위치: {OUT_DIR.resolve()}")

def main():
    import argparse
    global RATE_SLEEP, OUT_DIR

    parser = argparse.ArgumentParser(
        description="GPT 기반 ADMET 논문 수집기"
    )
    parser.add_argument("--target_pdfs", type=int, default=10,
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
        harvest_with_gpt_validation(
            target_pdfs=args.target_pdfs,
            query=args.query
        )
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 작업을 취소했습니다.")
    except Exception as e:
        print(f"[오류] 수집 중 예외 발생: {e}")

if __name__ == "__main__":
    main()
