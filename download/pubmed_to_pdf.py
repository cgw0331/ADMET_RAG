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

from Bio import Entrez
# Entrez.email = "chlrjs3@kangwon.ac.kr"

EMAIL = os.getenv("NCBI_EMAIL", "chlrjs3@kangwon.ac.kr")
API_KEY = os.getenv("NCBI_API_KEY", None)

DATE_FROM = "2010/01/01"
DATE_TO = datetime.now().strftime("%Y/%m/%d")

# 검색 쿼리: organoid + 8지표 키워드 (Title/Abstract 우선, PMC 소유 문헌 우선)
BASE_ORGANOID = '('
BASE_ORGANOID += 'organoid[tiab] OR organoids[tiab] OR organoid*[tiab] '
BASE_ORGANOID += 'OR Organoids[mh] OR organoid-derived[tiab]'
BASE_ORGANOID += ')'


# 19종 ADMET 지표 키워드 (ADMETlab 3.0 기준)
KEYWORD_BLOCKS = {
    # Absorption (흡수)
    "caco2_permeability": '(Caco-2[tiab] OR "Caco-2 permeability"[tiab] OR Papp[tiab] OR "apparent permeability"[tiab])',
    "mdck_permeability": '(MDCK[tiab] OR "MDCK permeability"[tiab] OR "Madin-Darby canine kidney"[tiab])',
    "pampa": '(PAMPA[tiab] OR "parallel artificial membrane permeability"[tiab])',
    "lipinski_rule_of_five": '("Lipinski rule"[tiab] OR "rule of five"[tiab] OR "Lipinski\'s rule"[tiab])',
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
    "cl": '(clearance[tiab] OR CL[tiab] OR "renal clearance"[tiab] OR "hepatic clearance"[tiab])',
    "t_half": '("half-life"[tiab] OR "t1/2"[tiab] OR "elimination half life"[tiab] OR "terminal half-life"[tiab])',
    
    # Toxicity (독성)
    "herg": '(hERG[tiab] OR "hERG blockers"[tiab] OR "QT prolongation"[tiab] OR "cardiac toxicity"[tiab])',
    "dili": '("Drug-Induced Liver Injury"[mh] OR DILI[tiab] OR "hepatotoxicity"[tiab] OR "liver injury"[tiab])',
    "ames_test": '("Ames test"[tiab] OR "bacterial mutagenicity"[tiab] OR "mutagenicity"[tiab])',
    "carcinogenicity": '(carcinogenic*[tiab] OR genotoxic*[tiab] OR "cancer risk"[tiab])',
}

# PMC 자유원문 필터 (PubMed subset)
PMC_SUBSET = '"pubmed pmc"[sb]'

OUT_DIR = Path("./raws_gpt")  # 기본값
LOG_DIR = Path("./logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# NCBI rate limit (보수적)  API 있으면 초당 10회, 없으면 초당 3회 
RATE_SLEEP = 0.34 if API_KEY else 0.5  # 초당 3회~2회 정도

HTTP_TIMEOUT = 30

# =========================
# 유틸
# =========================

def sleep():
    time.sleep(RATE_SLEEP)

def http_get(url, params=None, stream=False):
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT, stream=stream, headers={"User-Agent":"Mozilla/5.0"})
        # 2xx
        if 200 <= r.status_code < 300:
            return r
        r.raise_for_status()
    except Exception as e:
        print(f"[HTTP-ERROR] {url}: {e}")
        raise

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

def is_research_paper_from_jats(pmcid: str) -> bool:
    """
    JATS XML 메타데이터로 논문인지 사전 판단하는 함수
    PDF 다운로드 전에 실행
    """
    try:
        jats = pmc_fetch_jats(pmcid)
        if not jats:
            return True  # JATS가 없으면 일단 시도
        
        # JATS XML 파싱
        xml = etree.fromstring(jats.encode("utf-8"))
        
        # 문서 유형 확인
        article_type = xml.find(".//{http://jats.nlm.nih.gov}article-categories")
        if article_type is not None:
            subj_groups = article_type.findall(".//{http://jats.nlm.nih.gov}subj-group")
            for subj_group in subj_groups:
                subj_group_type = subj_group.get("subj-group-type", "").lower()
                if subj_group_type in ["article-type", "publication-type"]:
                    for subj in subj_group.findall(".//{http://jats.nlm.nih.gov}subject"):
                        subject_text = subj.text.lower() if subj.text else ""
                        # 논문이 아닌 문서 유형들
                        non_research_types = [
                            "editorial", "letter", "comment", "correction", "retraction",
                            "guideline", "protocol", "manual", "handbook", "book",
                            "supplement", "appendix", "figure", "table", "data",
                            "review", "meta-analysis", "case-report", "brief-report"
                        ]
                        if any(non_type in subject_text for non_type in non_research_types):
                            print(f"[JATS-FILTER] {pmcid}: Non-research type detected: {subject_text}")
                            return False
        
        # 제목에서 논문이 아닌 키워드 확인
        title_elem = xml.find(".//{http://jats.nlm.nih.gov}article-title")
        if title_elem is not None and title_elem.text:
            title_lower = title_elem.text.lower()
            non_research_title_keywords = [
                "guideline", "protocol", "manual", "handbook", "instruction",
                "supplement", "appendix", "figure", "table", "data",
                "editorial", "letter", "comment", "correction", "retraction"
            ]
            if any(keyword in title_lower for keyword in non_research_title_keywords):
                print(f"[JATS-FILTER] {pmcid}: Non-research title detected: {title_elem.text}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[WARN] Error checking JATS for {pmcid}: {e}")
        return True  # 에러 시 일단 시도

def is_research_paper(pdf_bytes: bytes) -> bool:
    """
    PDF가 실제 논문인지 판단하는 함수
    """
    try:
        from io import BytesIO
        import pdfplumber
        
        # PDF를 메모리에서 읽기
        pdf_buffer = BytesIO(pdf_bytes)
        
        with pdfplumber.open(pdf_buffer) as pdf:
            # 처음 3페이지의 텍스트 추출
            text = ""
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                if len(text) > 2000:  # 충분한 텍스트가 있으면 중단
                    break
        
        text_lower = text.lower()
        
        # 논문 관련 키워드
        research_keywords = [
            'abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion',
            'doi:', 'published', 'journal', 'volume', 'issue', 'pages', 'pmid',
            'corresponding author', 'affiliation', 'keywords', 'acknowledgments',
            'peer reviewed', 'manuscript', 'article', 'research', 'study'
        ]
        
        # 논문이 아닌 문서 키워드
        non_research_keywords = [
            'laboratory manual', 'handbook', 'textbook', 'edition', 'chapter',
            'table of contents', 'preface', 'appendix', 'glossary', 'index',
            'cold spring harbor', 'wiley', 'elsevier', 'springer',
            'slide', 'presentation', 'conference', 'symposium', 'workshop',
            'powerpoint', 'keynote', 'agenda', 'speaker', 'session',
            'report', 'assessment', 'evaluation', 'analysis', 'study',
            'final report', 'technical report', 'contract', 'submitted to',
            'summary of product characteristics', 'spc', 'package insert',
            'label', 'fda', 'ema', 'guideline', 'protocol',
            'manuscript body formatting guidelines', 'formatting guidelines',
            'guidelines for', 'instruction', 'instructions', 'how to',
            'sample', 'template', 'example', 'lorem ipsum', 'dummy text'
        ]
        
        # 논문 키워드 점수
        research_score = sum(1 for keyword in research_keywords if keyword in text_lower)
        
        # 논문이 아닌 키워드 점수
        non_research_score = sum(1 for keyword in non_research_keywords if keyword in text_lower)
        
        # 판단 기준 (더 엄격하게)
        # 1. 비논문 키워드가 3개 이상이면 무조건 제외
        if non_research_score >= 3:
            return False
        
        # 2. 비논문 키워드가 2개 이상이고 논문 키워드가 비논문 키워드보다 적으면 제외
        if non_research_score >= 2 and research_score <= non_research_score:
            return False
        
        # 3. 논문 키워드가 5개 이상이면 포함
        if research_score >= 5:
            return True
        
        # 4. 논문 키워드가 3개 이상이고 비논문 키워드가 1개 이하면 포함
        if research_score >= 3 and non_research_score <= 1:
            return True
        
        # 5. 애매한 경우 텍스트 길이와 구조로 판단
        if len(text) > 2000 and ('abstract' in text_lower and 'introduction' in text_lower and 'lorem ipsum' not in text_lower):
            return True
        
        return False
            
    except Exception as e:
        print(f"[WARN] Error checking if PDF is research paper: {e}")
        return False  # 에러 시 논문이 아닌 것으로 간주


# =========================
# Entrez 초기화
# =========================
Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

# =========================
# PubMed 래퍼  
# =========================


# 지표 들어있는 키워드 묶음 (Similarity 기반으로 검색을 바꿔야 될 수도 있음. )
def build_query() -> str:
    metrics_or = "(" + " OR ".join(KEYWORD_BLOCKS.values()) + ")"
    date_filter = f'("{DATE_FROM}"[dp] : "{DATE_TO}"[dp])'
    return f"({BASE_ORGANOID}) AND {metrics_or} AND {date_filter} AND {PMC_SUBSET}"

def build_pmc_query() -> str:
    # PMC는 pmc subset 필터가 필요 없음. (이미 PMC DB이므로)
    metrics_or = "(" + " OR ".join(KEYWORD_BLOCKS.values()) + ")"
    date_filter = f'("{DATE_FROM}"[dp] : "{DATE_TO}"[dp])'
    # PMC에서도 [tiab], MeSH, 날짜 필터 사용 가능
    return f"({BASE_ORGANOID}) AND {metrics_or} AND {date_filter}"

def esearch_pmcids(query: str, retmax=200, retstart=0) -> Tuple[List[str], int]:
    """
    PMC DB에서 PMCID 리스트와 total Count를 직접 가져옴
    예: ['PMC1234567', 'PMC9876543', ...]
    """
    sleep()
    h = Entrez.esearch(db="pmc",
                       term=query,
                       datetype="pdat",
                       mindate=DATE_FROM,
                       maxdate=DATE_TO,
                       retmax=retmax,
                       retstart=retstart)
    rec = Entrez.read(h); h.close()
    ids = rec.get("IdList", [])
    total = int(rec.get("Count", "0"))
    # PMC 검색은 PMCID 문자열을 바로 준다
    pmcids = [i if i.startswith("PMC") else f"PMC{i}" for i in ids]
    return pmcids, total

def esearch_pmids(query: str, retmax=500, retstart=0) -> Tuple[List[str], int]:
    """
    PubMed ESearch: 페이징 지원
    return: (PMID 리스트, 전체 카운트 Count)
    """
    sleep()
    h = Entrez.esearch(db="pubmed",
                       term=query,
                       datetype="pdat",
                       mindate=DATE_FROM,
                       maxdate=DATE_TO,
                       retmax=retmax,
                       retstart=retstart)
    rec = Entrez.read(h)
    h.close()
    ids = rec.get("IdList", [])
    total = int(rec.get("Count", "0"))
    return ids, total


# PubMed ESummary로 PMID들에 대한 정보를 B개 만큼 Batch로 가져옴 
def esummary(pmids: List[str]) -> Dict[str, dict]:
    out = {}
    if not pmids:
        return out
    B = 200 # Batch 200개 
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


# PubMed의 PMID를 PMC의 PMCID로 맵핑함. 
# PMC에 원문이 없는 논문은 PMCID가 없어서 None으로 맵핑됨. 
def elink_pubmed_to_pmc(pmids: List[str]) -> Dict[str, Optional[str]]:
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


# =========================
# PMC PDF / JATS / Supplement
# =========================

# PMC 원문 PDF 가져오기 
def pmc_pdf_via_oa(pmcid: str) -> Optional[bytes]:
    try:
        r = http_get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
                     params={"id": pmcid})
        if not (200 <= r.status_code < 300):
            return None
        text = r.text
        # .pdf / .PDF 모두
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

def is_valid_research_pdf(pdf_bytes: bytes) -> bool:
    """PDF가 실제 논문인지 검증"""
    try:
        if not pdf_bytes.startswith(b'%PDF'):
            return False
        
        # 처음 3000바이트 샘플링
        sample = pdf_bytes[:3000].decode('utf-8', errors='ignore').lower()
        
        # 가이드라인/템플릿 제외
        invalid_keywords = [
            'guideline', 'template', 'sample', 'lorem ipsum', 
            'formatting', 'instruction', 'manual', 'dummy',
            'manuscript body formatting', 'guidelines for'
        ]
        
        if any(keyword in sample for keyword in invalid_keywords):
            print(f"[PDF-VALIDATION] Invalid content detected: {[kw for kw in invalid_keywords if kw in sample]}")
            return False
        
        # 논문 구조 확인
        research_keywords = ['abstract', 'introduction', 'methods', 'results', 'discussion']
        if any(keyword in sample for keyword in research_keywords):
            print(f"[PDF-VALIDATION] Valid research article detected")
            return True
        
        print(f"[PDF-VALIDATION] No clear research structure found")
        return False
        
    except Exception as e:
        print(f"[PDF-VALIDATION] Error: {e}")
        return False

def get_pmc_direct_pdf(pmcid: str) -> Optional[bytes]:
    """PMC 페이지에서 직접 PDF 다운로드"""
    try:
        from bs4 import BeautifulSoup
        
        # PMC 아티클 페이지 접근
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        response = http_get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # PDF 다운로드 링크 찾기
        pdf_urls = []
        
        # 1. "PDF" 텍스트가 있는 링크
        for link in soup.find_all('a', href=True):
            if 'PDF' in link.get_text() and 'pdf' in link['href'].lower():
                href = link['href']
                if href.startswith('http'):
                    pdf_urls.append(href)
                else:
                    pdf_urls.append(urljoin(url, href))
        
        # 2. 메타데이터에서 PDF URL
        meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
        if meta_pdf:
            pdf_urls.append(meta_pdf.get('content'))
        
        # 3. PMC 표준 PDF URL 패턴들
        pdf_urls.extend([
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf"
        ])
        
        # PDF URL들 시도
        for pdf_url in pdf_urls:
            try:
                print(f"[PMC-DIRECT] Trying: {pdf_url}")
                pdf_response = http_get(pdf_url)
                
                if pdf_response.content.startswith(b'%PDF'):
                    if is_valid_research_pdf(pdf_response.content):
                        print(f"[PMC-DIRECT] ✓ Valid research article found")
                        return pdf_response.content
                    else:
                        print(f"[PMC-DIRECT] Skipping invalid content")
                        continue
                        
            except Exception as e:
                print(f"[PMC-DIRECT] Failed {pdf_url}: {e}")
                continue
                
    except Exception as e:
        print(f"[WARN] PMC direct download failed for {pmcid}: {e}")
    
    return None

def europe_pmc_service(pmcid: str) -> Optional[bytes]:
    """Europe PMC 서비스 사용"""
    try:
        url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        response = http_get(url)
        if response.content.startswith(b'%PDF'):
            if is_valid_research_pdf(response.content):
                return response.content
    except Exception as e:
        print(f"[WARN] Europe PMC failed for {pmcid}: {e}")
    return None

def pmc_try_main_pdf(pmcid: str) -> Optional[bytes]:
    """개선된 PDF 다운로드 - 실제 논문만"""
    def _is_pdf(resp) -> bool:
        ctype = resp.headers.get("Content-Type", "").lower()
        return ctype.startswith("application/pdf") or resp.content[:4] == b"%PDF"

    # 1) PMC 직접 다운로드 (우선순위)
    pdf = get_pmc_direct_pdf(pmcid)
    if pdf:
        return pdf

    # 2) Europe PMC 서비스
    pdf = europe_pmc_service(pmcid)
    if pdf:
        return pdf

    # 3) 기존 방법들 (검증 추가)
    # OA
    pdf = pmc_pdf_via_oa(pmcid)
    if pdf and is_valid_research_pdf(pdf):
        return pdf

    # JATS
    jats = pmc_fetch_jats(pmcid)
    if jats:
        try:
            xml = etree.fromstring(jats.encode("utf-8"))
        except Exception:
            xml = None

        def _xhref(elem):
            for k, v in elem.attrib.items():
                if k.endswith('}href') or k.endswith(':href') or k == 'xlink:href' or k == 'href':
                    return v
            return None

        if xml is not None:
            pdf_hrefs = []
            for node in xml.xpath("//*[local-name()='self-uri' and @content-type='pdf']"):
                href = _xhref(node);  pdf_hrefs.append(href) if href else None
            for node in xml.xpath("//*[local-name()='ext-link' and @ext-link-type='pdf']"):
                href = _xhref(node);  pdf_hrefs.append(href) if href else None

            base_article = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
            base_bin     = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/"
            tried = set()
            for href in pdf_hrefs:
                candidates = [href] if href.startswith("http") else [urljoin(base_article, href), urljoin(base_bin, href)]
                for url in candidates:
                    if url in tried: 
                        continue
                    tried.add(url)
                    try:
                        r = http_get(url)
                        if 200 <= r.status_code < 300 and _is_pdf(r) and is_valid_research_pdf(r.content):
                            return r.content
                    except Exception:
                        continue

    # 4) 표준 패턴 (검증 추가)
    bases = [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1",
    ]
    for url in bases:
        try:
            r = http_get(url)
            if 200 <= r.status_code < 300 and _is_pdf(r) and is_valid_research_pdf(r.content):
                return r.content
        except Exception:
            pass

    # 5) 랜딩 HTML 폴백 (검증 추가)
    try:
        base = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        landing = http_get(base)
        candidates = re.findall(r'href="([^"]+\.pdf[^"]*)"', landing.text, flags=re.IGNORECASE)
        tried = set()
        for href in candidates:
            url = href if href.startswith("http") else urljoin(base, href)
            if url in tried:
                continue
            tried.add(url)
            try:
                r = http_get(url)
                if 200 <= r.status_code < 300 and _is_pdf(r) and is_valid_research_pdf(r.content):
                    return r.content
            except Exception:
                continue
    except Exception:
        pass

    return None

#PMC에서 해당 논문의 JATS(XML 본문 메타/구조 문서)를 가져옴.
# PDF 링크, Supplementary, Media 링크 등을 Parsing 할 때 사용함. 
def pmc_fetch_jats(pmcid: str) -> Optional[str]:
    params = {"db": "pmc", "id": pmcid, "rettype": "xml"}
    try:
        r = http_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        pass
    return None

def pmc_list_supplements_from_html(pmcid: str) -> List[Dict[str, str]]:
    """
    PMC 아티클 랜딩 HTML에서 보충자료 링크를 수집한다.
    - /bin/ 경로
    - .pdf/.zip/.xlsx/.xls/.csv/.tsv 확장자
    - 'supp','supplement','S1' 등 키워드
    - '#', 'javascript:' 같은 가짜 링크는 제외
    반환: [{source:'HTML', url:'...', desc:'file|bin|supp-keyword'}, ...]
    """
    base_article = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    try:
        r = http_get(base_article)
    except Exception:
        return []

    html = r.text
    hrefs = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)

    out: List[Dict[str, str]] = []
    seen: set = set()

    def _emit(url: str, desc: str):
        if url in seen:
            return
        seen.add(url)
        out.append({"source": "HTML", "url": url, "desc": desc})

    for href in hrefs:
        # 0) 가짜/무의미 링크 필터링
        if not href:
            continue
        low = href.lower()
        if href.startswith("#") or low.startswith("javascript:"):
            continue

        # 1) 절대/상대 → 절대 URL
        url = href if href.startswith("http") else urljoin(base_article, href)

        # 2) 후보 규칙
        if re.search(r'\.(pdf|zip|xlsx|xls|csv|tsv)(\?|$)', url, flags=re.IGNORECASE):
            _emit(url, "file")
            continue
        if "/bin/" in url:
            _emit(url, "bin")
            continue
        if re.search(r'(supp|supplement|S[0-9]+)', url, flags=re.IGNORECASE):
            _emit(url, "supp-keyword")

    # 디버그: 상위 몇 개 URL 프린트
    if out:
        print(f"[HTML-SUPP] {len(out)} candidates (first 5):")
        for s in out[:5]:
            print("  -", s["url"], "(", s["desc"], ")")

    return out



# JATS XML을 Parsing 해서 Supplement 파일 후보 URL 목록을 만듦 
def pmc_list_supplements_from_jats(pmcid: str) -> List[Dict[str,str]]:
    """
    JATS의 <supplementary-material>, <media>, <ext-link>에서 부록 후보 URL 수집
    상대경로는 PMCID 베이스로 절대화
    """
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

    # 1) media xlink:href
    for media in xml.findall(".//media"):
        href = media.get(XLINK+"href") or media.get("xlink:href")
        if href:
            url = href if href.startswith("http") else urljoin(base_bin, href)
            supps.append({"source":"JATS-media", "url":url, "desc":media.get("mimetype","")})

    # 2) supplementary-material 내부 ext-link / self-uri
    for supp in xml.findall(".//supplementary-material"):
        # self-uri
        for s in supp.findall(".//self-uri"):
            href = s.get(XLINK+"href") or s.get("xlink:href")
            if href:
                url = href if href.startswith("http") else urljoin(base_article, href)
                supps.append({"source":"JATS-selfuri", "url":url, "desc":s.get("content-type","")})
        # ext-link
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

# 주어진 URL로 파일을 다운로드하여 path에 저장. 
def download_to(path: Path, url: str) -> bool:
    """
    주어진 URL을 path에 저장.
    - 기본 URL 시도
    - 필요 시 '?download=1'도 폴백 (함수 내부에서 호출하지 않음: 상위 로직에서 URL 그대로 전달)
    - 저장 후 파일 크기 출력
    """
    try:
        r = http_get(url, stream=True)
        # 저장
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # 디버그: 저장 결과 로그
        try:
            size = path.stat().st_size
            print(f"[DL] saved {size} bytes → {path}")
        except Exception:
            pass

        # 크기 0이면 실패 처리
        if not path.exists() or path.stat().st_size == 0:
            return False
        return True
    except Exception as e:
        print(f"[DL-ERR] {url} -> {e}")
        return False


# =========================
# 메인 파이프라인
# =========================

# 요약 로그 생성 (작업 이력 누적 가능)
def append_run_log(n, query):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / "runs.csv"
    row = [datetime.now().isoformat(), n, query]
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts","records","query"])
        w.writerow(row)

# 보충자료에 대한 csv를 생성 
def write_supp_index(folder: Path, supps: List[Dict[str,str]]):
    if not supps:
        return
    p = folder / "supp_index.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","url","desc","saved_as"])
        w.writeheader()
        for s in supps:
            w.writerow({**s, "saved_as": ""})

# 사용자가 쿼리 안 넘겼을 떄 harvest()가 사용함. 
def build_default_query():
    return build_query()

# 검색 -> ID 매핑 -> 메타 저장 -> PDF 다운로드 -> Supplement 목록 파싱 + 다운로드
def harvest(retmax=200, query: Optional[str]=None):
    q = query or build_default_query()
    print("[QUERY]\n", q, "\n")
    pmids, total = esearch_pmids(q, retmax=retmax, retstart=0)
    print(f"[ESearch] got {len(pmids)} / total={total}")
    pmc_map = elink_pubmed_to_pmc(pmids)
    meta = esummary(pmids)

    for pid in tqdm(pmids, desc="Downloading PDFs"):
        pmcid = pmc_map.get(pid)
        folder = OUT_DIR / f"{pid}_{(pmcid or 'NA')}"
        folder.mkdir(parents=True, exist_ok=True)




        # PubMed 메타 저장
        if pid in meta:
            save_json(meta[pid], folder / "pubmed_summary.json")


        print(f"PID {pid} → PMCID {pmcid}")
        # 메인 PDF (PMCID가 있을 때만)
        if pmcid:
            try:
                pdf_bytes = pmc_try_main_pdf(pmcid)
            except Exception:
                pdf_bytes = None
            if pdf_bytes:
                save_bin(pdf_bytes, folder / "article.pdf")

            # Supplement 목록 파싱 + 다운로드
            # 1) JATS + HTML 합치기
            supps_jats = pmc_list_supplements_from_jats(pmcid)
            supps_html = pmc_list_supplements_from_html(pmcid)
            # URL 기준 dedup
            supps_map = {}
            for s in (supps_jats + supps_html):
                supps_map[s["url"]] = s
            supps = list(supps_map.values())

            if supps:
                supp_dir = folder / "supplements"
                supp_dir.mkdir(parents=True, exist_ok=True)
                index_rows = []
                saved_cnt = 0
                for s in supps:
                    url = s["url"]
                    # 파일명 결정: URL path 마지막 + 쿼리 제거 → sanitize
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

                # 인덱스 저장
                p = folder / "supp_index.csv"
                with p.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["source","url","desc","saved_as"])
                    w.writeheader()
                    w.writerows(index_rows)

                print(f"[SUPP] {pmcid}: found={len(supps)} saved={saved_cnt}")


    append_run_log(len(pmids), q)
    print("\n[Done] PDF 수집 완료 →", OUT_DIR.resolve())

def harvest_until(target_pdfs=200, query: Optional[str]=None, page_size=200, max_pages=50):
    """
    목표 개수(target_pdfs)만큼 PDF를 저장할 때까지
    PubMed를 페이징(retstart)하며 반복 수집한다.

    - PDF 저장 성공한 경우만 raws/ 에 저장
    - PMCID 없는 PMID는 스킵
    - page_size: ESearch retmax
    - max_pages: 안전장치 (무한 루프 방지)
    """
    q = query or build_default_query()
    print("[QUERY]\n", q, "\n")

    saved = 0
    retstart = 0
    page = 0

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

            # PDF 받기 (성공 시에만 폴더 생성)
            try:
                pdf_bytes = pmc_try_main_pdf(pmcid)
            except Exception as e:
                print(f"[WARN] PDF fetch error for {pmcid}: {e}")
                pdf_bytes = None

            if not pdf_bytes:
                print(f"[MISS] no PDF for {pmcid}")
                continue

            # === 여기서부터 '성공한 경우에만' 저장 ===
            folder = OUT_DIR / f"{pid}_{pmcid}"
            folder.mkdir(parents=True, exist_ok=True)
            save_bin(pdf_bytes, folder / "article.pdf")
            print(f"[OK] saved PDF for {pmcid}")
            saved += 1

            # (선택) 메타 저장
            if pid in meta:
                save_json(meta[pid], folder / "pubmed_summary.json")

            # (선택) 부록 저장
            # (선택) 부록 저장: JATS + HTML 병합
            try:
                supps_jats = pmc_list_supplements_from_jats(pmcid)
                supps_html = pmc_list_supplements_from_html(pmcid)
            except Exception as e:
                print(f"[WARN] supplement parse error for {pmcid}: {e}")
                supps_jats, supps_html = [], []

            # URL 기준 dedup
            supps_map = {}
            for s in (supps_jats + supps_html):
                supps_map[s["url"]] = s
            supps = list(supps_map.values())

            if supps:
                supp_dir = folder / "supplements"
                supp_dir.mkdir(parents=True, exist_ok=True)
                index_rows = []
                saved_cnt = 0
                for s in supps:
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

        # 다음 페이지로
        retstart += len(pmids)
        page += 1

    append_run_log(saved, q)
    print(f"\n[Done] PDF 저장 개수 = {saved} (목표 {target_pdfs}) → {OUT_DIR.resolve()}")

def harvest_pmc_first(target_pdfs=200, query: Optional[str]=None,
                      page_size=200, max_pages=50):
    """
    1단계: PMC(db=pmc)에서 PMCID 직접 수집 → PDF 저장 (성공건만 저장)
    2단계(옵션): 부족하면 PubMed에서 추가로 PMID→PMCID 매핑 후 채움
    """
    # 1) PMC 쿼리
    pmc_q = query or build_pmc_query()
    print("[PMC-QUERY]\n", pmc_q, "\n")

    saved = 0
    failed_pmcids = []  # 실패한 PMCID 목록
    # ---------- 1단계: PMC ----------
    retstart = 0
    page = 0
    while saved < target_pdfs and page < max_pages:
        pmcids, total = esearch_pmcids(pmc_q, retmax=page_size, retstart=retstart)
        if not pmcids:
            print(f"[PMC] 더 이상 PMCID 없음. total={total}, retstart={retstart}")
            break
        print(f"[PMC] page={page+1} retstart={retstart} -> got {len(pmcids)} / total={total}")

        for pmcid in pmcids:
            if saved >= target_pdfs:
                break

            # 1) JATS로 사전 필터링 (PDF 다운로드 전)
            print(f"[JATS-CHECK] Pre-filtering {pmcid} using JATS metadata...")
            if not is_research_paper_from_jats(pmcid):
                print(f"[SKIP] {pmcid} filtered out by JATS metadata, skipping...")
                continue

            # 2) PDF 시도 (내부에서 검증 포함)
            try:
                pdf_bytes = pmc_try_main_pdf(pmcid)
            except Exception as e:
                print(f"[WARN] PMC pdf fetch error {pmcid}: {e}")
                pdf_bytes = None

            if not pdf_bytes:
                print(f"[MISS] no valid research PDF for {pmcid}")
                failed_pmcids.append(pmcid)
                continue

            # === 성공한 경우만 저장 ===
            folder = OUT_DIR / f"{pmcid}"
            folder.mkdir(parents=True, exist_ok=True)
            save_bin(pdf_bytes, folder / "article.pdf")
            print(f"[OK] saved research paper PDF for {pmcid}")
            saved += 1

            # 보충자료 다운로드 비활성화 (PDF만 다운로드)
            # print(f"[INFO] {pmcid}: 보충자료 다운로드 건너뜀 (PDF만 저장)")


        retstart += len(pmcids)
        page += 1

    # ---------- 2단계: 부족분을 PubMed로 보충(옵션) ----------
    if saved < target_pdfs:
        print(f"[INFO] PMC만으로 부족 → PubMed로 보충 시도")
        pub_q = query or build_query()
        retstart = 0
        page = 0
        while saved < target_pdfs and page < max_pages:
            pmids, total = esearch_pmids(pub_q, retmax=page_size, retstart=retstart)
            if not pmids:
                break
            print(f"[PubMed] page={page+1} retstart={retstart} -> got {len(pmids)} / total={total}")
            pmc_map = elink_pubmed_to_pmc(pmids)
            for pid, pmcid in pmc_map.items():
                if saved >= target_pdfs:
                    break
                if not pmcid:
                    continue
                
                # JATS로 사전 필터링
                print(f"[JATS-CHECK] Pre-filtering {pmcid} using JATS metadata...")
                if not is_research_paper_from_jats(pmcid):
                    print(f"[SKIP] {pmcid} filtered out by JATS metadata, skipping...")
                    continue
                
                try:
                    pdf_bytes = pmc_try_main_pdf(pmcid)
                except Exception:
                    pdf_bytes = None
                if not pdf_bytes:
                    continue
                
                folder = OUT_DIR / f"{pid}_{pmcid}"
                folder.mkdir(parents=True, exist_ok=True)
                save_bin(pdf_bytes, folder / "article.pdf")
                print(f"[OK] saved research paper PDF for {pmcid}")
                saved += 1
            retstart += len(pmids)
            page += 1

    # 실패 내역 저장
    if failed_pmcids:
        failed_file = OUT_DIR / "failed_pmcids.txt"
        with open(failed_file, 'w', encoding='utf-8') as f:
            for pmcid in failed_pmcids:
                f.write(f"{pmcid}\n")
        print(f"[FAILED] {len(failed_pmcids)}개 PMCID 다운로드 실패 → {failed_file}")
    
    append_run_log(saved, pmc_q)
    print(f"\n[Done] PDF 저장 개수 = {saved} (목표 {target_pdfs}) → {OUT_DIR.resolve()}")




def main():
    global RATE_SLEEP
    import argparse

    parser = argparse.ArgumentParser(
        description="PubMed→PMC PDF/Supplement 수집기 (Organoid + ADMET 키워드)"
    )
    # === 옵션 정의 (중복 제거) ===
    parser.add_argument("--pmc_first", action="store_true",
                        help="PMC(db=pmc)에서 PMCID 직접 수집 → PDF 우선 수집 모드")
    parser.add_argument("--target_pdfs", type=int, default=5000,
                        help="저장할 PDF 목표 개수. 지정하면 목표 달성까지 페이징 반복")
    parser.add_argument("--retmax", type=int, default=200,
                        help="페이지당 문서 수 (ESearch retmax)")
    parser.add_argument("--max_pages", type=int, default=50,
                        help="최대 페이지 수 (안전장치)")
    parser.add_argument("--query", type=str, default=None,
                        help="사용자 정의 쿼리 문자열(미지정 시 기본 쿼리)")
    parser.add_argument("--email", type=str, default=EMAIL,
                        help="NCBI 이메일 (기본: 전역 EMAIL)")
    parser.add_argument("--api_key", type=str, default=API_KEY,
                        help="NCBI API 키 (기본: 전역 API_KEY)")
    parser.add_argument("--sleep", type=float, default=RATE_SLEEP,
                       help="레이트리밋(초). API키 없으면 0.5~0.6 권장")
    parser.add_argument("--output_dir", type=str, default="./raws_test",
                       help="논문을 저장할 디렉토리 (기본: ./raws_test)")

    args = parser.parse_args()

    # === 실행 전 설정 ===
    if args.email:
        Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
    if args.sleep is not None:
        RATE_SLEEP = max(0.2, float(args.sleep))
    
    # 출력 디렉토리 설정
    global OUT_DIR
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUT_DIR.resolve()}")

    if (getattr(Entrez, "email", None) is None) or (Entrez.email == "youremail@example.com"):
        print("[경고] Entrez.email을 반드시 본인 이메일로 설정하세요. (--email 또는 환경변수 NCBI_EMAIL)")
    if not args.api_key:
        print("[안내] API 키 없이 실행합니다. 호출 간 대기(sleep)를 길게 두는 것이 안전합니다.")

    # === 실행 분기 (한 번만 호출) ===
    try:
        if args.pmc_first:
            # PMC 우선 모드: PMCID 직접 수집해서 PDF 목표 개수 채우기
            harvest_pmc_first(target_pdfs=args.target_pdfs or 5000,
                              query=args.query,
                              page_size=args.retmax,
                              max_pages=args.max_pages)
        elif args.target_pdfs:
            # PubMed 기반 페이지 순회로 목표 개수 채우기
            harvest_until(target_pdfs=args.target_pdfs,
                          query=args.query,
                          page_size=args.retmax,
                          max_pages=args.max_pages)
        else:
            # 단발 수집 (현재 페이지 retmax 만큼만)
            # harvest() 내부에서 esearch_pmids 호출 시 (ids, total)로 받도록 수정했는지 확인
            harvest(retmax=args.retmax, query=args.query)
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 작업을 취소했습니다.")
    except Exception as e:
        print(f"[오류] 수집 중 예외 발생: {e}")

def test():
    """
    단일 PMCID로 메인 PDF + 보충자료 저장 테스트.
    실행: python collect_raw.py  (main 대신 test 호출 부분이 활성화되어 있어야 함)
    """
    pmcid = "PMC11469493"  # 원하는 PMCID로 바꿔서 테스트

    # 출력 디렉터리 확인
    print("[OUT_DIR]", OUT_DIR.resolve())

    # 메인 PDF 저장
    pdf = pmc_try_main_pdf(pmcid)
    folder = OUT_DIR / pmcid
    folder.mkdir(parents=True, exist_ok=True)
    if pdf:
        (folder / "article.pdf").write_bytes(pdf)
        print(f"[OK] main PDF saved → {folder/'article.pdf'}")
    else:
        print("[MISS] main PDF not found")

    # 보충자료 수집 (JATS + HTML 병합 + dedup)
    try:
        supps_j = pmc_list_supplements_from_jats(pmcid)
    except Exception as e:
        print(f"[WARN] JATS parse error: {e}")
        supps_j = []
    try:
        supps_h = pmc_list_supplements_from_html(pmcid)
    except Exception as e:
        print(f"[WARN] HTML scan error: {e}")
        supps_h = []

    print("JATS supp:", len(supps_j), "HTML supp:", len(supps_h))

    supps_map: Dict[str, Dict[str, str]] = {}
    for s in (supps_j + supps_h):
        supps_map[s["url"]] = s
    supps = list(supps_map.values())

    if not supps:
        print("[SUPP] no supplements found")
        return

    # 보충자료 다운로드
    supp_dir = folder / "supplements"
    supp_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    saved_cnt = 0

    for i, s in enumerate(supps, 1):
        url = s["url"]
        print(f"[SUPP-TRY {i}/{len(supps)}] {url}")

        # 파일명 생성: 경로 마지막 + ?앞까지만 → sanitize
        raw_name = urlparse(url).path.split("/")[-1] or "supp"
        raw_name = raw_name.split("?")[0]
        fname = sanitize_filename(raw_name if raw_name else "supp")
        out = supp_dir / fname

        ok = False
        if url.startswith("http"):
            ok = download_to(out, url)
            # 폴백: ?download=1
            if (not ok) and ("?" not in url):
                ok = download_to(out, url + "?download=1")

        index_rows.append({**s, "saved_as": (str(out.name) if ok else "")})
        if ok:
            saved_cnt += 1
            print(f"[SUPP-OK] {fname}")
        else:
            print(f"[SUPP-MISS] {url}")

    # 인덱스 저장
    p = folder / "supp_index.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "url", "desc", "saved_as"])
        w.writeheader()
        w.writerows(index_rows)

    print(f"[SUPP] found={len(supps)} saved={saved_cnt} → {supp_dir}")

    # 저장된 파일 목록 요약
    if supp_dir.exists():
        saved_files = sorted([p.name for p in supp_dir.iterdir() if p.is_file()])
        print(f"[SUPP-LS] {len(saved_files)} files:")
        for name in saved_files[:10]:
            print("  -", name)
        if len(saved_files) > 10:
            print(f"  ... (+{len(saved_files)-10} more)")


if __name__ == "__main__":
    main()
    # test()
    # PMC에서 바로 PMCID를 모아서, PDF 200개 채울 때까지 페이징
    # python collect_raw.py --pmc_first --target_pdfs 200 --retmax 200 --sleep 0.6
