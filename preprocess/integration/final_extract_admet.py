#!/usr/bin/env python3
"""
Final ADMET Extraction - 통합 버전
- 텍스트 추출, 이미지 분석, 보충자료 추출 결과, Coreference를 통합
- GPT-4o로 최종 구조화된 ADMET 테이블 생성
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic Models for Structured Output (nanoMINER 방식)
class ADMETIndicator(BaseModel):
    """ADMET 지표 하나 (value, unit, source 포함)"""
    value: Union[float, int, str, bool, None] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    ph: Optional[float] = None  # logD용
    assay: Optional[str] = None  # CYP, hERG용

class LipinskiIndicator(BaseModel):
    """Lipinski's Rule of Five"""
    rule_of_five_pass: Optional[bool] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    source: Optional[str] = None

class CYPInhibition(BaseModel):
    """CYP Inhibition status"""
    status: Optional[str] = None  # "yes/no/unknown"
    source: Optional[str] = None

class DILI(BaseModel):
    """Drug-induced liver injury risk"""
    risk: Optional[str] = None  # "High/Medium/Low"
    source: Optional[str] = None

class AmesTest(BaseModel):
    """Ames test result"""
    result: Optional[str] = None  # "Positive/Negative"
    source: Optional[str] = None

class Carcinogenicity(BaseModel):
    """Carcinogenicity result"""
    result: Optional[str] = None  # "Positive/Negative"
    source: Optional[str] = None

class ADMETRecord(BaseModel):
    """화합물 하나의 ADMET 레코드"""
    compound_name: str
    aliases: List[str] = Field(default_factory=list)
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    well_position: Optional[str] = None
    source_ids: List[str] = Field(default_factory=list)
    
    # ADMET 지표들
    caco2: Optional[ADMETIndicator] = None
    mdck: Optional[ADMETIndicator] = None
    pampa: Optional[ADMETIndicator] = None
    lipinski: Optional[LipinskiIndicator] = None
    logd: Optional[ADMETIndicator] = None
    logs: Optional[ADMETIndicator] = None
    pka: Optional[ADMETIndicator] = None
    ppb: Optional[ADMETIndicator] = None
    bbb: Optional[ADMETIndicator] = None
    vd: Optional[ADMETIndicator] = None
    cyp1a2: Optional[ADMETIndicator] = None
    cyp2c9: Optional[ADMETIndicator] = None
    cyp2c19: Optional[ADMETIndicator] = None
    cyp2d6: Optional[ADMETIndicator] = None
    cyp3a4: Optional[ADMETIndicator] = None
    cyp_inhibition: Optional[CYPInhibition] = None
    cl: Optional[ADMETIndicator] = None
    t_half: Optional[ADMETIndicator] = None
    herg: Optional[ADMETIndicator] = None
    dili: Optional[DILI] = None
    ames: Optional[AmesTest] = None
    carcinogenicity: Optional[Carcinogenicity] = None
    additional_indicators: Optional[Dict[str, ADMETIndicator]] = Field(default_factory=dict)
    notes: Optional[str] = None

class ADMETResponse(BaseModel):
    """최종 ADMET 추출 결과"""
    schema_version: str = "2.0"
    created_at: str
    pmc_id: str
    records: List[ADMETRecord]

class FinalADMETExtractor:
    def __init__(self, base_dir: str = "data_test"):
        """
        Args:
            base_dir: 기본 디렉토리 (모든 입력/출력이 이 하위에 위치)
                     - base_dir/raws/ : 원문 PDF
                     - base_dir/supp/ : 보충자료
                     - base_dir/text_extracted/ : 텍스트 추출 결과
                     - base_dir/graph_analyzed/ : 이미지 분석 결과
                     - base_dir/supp_extracted/ : 보충자료 추출 결과
                     - base_dir/entity_analyzed/ : 코어퍼런스 분석 결과
                     - base_dir/final_extracted/ : 최종 ADMET 결과
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.base_dir = Path(base_dir)
    
    def load_text_extracted(self, text_path: Path) -> str:
        """추출된 텍스트 로드 - ADMET 관련 부분만 필터링"""
        if not text_path.exists():
            logger.warning(f"텍스트 파일 없음: {text_path}")
            return ""
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # ADMET 관련 키워드로 필터링하여 관련 부분만 추출
        admet_keywords = [
            'Caco-2', 'MDCK', 'PAMPA', 'Lipinski', 'logD', 'logS', 'pKa',
            'plasma protein binding', 'PPB', 'blood-brain barrier', 'BBB',
            'volume of distribution', 'Vd', 'CYP1A2', 'CYP2C9', 'CYP2C19',
            'CYP2D6', 'CYP3A4', 'CYP inhibition', 'clearance', 'CL',
            'half-life', 't1/2', 'hERG', 'DILI', 'Ames', 'carcinogenicity',
            'permeability', 'absorption', 'distribution', 'metabolism', 'excretion', 'toxicity'
        ]
        
        # 문장 단위로 분리하고 ADMET 관련 문장만 추출
        sentences = text.split('.')
        relevant_sentences = []
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in admet_keywords):
                relevant_sentences.append(sentence.strip())
        
        # 관련 문장이 있으면 그것만 사용, 없으면 처음 15000자 사용
        if relevant_sentences:
            filtered_text = '. '.join(relevant_sentences[:200])  # 최대 200개 문장
            logger.info(f"ADMET 관련 문장 {len(relevant_sentences)}개 중 {min(200, len(relevant_sentences))}개 사용")
            return filtered_text[:15000]  # 최대 15000자
        else:
            logger.warning("ADMET 관련 문장 없음, 처음 15000자 사용")
            return text[:15000]
    
    def _extract_compounds_from_analysis(self, analysis: Any, compounds: set, depth=0):
        """재귀적으로 화합물 이름 추출 (GPT의 자유로운 구조 지원)"""
        if depth > 5:
            return
        
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                # 화합물 관련 키워드 찾기
                if any(kw in key.lower() for kw in ['compound', 'drug', 'molecule', 'chemical', 'title', 'name', 'label']):
                    if isinstance(value, str) and 2 < len(value) < 100:
                        # 화합물일 가능성 (짧은 문자열)
                        # 단, 너무 일반적인 단어 제외
                        if value.lower() not in ['control', 'treatment', 'group', 'sample', 'test', 'data', 'figure', 'table']:
                            compounds.add(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, str) and 2 < len(v) < 100:
                                if v.lower() not in ['control', 'treatment', 'group', 'sample', 'test', 'data', 'figure', 'table']:
                                    compounds.add(v)
                            elif isinstance(v, dict):
                                self._extract_compounds_from_analysis(v, compounds, depth+1)
                
                # 재귀 탐색
                self._extract_compounds_from_analysis(value, compounds, depth+1)
        
        elif isinstance(analysis, list):
            for item in analysis:
                self._extract_compounds_from_analysis(item, compounds, depth+1)
    
    def load_image_analysis(self, img_path: Path) -> Dict[str, Any]:
        """이미지 분석 결과 로드 및 화합물 추출"""
        if not img_path.exists():
            logger.warning(f"이미지 분석 파일 없음: {img_path}")
            return {}
        
        with open(img_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 화합물 추출
        compounds_in_images = set()
        image_compound_data = {}  # 화합물별 이미지 정보
        
        if isinstance(data, list):
            for item in data:
                analysis = item.get('analysis', {})
                filename = item.get('filename', 'unknown')
                
                # 재귀적으로 화합물 추출
                found_compounds = set()
                self._extract_compounds_from_analysis(analysis, found_compounds)
                
                for comp in found_compounds:
                    compounds_in_images.add(comp)
                    if comp not in image_compound_data:
                        image_compound_data[comp] = []
                    image_compound_data[comp].append({
                        "filename": filename,
                        "image_class": item.get('class', 'unknown'),
                        "analysis_sample": str(analysis)[:500]  # 샘플만
                    })
        
        # 요약 생성
        summary = {
            "total_images": len(data) if isinstance(data, list) else 0,
            "compounds_found": list(compounds_in_images),
            "compound_image_map": {k: v[:3] for k, v in image_compound_data.items()},  # 각 화합물당 최대 3개 이미지
            "sample_analyses": data[:5] if isinstance(data, list) else []
        }
        
        return summary
    
    def load_supplement_data(self, supp_dir: Path) -> Dict[str, Any]:
        """보충자료 추출 결과 로드 (Excel, Word, PDF 텍스트, PDF 이미지 분석)"""
        supp_data = {
            "excel": [],
            "word": [],
            "pdf_text": [],
            "pdf_images": []
        }
        
        # Excel 추출 결과 (신버전 경로)
        excel_dir = supp_dir / "excel"
        if excel_dir.exists():
            for json_file in excel_dir.glob("*compounds_from_excel.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        supp_data["excel"].extend(data if isinstance(data, list) else [])
                except Exception as e:
                    logger.warning(f"Excel JSON 로드 실패 {json_file}: {e}")
        
        # Excel 추출 결과 (구버전 경로: supplement_extracted/ - 호환성)
        supp_old_dir = Path("supplement_extracted") / supp_dir.name
        if not excel_dir.exists() and supp_old_dir.exists():
            logger.info(f"기존 보충자료 경로 발견: {supp_old_dir}")
            supp_old_excel_dir = supp_old_dir / "excel"
            if supp_old_excel_dir.exists():
                for json_file in supp_old_excel_dir.glob("*compounds_from_excel.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            supp_data["excel"].extend(data if isinstance(data, list) else [])
                    except Exception as e:
                        logger.warning(f"Excel JSON 로드 실패 (구버전) {json_file}: {e}")
            # 구버전 경로에서 직접 파일 찾기
            for json_file in supp_old_dir.glob("*compounds_from_excel.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        supp_data["excel"].extend(data if isinstance(data, list) else [])
                except Exception as e:
                    logger.warning(f"Excel JSON 로드 실패 (구버전) {json_file}: {e}")
        
        # Word 추출 결과 (신버전 경로)
        word_dir = supp_dir / "word"
        if word_dir.exists():
            for json_file in word_dir.glob("*compounds_from_word.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        supp_data["word"].extend(data if isinstance(data, list) else [])
                except Exception as e:
                    logger.warning(f"Word JSON 로드 실패 {json_file}: {e}")
        
        # Word 추출 결과 (구버전 경로)
        if supp_old_dir.exists():
            for json_file in supp_old_dir.glob("*compounds_from_word.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        supp_data["word"].extend(data if isinstance(data, list) else [])
                except Exception as e:
                    logger.warning(f"Word JSON 로드 실패 (구버전) {json_file}: {e}")
        
        # PDF 텍스트 추출 결과 (요약만)
        pdf_text_dir = supp_dir / "pdf_text"
        if pdf_text_dir.exists():
            text_files = list(pdf_text_dir.rglob("*.txt"))
            if text_files:
                # 첫 번째 텍스트 파일만 샘플로
                try:
                    with open(text_files[0], 'r', encoding='utf-8') as f:
                        text = f.read()
                        supp_data["pdf_text"].append({
                            "file": text_files[0].name,
                            "text_preview": text[:2000]  # 처음 2000자만
                        })
                except Exception as e:
                    logger.warning(f"PDF 텍스트 로드 실패: {e}")
        
        # PDF 이미지 분석 결과 (YOLO + GPT-4o Vision)
        # pdf_info/{pdf_name}_yolo_gpt_analysis.json 형태로 저장된 파일들 로드
        pdf_info_dir = supp_dir / "pdf_info"
        if pdf_info_dir.exists():
            yolo_files = list(pdf_info_dir.glob("*_yolo_gpt_analysis.json"))
            if yolo_files:
                try:
                    # 모든 YOLO 분석 결과 통합
                    all_yolo_compounds = []
                    for yolo_file in yolo_files:
                        with open(yolo_file, 'r', encoding='utf-8') as f:
                            yolo_data = json.load(f)
                            compounds = yolo_data.get("compounds", [])
                            
                            # 화합물 데이터를 pdf_images 형식으로 변환
                            for compound in compounds:
                                comp_name = compound.get("compound_name", "")
                                attributes = compound.get("attributes", {})
                                
                                # 각 속성을 레코드로 변환
                                for attr_name, attr_data in attributes.items():
                                    value = attr_data.get("value", "")
                                    if isinstance(value, list):
                                        for v in value:
                                            all_yolo_compounds.append({
                                                "compound_name": comp_name,
                                                "attribute_name": attr_name,
                                                "value": str(v),
                                                "unit": attr_data.get("unit", ""),
                                                "source": attr_data.get("source", yolo_file.name)
                                            })
                                    else:
                                        all_yolo_compounds.append({
                                            "compound_name": comp_name,
                                            "attribute_name": attr_name,
                                            "value": str(value),
                                            "unit": attr_data.get("unit", ""),
                                            "source": attr_data.get("source", yolo_file.name)
                                        })
                    
                    supp_data["pdf_images"] = all_yolo_compounds
                    logger.info(f"YOLO 분석 결과 로드: {len(yolo_files)}개 파일, {len(all_yolo_compounds)}개 레코드")
                except Exception as e:
                    logger.warning(f"PDF 이미지 분석 로드 실패: {e}")
        
        return supp_data
    
    def load_coreference(self, coref_path: Path) -> Dict[str, Any]:
        """Coreference dictionary 로드"""
        if not coref_path.exists():
            logger.warning(f"Coreference 파일 없음: {coref_path}")
            return {"coreference_groups": {}, "relationships": []}
        
        with open(coref_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_with_gpt4o(
        self,
        text_content: str,
        image_analysis: Dict[str, Any],
        supplement_data: Dict[str, Any],
        coreference: Dict[str, Any],
        pmc_id: str,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """GPT-4o로 최종 ADMET 테이블 추출 (배치 처리 지원)"""
        
        logger.info("GPT-4o로 최종 ADMET 테이블 추출 중...")
        
        # 배치 처리: 화합물이 많으면 여러 번 나눠서 추출
        all_compounds = set()
        
        # 모든 소스에서 화합물 수집
        if supplement_data.get('excel'):
            excel_items = supplement_data.get('excel', [])
            for item in excel_items:
                comp_name = item.get('compound_name', '').strip()
                if comp_name:
                    all_compounds.add(comp_name)
        
        if supplement_data.get('pdf_images'):
            pdf_img_items = supplement_data.get('pdf_images', [])
            for item in pdf_img_items:
                comp_name = item.get('compound_name', '').strip()
                if comp_name:
                    all_compounds.add(comp_name)
        
        total_compounds = len(all_compounds)
        logger.info(f"총 발견된 화합물 수: {total_compounds}개")
        
        # 화합물이 많으면 배치 처리
        if total_compounds > batch_size:
            logger.info(f"화합물 수가 {batch_size}개를 초과하여 배치 처리 시작 ({total_compounds}개 → {((total_compounds - 1) // batch_size) + 1}개 배치)")
            return self._extract_with_batching(
                text_content, image_analysis, supplement_data, coreference, pmc_id, 
                all_compounds, batch_size
            )
        else:
            logger.info("화합물 수가 적어서 단일 배치로 처리")
            return self._extract_single_batch(
                text_content, image_analysis, supplement_data, coreference, pmc_id
            )
    
    def _extract_single_batch(
        self,
        text_content: str,
        image_analysis: Dict[str, Any],
        supplement_data: Dict[str, Any],
        coreference: Dict[str, Any],
        pmc_id: str,
        target_compounds: set = None
    ) -> Dict[str, Any]:
        """단일 배치로 추출 (기존 로직)"""
        
        # 보충자료 요약 (각 타입이 없어도 OK)
        supp_summary_parts = []
        
        if supplement_data.get('excel'):
            excel_items = supplement_data.get('excel', [])
            # 화합물별로 그룹화 (모든 compound, 모든 속성 포함, 샘플링 없음)
            compound_groups = {}
            
            for item in excel_items:
                comp_name = item.get('compound_name', '').strip()
                attr_name = item.get('attribute_name', '').strip()
                value = item.get('value', '').strip()
                
                # compound가 있으면 모든 속성 포함 (ADMET 필터링 없음, 샘플링 없음)
                if comp_name and attr_name and value:
                    if comp_name not in compound_groups:
                        compound_groups[comp_name] = {}
                    # 새로운 속성이 나오면 추가, 기존 속성이면 값 업데이트
                    if attr_name in compound_groups[comp_name]:
                        # 기존 값이 있으면 리스트로 만들어서 추가하거나, 문자열로 결합
                        existing = compound_groups[comp_name][attr_name]
                        if isinstance(existing, list):
                            compound_groups[comp_name][attr_name].append(value)
                        else:
                            compound_groups[comp_name][attr_name] = [existing, value]
                    else:
                        compound_groups[comp_name][attr_name] = value
            
            # 화합물 이름만 리스트로 제공 (전체 데이터가 너무 길어서)
            compound_names = sorted(list(compound_groups.keys()))
            # 배치 처리인 경우: 화합물 리스트는 전체를 유지, 속성 데이터만 필터링
            if target_compounds:
                filtered_compound_groups = {k: v for k, v in compound_groups.items() if k in target_compounds}
            else:
                filtered_compound_groups = compound_groups
            
            supp_summary_parts.append(f"""
**Supplementary Excel Data:**
- Total compound-attribute pairs: {len(excel_items)}
- Unique compounds: {len(compound_names)} (ALL - must extract all)
- **ALL COMPOUND NAMES (MUST EXTRACT ALL):** {json.dumps(compound_names, ensure_ascii=False)}
- Sample compound data (first 5 compounds with all attributes):
{json.dumps(dict(list(filtered_compound_groups.items())[:5]), ensure_ascii=False, indent=2)}
""")
        
        if supplement_data.get('word'):
            word_items = supplement_data.get('word', [])
            # Word 데이터도 compound 중심으로 그룹화 (샘플링 없음)
            word_compound_groups = {}
            
            for item in word_items:
                comp_name = item.get('compound_name', '').strip()
                attr_name = item.get('attribute_name', '').strip()
                value = item.get('value', '').strip()
                
                if comp_name and attr_name and value:
                    if comp_name not in word_compound_groups:
                        word_compound_groups[comp_name] = {}
                    if attr_name in word_compound_groups[comp_name]:
                        existing = word_compound_groups[comp_name][attr_name]
                        if isinstance(existing, list):
                            word_compound_groups[comp_name][attr_name].append(value)
                        else:
                            word_compound_groups[comp_name][attr_name] = [existing, value]
                    else:
                        word_compound_groups[comp_name][attr_name] = value
            
            word_compound_names = sorted(list(word_compound_groups.keys()))
            supp_summary_parts.append(f"""
**Supplementary Word Data:**
- Total compound-attribute pairs: {len(word_items)}
- Unique compounds: {len(word_compound_groups)}
- **ALL COMPOUND NAMES (MUST EXTRACT ALL):** {json.dumps(word_compound_names, ensure_ascii=False)}
- Sample compound data (first 5 compounds with all attributes):
{json.dumps(dict(list(word_compound_groups.items())[:5]), ensure_ascii=False, indent=2)}
""")
        
        if supplement_data.get('pdf_text'):
            supp_summary_parts.append(f"""
**Supplementary PDF Text:**
{json.dumps(supplement_data.get('pdf_text', []), ensure_ascii=False, indent=2)[:2000]}
""")
        
        if supplement_data.get('pdf_images'):
            # 보충자료 PDF 이미지 (YOLO + GPT 분석 결과)
            # 이미 화합물-속성 쌍 리스트 형태로 변환되어 있음
            pdf_img_items = supplement_data.get('pdf_images', [])
            
            # 화합물별로 그룹화
            pdf_img_compound_groups = {}
            pdf_img_compounds = set()
            
            for item in pdf_img_items:
                comp_name = item.get('compound_name', '').strip()
                attr_name = item.get('attribute_name', '').strip()
                value = item.get('value', '').strip()
                
                if comp_name and attr_name and value:
                    pdf_img_compounds.add(comp_name)
                    
                    if comp_name not in pdf_img_compound_groups:
                        pdf_img_compound_groups[comp_name] = {}
                    
                    if attr_name in pdf_img_compound_groups[comp_name]:
                        existing = pdf_img_compound_groups[comp_name][attr_name]
                        if isinstance(existing, list):
                            pdf_img_compound_groups[comp_name][attr_name].append(value)
                        else:
                            pdf_img_compound_groups[comp_name][attr_name] = [existing, value]
                    else:
                        pdf_img_compound_groups[comp_name][attr_name] = value
            
            pdf_img_compound_names = sorted(list(pdf_img_compounds))
            # 배치 처리인 경우: 화합물 리스트는 전체를 유지, 속성 데이터만 필터링
            # (GPT가 모든 화합물을 인식하고 추출할 수 있도록)
            if target_compounds:
                # 속성 데이터는 타겟 화합물만 필터링
                filtered_compound_groups = {k: v for k, v in pdf_img_compound_groups.items() if k in target_compounds}
            else:
                filtered_compound_groups = pdf_img_compound_groups
            
            # 화합물 리스트는 전체를 항상 포함 (배치와 무관)
            compound_list_str = ', '.join(pdf_img_compound_names)
            supp_summary_parts.append(f"""
**Supplementary PDF Images (YOLO + GPT-4o Vision Analysis):**
- Total compound-attribute pairs: {len(pdf_img_items)}
- Unique compounds: {len(pdf_img_compound_names)} (ALL - must extract all)
- **CRITICAL: YOU MUST EXTRACT ALL {len(pdf_img_compound_names)} COMPOUNDS LISTED BELOW**
- **ALL COMPOUND NAMES (EXTRACT EVERY SINGLE ONE):** {compound_list_str}
- Sample compound data (first 10 compounds with all attributes):
{json.dumps(dict(list(filtered_compound_groups.items())[:10]), ensure_ascii=False, indent=2)}
""")
        
        supp_summary = "\n".join(supp_summary_parts) if supp_summary_parts else "\n**Supplementary Materials: None**\n"
        
        # 이미지 분석 요약 (화합물 정보 포함, 더 간결하게)
        compounds_in_images = image_analysis.get('compounds_found', [])
        compound_image_map = image_analysis.get('compound_image_map', {})
        
        # 화합물별로 간단히 요약
        compound_summary = {}
        for comp, img_list in list(compound_image_map.items())[:30]:  # 최대 30개 화합물
            compound_summary[comp] = {
                "image_count": len(img_list),
                "image_types": list(set([img.get('image_class', 'unknown') for img in img_list]))
            }
        
        img_summary = f"""
**Main Paper Image Analysis:**
- Total images: {image_analysis.get('total_images', 0)}
- Compounds/treatments found: {len(compounds_in_images)}
- Compound summary (max 30):
{json.dumps(compound_summary, ensure_ascii=False, indent=2)[:2000]}
- Sample analyses (first 2): {json.dumps(image_analysis.get('sample_analyses', [])[:2], ensure_ascii=False, indent=2)[:1500]}
"""
        
        # Coreference 요약 (화합물만, 더 간결하게)
        compound_coref = {}
        for k, v in list(coreference.get('coreference_groups', {}).items())[:20]:
            if v.get('entity_type') == 'compound':
                compound_coref[k] = {
                    "aliases": v.get('aliases', [])[:5],  # 최대 5개 별칭
                    "entity_type": v.get('entity_type')
                }
        
        coref_summary = f"""
**Coreference Dictionary (Compounds Only):**
- Compound groups: {len(compound_coref)}
- Total groups: {len(coreference.get('coreference_groups', {}))}
- Key compound groups (max 20):
{json.dumps(compound_coref, ensure_ascii=False, indent=2)[:1500]}
"""
        
        prompt = f"""## Task (Objective)
Extract **ALL compounds** and their **complete attributes/indicators** from the provided data sources by **organically connecting**:
- Main text (ADMET-related sentences)
- Figures/Tables (from main paper image analysis)
- Supplementary materials (Excel, Word, PDF text, PDF images)
- Coreference dictionary (entity aliases and relationships)

Output MUST be **STRICT JSON** following the exact schema. No markdown, no explanations, no code blocks - ONLY valid JSON.

## Data Sources (Provided Below)
You have access to:
1. **Main Text**: ADMET-related sentences extracted from PDF
2. **Image Analysis**: Figures and tables from main paper with compound information
3. **Supplementary Materials**: 
   - Excel data (compound-attribute pairs)
   - Word data (compound-attribute pairs)
   - PDF text (preview)
   - PDF images (YOLO-extracted compound data: SMILES, Well Position, etc.)
4. **Coreference Dictionary**: Compound aliases and relationships

**CRITICAL - Exhaustiveness:**
- **NEVER stop after the first match**. Extract from ALL pages, ALL tables, ALL sheets, ALL figures.
- **MUST extract ALL compounds listed in the data summaries below** - if a compound name appears in any source, it MUST be included in the output.
- Process EVERY compound found across ALL sources.
- Cross-reference between sources to prevent omissions.
- **If a compound list is provided (e.g., "ALL COMPOUND NAMES"), extract ALL of them, not just a sample.**

## Data Provided:

**Main Text (ADMET-related only):**
{text_content}

{img_summary}

{supp_summary}

{coref_summary}

## What to Extract

For EACH compound, extract:

### Identifiers
- `compound_name`: Canonical name (use coreference to resolve aliases)
- `aliases`: List of all aliases/abbreviations (e.g., ["APA", "Liposomal APA"])
- `smiles`: SMILES string if available, else null
- `inchi`: InChI if available, else null
- `well_position`: Plate position if available (e.g., "1 A01"), else null
- `source_ids`: List of sources where found (e.g., ["Supplementary Table 1", "Figure 2", "Main text p.14"])

### Normalization Rules
- **Alias Merging (Coreference)**: Use coreference dictionary to merge aliases into one record.
  Example: "APA" = "6-(4-aminophenyl)-N-(3,4,5-trimethoxyphenyl)pyrazin-2-amine" = "Liposomal APA" → ONE record with aliases list.
- **SMILES Canonicalization**: Normalize SMILES notation if possible.
- **Unit Normalization**: 
  - µM/μM/uM → "µM"
  - hr/h → "h"
  - 1e-6 cm/s → "1e-6 cm/s" (consistent)
- **Conflict Resolution (Priority)**: supplement > image > text
  - If conflicting values, use supplement data
  - Document conflicts in `notes` field

### ADMET Standard Indicators (22 + any additional found)
1. **caco2**: Caco-2 Permeability (units: 1e-6 cm/s or Papp)
2. **mdck**: MDCK Permeability (units: 1e-6 cm/s or Papp)
3. **pampa**: PAMPA (units: 1e-6 cm/s)
4. **lipinski**: Lipinski's Rule of Five (rule_of_five_pass, molecular_weight, logp, hbd, hba)
5. **logd**: Distribution coefficient (ph-dependent, include ph if available)
6. **logs**: Aqueous solubility (log units)
7. **pka**: Acid dissociation constant
8. **ppb**: Plasma protein binding (units: %)
9. **bbb**: Blood-brain barrier (units: logBB)
10. **vd**: Volume of distribution (units: L/kg)
11. **cyp1a2**: CYP1A2 inhibition (units: µM, assay: IC50)
12. **cyp2c9**: CYP2C9 inhibition (units: µM, assay: IC50)
13. **cyp2c19**: CYP2C19 inhibition (units: µM, assay: IC50)
14. **cyp2d6**: CYP2D6 inhibition (units: µM, assay: IC50)
15. **cyp3a4**: CYP3A4 inhibition (units: µM, assay: IC50)
16. **cyp_inhibition**: General CYP inhibition status (yes/no/unknown)
17. **cl**: Clearance (units: mL/min/kg or L/h)
18. **t_half**: Half-life (units: hours or h)
19. **herg**: hERG IC50 (units: µM, assay: IC50)
20. **dili**: Drug-induced liver injury risk (High/Medium/Low)
21. **ames**: Ames test result (Positive/Negative)
22. **carcinogenicity**: Carcinogenicity result (Positive/Negative)
23. **additional_indicators**: Any other attributes found (EC50, Ki, solubility at pH X, etc.)

## Output Schema (STRICT JSON - MUST FOLLOW EXACTLY)
```json
{{
  "schema_version": "2.0",
  "created_at": "ISO 8601 timestamp",
  "pmc_id": "{pmc_id}",
  "records": [
    {{
      "compound_name": "Canonical compound name (use coreference to resolve aliases)",
      "aliases": ["alias1", "alias2"],
      "smiles": "SMILES string if available, else null",
      "inchi": "InChI string if available, else null",
      "well_position": "Plate position (e.g., '1 A01') or null",
      "source_ids": ["Supplementary Table 1", "Figure 2", "Main text p.14"],
      "caco2": {{"value": number or null, "unit": "1e-6 cm/s" or null, "source": "text/supplement/image"}},
      "mdck": {{"value": number or null, "unit": "1e-6 cm/s" or null, "source": "text/supplement/image"}},
      "pampa": {{"value": number or null, "unit": "1e-6 cm/s" or null, "source": "text/supplement/image"}},
      "lipinski": {{"rule_of_five_pass": true/false/null, "molecular_weight": number or null, "logp": number or null, "hbd": number or null, "hba": number or null, "source": "text/supplement/image"}},
      "logd": {{"value": number or null, "ph": number or null, "source": "text/supplement/image"}},
      "logs": {{"value": number or null, "unit": "log units" or null, "source": "text/supplement/image"}},
      "pka": {{"value": number or null, "source": "text/supplement/image"}},
      "ppb": {{"value": number or null, "unit": "%" or null, "source": "text/supplement/image"}},
      "bbb": {{"value": number or null, "unit": "logBB" or null, "source": "text/supplement/image"}},
      "vd": {{"value": number or null, "unit": "L/kg" or null, "source": "text/supplement/image"}},
      "cyp1a2": {{"value": number or null, "unit": "µM" or null, "assay": "IC50" or null, "source": "text/supplement/image"}},
      "cyp2c9": {{"value": number or null, "unit": "µM" or null, "assay": "IC50" or null, "source": "text/supplement/image"}},
      "cyp2c19": {{"value": number or null, "unit": "µM" or null, "assay": "IC50" or null, "source": "text/supplement/image"}},
      "cyp2d6": {{"value": number or null, "unit": "µM" or null, "assay": "IC50" or null, "source": "text/supplement/image"}},
      "cyp3a4": {{"value": number or null, "unit": "µM" or null, "assay": "IC50" or null, "source": "text/supplement/image"}},
      "cyp_inhibition": {{"status": "yes/no/unknown" or null, "source": "text/supplement/image"}},
      "cl": {{"value": number or null, "unit": "mL/min/kg" or "L/h" or null, "source": "text/supplement/image"}},
      "t_half": {{"value": number or null, "unit": "hours" or "h" or null, "source": "text/supplement/image"}},
      "herg": {{"value": number or null, "unit": "µM" or null, "assay": "IC50" or null, "source": "text/supplement/image"}},
      "dili": {{"risk": "High/Medium/Low" or null, "source": "text/supplement/image"}},
      "ames": {{"result": "Positive/Negative" or null, "source": "text/supplement/image"}},
      "carcinogenicity": {{"result": "Positive/Negative" or null, "source": "text/supplement/image"}},
      "additional_indicators": {{"indicator_name": {{"value": any, "unit": string or null, "source": "text/supplement/image"}}}},
      "notes": "Additional information or null"
    }}
  ]
}}
```

## Procedure (Steps)
1. **FIRST: Identify ALL compound names from the lists provided above** (especially "ALL COMPOUND NAMES" sections)
2. **Create a record for EVERY compound name found** - even if ADMET data is incomplete (use null)
3. **For each compound**, extract ALL attributes/indicators from ALL sources
4. **Use coreference dictionary** to merge aliases into canonical names
5. **Normalize units and notation** (SMILES, units, etc.)
6. **Resolve conflicts** using priority (supplement > image > text), document in `notes`
7. **Output STRICT JSON only** - no markdown, no explanations, no code blocks
8. **Output Format: Keep it concise** - use null for missing values, don't include verbose descriptions

**IMPORTANT: If a compound list is provided (e.g., "ALL COMPOUND NAMES: CBK001, CBK002, ..."), you MUST create a record for EVERY compound in that list, not just a sample.**

## Exhaustiveness (Critical)
- Extract **ALL compounds** found, not just a subset
- **If a numbered list of compounds is provided (e.g., "178 compounds"), you MUST extract ALL of them**
- Include compounds even if ADMET data is incomplete (use null for missing values)
- Cross-reference between sources to ensure no omissions
- Process supplementary Excel/Word data completely (all rows, all sheets)
- Process all YOLO-extracted compounds from PDF images
- **DO NOT skip compounds just because they don't have ADMET values - create records with null values**

## Constraints
- Output MUST be **valid JSON object only**
- **NO markdown formatting** (no ```json code blocks)
- **NO explanations** or natural language
- **NO code** or comments
- If a field is missing, use `null` but **keep the field**

## Final Reminder
Return ONLY the JSON object. Nothing else.
"""
        
        # 프롬프트 디버깅: 실제 전달되는 프롬프트 확인
        prompt_preview = prompt[:2000] + "\n\n... (중략) ...\n\n" + prompt[-1000:] if len(prompt) > 3000 else prompt
        logger.info(f"프롬프트 길이: {len(prompt):,}자")
        logger.info(f"프롬프트 미리보기:\n{prompt_preview}")
        
        # 프롬프트 전체를 파일로 저장 (디버깅용)
        debug_dir = self.base_dir / "final_extracted" / pmc_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / f"{pmc_id}_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(prompt)
        logger.info(f"프롬프트 전체 저장: {debug_dir / f'{pmc_id}_prompt.txt'}")
        
        try:
            # nanoMINER 방식: structured output 사용 (Pydantic BaseModel)
            # response_format을 사용하면 JSON 스키마가 강제되고 모든 필드가 채워짐
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",  # structured output 지원 모델
                messages=[
                    {"role": "system", "content": "You are an expert data extractor for biomedical ADMET. Extract ALL compounds and their attributes from the provided data sources. Return structured data following the exact schema."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16384,
                response_format=ADMETResponse  # Pydantic 모델로 스키마 강제
            )
            
            message = completion.choices[0].message
            if not message.parsed:
                logger.error(f"Structured output parsing failed: {message.refusal}")
                # Fallback: 일반 JSON 파싱 시도
                if message.content:
                    generated_text = message.content
                    if "```json" in generated_text:
                        json_start = generated_text.find("```json") + 7
                        json_end = generated_text.find("```", json_start)
                        json_text = generated_text[json_start:json_end].strip()
                    else:
                        json_text = generated_text.strip()
                    result = json.loads(json_text)
                    return result
                else:
                    raise ValueError(f"Failed to parse structured output: {message.refusal}")
            
            # Pydantic 모델을 dict로 변환
            result = message.parsed.model_dump()
            logger.info(f"GPT-4o structured output 받음 (레코드 수: {len(result.get('records', []))}개)")
            return result
            
        except Exception as e:
            logger.error(f"GPT-4o 추출 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "schema_version": "2.0",
                "created_at": datetime.now().isoformat(),
                "pmc_id": pmc_id,
                "records": []
            }
    
    def _extract_with_batching(
        self,
        text_content: str,
        image_analysis: Dict[str, Any],
        supplement_data: Dict[str, Any],
        coreference: Dict[str, Any],
        pmc_id: str,
        all_compounds: set,
        batch_size: int
    ) -> Dict[str, Any]:
        """배치 처리: 화합물을 여러 그룹으로 나눠 추출"""
        
        compounds_list = sorted(list(all_compounds))
        total_batches = ((len(compounds_list) - 1) // batch_size) + 1
        
        logger.info(f"배치 처리: {len(compounds_list)}개 화합물을 {total_batches}개 배치로 분할")
        
        all_records = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(compounds_list))
            batch_compounds = set(compounds_list[start_idx:end_idx])
            
            logger.info(f"배치 {batch_idx + 1}/{total_batches} 처리 중 ({len(batch_compounds)}개 화합물: {list(batch_compounds)[:5]}...)")
            
            # 이 배치에 해당하는 화합물만 필터링
            filtered_supplement_data = self._filter_supplement_data_by_compounds(
                supplement_data, batch_compounds
            )
            
            # 단일 배치 추출
            batch_result = self._extract_single_batch(
                text_content, image_analysis, filtered_supplement_data, coreference, pmc_id,
                target_compounds=batch_compounds
            )
            
            batch_records = batch_result.get('records', [])
            all_records.extend(batch_records)
            logger.info(f"배치 {batch_idx + 1} 완료: {len(batch_records)}개 레코드 추출")
        
        # 최종 결과 병합
        logger.info(f"모든 배치 완료: 총 {len(all_records)}개 레코드")
        
        # 중복 제거 (같은 화합물이 여러 배치에 포함될 수 있음)
        seen_compounds = {}
        merged_records = []
        for record in all_records:
            comp_name = record.get('compound_name')
            if comp_name not in seen_compounds:
                seen_compounds[comp_name] = record
                merged_records.append(record)
            else:
                # 병합: aliases와 source_ids 통합
                existing = seen_compounds[comp_name]
                existing_aliases = set(existing.get('aliases', []))
                existing_aliases.update(record.get('aliases', []))
                existing['aliases'] = list(existing_aliases)
                
                existing_sources = set(existing.get('source_ids', []))
                existing_sources.update(record.get('source_ids', []))
                existing['source_ids'] = list(existing_sources)
        
        logger.info(f"중복 제거 후: {len(merged_records)}개 레코드")
        
        return {
            "schema_version": "2.0",
            "created_at": datetime.now().isoformat(),
            "pmc_id": pmc_id,
            "records": merged_records
        }
    
    def _filter_supplement_data_by_compounds(
        self,
        supplement_data: Dict[str, Any],
        target_compounds: set
    ) -> Dict[str, Any]:
        """특정 화합물만 필터링"""
        filtered = {}
        
        # Excel 데이터 필터링
        if supplement_data.get('excel'):
            filtered_excel = []
            for item in supplement_data.get('excel', []):
                comp_name = item.get('compound_name', '').strip()
                if comp_name in target_compounds:
                    filtered_excel.append(item)
            filtered['excel'] = filtered_excel
        
        # Word 데이터 필터링
        if supplement_data.get('word'):
            filtered_word = []
            for item in supplement_data.get('word', []):
                comp_name = item.get('compound_name', '').strip()
                if comp_name in target_compounds:
                    filtered_word.append(item)
            filtered['word'] = filtered_word
        
        # PDF 이미지 데이터 필터링
        if supplement_data.get('pdf_images'):
            filtered_pdf_img = []
            for item in supplement_data.get('pdf_images', []):
                comp_name = item.get('compound_name', '').strip()
                if comp_name in target_compounds:
                    filtered_pdf_img.append(item)
            filtered['pdf_images'] = filtered_pdf_img
        
        # PDF 텍스트는 그대로 (필터링 어려움)
        if supplement_data.get('pdf_text'):
            filtered['pdf_text'] = supplement_data.get('pdf_text')
        
        return filtered
    
    def process_pmc(self, pmc_id: str) -> Dict[str, Any]:
        """특정 PMC ID 처리"""
        logger.info(f"최종 ADMET 추출 시작: {pmc_id}")
        
        # 경로 설정
        text_path = self.base_dir / "text_extracted" / pmc_id / "extracted_text.txt"
        # graph_analyzed 또는 graph_extracted에서 이미지 분석 결과 찾기
        img_path = self.base_dir / "graph_analyzed" / pmc_id / "all_analyses.json"
        if not img_path.exists():
            # graph_extracted에서 찾기
            img_path = self.base_dir / "graph_extracted" / pmc_id / "all_analyses.json"
        supp_dir = self.base_dir / "supp_extracted" / pmc_id
        coref_path = self.base_dir / "entity_analyzed" / pmc_id / "global_coreference_gpt.json"
        
        # 만약 global_coreference_gpt.json이 없으면 global_coreference.json 시도
        if not coref_path.exists():
            coref_path = self.base_dir / "entity_analyzed" / pmc_id / "global_coreference.json"
        
        # 데이터 로드
        logger.info("데이터 소스 로딩 중...")
        text_content = self.load_text_extracted(text_path)
        image_analysis = self.load_image_analysis(img_path)
        supplement_data = self.load_supplement_data(supp_dir)
        coreference = self.load_coreference(coref_path)
        
        logger.info(f"  텍스트: {len(text_content):,}자")
        logger.info(f"  이미지 분석: {image_analysis.get('total_images', 0)}개")
        logger.info(f"  보충자료 Excel: {len(supplement_data.get('excel', []))}개 항목")
        logger.info(f"  보충자료 Word: {len(supplement_data.get('word', []))}개 항목")
        logger.info(f"  Coreference 그룹: {len(coreference.get('coreference_groups', {}))}개")
        
        # GPT-4o로 최종 ADMET 추출
        result = self.extract_with_gpt4o(
            text_content, image_analysis, supplement_data, coreference, pmc_id
        )
        
        # 결과 저장
        output_dir = self.base_dir / "final_extracted" / pmc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{pmc_id}_final_admet.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"최종 ADMET 결과 저장: {output_file}")
        
        return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="최종 ADMET 데이터 추출 (통합 버전)")
    parser.add_argument("--pmc_id", required=True, help="PMC ID (예: PMC7066191)")
    parser.add_argument("--base_dir", default="data_test", help="기본 디렉토리 (기본값: data_test)")
    
    args = parser.parse_args()
    
    extractor = FinalADMETExtractor(base_dir=args.base_dir)
    result = extractor.process_pmc(args.pmc_id)
    
    print(f"✅ 최종 ADMET 추출 완료!")
    print(f"  PMC ID: {result.get('pmc_id', 'N/A')}")
    print(f"  레코드 수: {len(result.get('records', []))}")
    print(f"  스키마 버전: {result.get('schema_version', 'N/A')}")

if __name__ == "__main__":
    main()
