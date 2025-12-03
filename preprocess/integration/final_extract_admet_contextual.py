#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual ADMET Extractor with Multi-turn Conversation
- 각 단계의 추출 결과를 누적하여 다음 단계에 전달
- GPT-4o의 multi-turn conversation을 활용하여 맥락 유지
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Union

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContextualADMETExtractor:
    """
    맥락을 유지하면서 ADMET 정보를 추출하는 클래스
    - 각 단계의 추출 결과를 누적
    - Multi-turn conversation으로 정보를 점진적으로 추가
    """
    
    def __init__(self, base_dir: str = "data_test"):
        self.base_dir = Path(base_dir)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        
        # 중간 결과 저장 디렉토리
        self.context_dir = self.base_dir / "extraction_context" 
        self.context_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_with_contextual_accumulation(self, pmc_id: str) -> Dict[str, Any]:
        """
        맥락을 유지하면서 단계별로 정보를 누적하여 추출
        
        단계:
        1. 본문 텍스트 추출 및 초기 화합물 목록 생성
        2. 이미지 분석 결과 추가 (이전 화합물 목록 참조)
        3. 보충자료 Excel/Word 추가 (이전 결과 참조)
        4. 보충자료 PDF 이미지 추가 (이전 결과 참조)
        5. 최종 통합 및 정제
        """
        logger.info(f"맥락 기반 ADMET 추출 시작: {pmc_id}")
        
        context_file = self.context_dir / f"{pmc_id}_context.json"
        
        # 기존 컨텍스트가 있으면 로드, 없으면 초기화
        if context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                context = json.load(f)
            logger.info(f"기존 컨텍스트 로드: {context_file}")
        else:
            context = {
                "pmc_id": pmc_id,
                "created_at": datetime.now().isoformat(),
                "steps": [],
                "accumulated_compounds": {},
                "conversation_history": []
            }
        
        # Step 1: 본문 텍스트에서 초기 화합물 추출
        if not self._step_completed(context, "text_extraction"):
            logger.info("Step 1: 본문 텍스트에서 초기 화합물 추출")
            text_result = self._extract_from_text(pmc_id, context)
            context = self._update_context(context, "text_extraction", text_result)
            self._save_context(context, context_file)
        
        # Step 2: 이미지 분석 결과 추가
        if not self._step_completed(context, "image_analysis"):
            logger.info("Step 2: 이미지 분석 결과 추가")
            image_result = self._add_image_analysis(pmc_id, context)
            context = self._update_context(context, "image_analysis", image_result)
            self._save_context(context, context_file)
        
        # Step 3: 보충자료 Excel/Word 추가
        if not self._step_completed(context, "supplement_excel_word"):
            logger.info("Step 3: 보충자료 Excel/Word 추가")
            supp_result = self._add_supplement_excel_word(pmc_id, context)
            context = self._update_context(context, "supplement_excel_word", supp_result)
            self._save_context(context, context_file)
        
        # Step 4: 보충자료 PDF 이미지 추가
        if not self._step_completed(context, "supplement_pdf_images"):
            logger.info("Step 4: 보충자료 PDF 이미지 추가")
            pdf_result = self._add_supplement_pdf_images(pmc_id, context)
            context = self._update_context(context, "supplement_pdf_images", pdf_result)
            self._save_context(context, context_file)
        
        # Step 5: 최종 통합 및 정제
        if not self._step_completed(context, "final_integration"):
            logger.info("Step 5: 최종 통합 및 정제")
            final_result = self._final_integration(pmc_id, context)
            context = self._update_context(context, "final_integration", final_result)
            self._save_context(context, context_file)
        
        # 최종 결과 저장
        output_dir = self.base_dir / "final_extracted" / pmc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{pmc_id}_final_admet.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"최종 결과 저장: {output_file}")
        return final_result
    
    def _step_completed(self, context: Dict, step_name: str) -> bool:
        """특정 단계가 완료되었는지 확인"""
        return any(step.get("step_name") == step_name and step.get("status") == "completed" 
                   for step in context.get("steps", []))
    
    def _update_context(self, context: Dict, step_name: str, result: Dict) -> Dict:
        """컨텍스트 업데이트"""
        # 단계 기록
        context["steps"].append({
            "step_name": step_name,
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result_summary": {
                "compounds_found": len(result.get("records", [])),
                "total_attributes": sum(len(r.get("attributes", {})) for r in result.get("records", []))
            }
        })
        
        # 누적된 화합물 정보 업데이트
        for record in result.get("records", []):
            comp_name = record.get("compound_name")
            if comp_name:
                if comp_name not in context["accumulated_compounds"]:
                    context["accumulated_compounds"][comp_name] = {
                        "aliases": set(),
                        "attributes": {},
                        "sources": []
                    }
                
                # Aliases 추가
                for alias in record.get("aliases", []):
                    context["accumulated_compounds"][comp_name]["aliases"].add(alias)
                
                # 속성 추가/업데이트
                for attr_name, attr_value in record.get("attributes", {}).items():
                    if attr_name not in context["accumulated_compounds"][comp_name]["attributes"]:
                        context["accumulated_compounds"][comp_name]["attributes"][attr_name] = []
                    context["accumulated_compounds"][comp_name]["attributes"][attr_name].append({
                        "value": attr_value,
                        "source": step_name
                    })
                
                # 출처 추가
                context["accumulated_compounds"][comp_name]["sources"].append(step_name)
        
        # 대화 히스토리 추가
        context["conversation_history"].append({
            "step": step_name,
            "user_message": result.get("prompt_preview", "")[:500],
            "assistant_response": f"Extracted {len(result.get('records', []))} compounds"
        })
        
        return context
    
    def _save_context(self, context: Dict, context_file: Path):
        """컨텍스트 저장 (aliases를 리스트로 변환)"""
        # aliases를 set에서 list로 변환
        for comp_name, comp_data in context["accumulated_compounds"].items():
            if isinstance(comp_data.get("aliases"), set):
                comp_data["aliases"] = list(comp_data["aliases"])
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
    
    def _extract_from_text(self, pmc_id: str, context: Dict) -> Dict[str, Any]:
        """Step 1: 본문 텍스트에서 초기 화합물 추출"""
        text_path = self.base_dir / "text_extracted" / pmc_id / "extracted_text.txt"
        
        if not text_path.exists():
            logger.warning(f"텍스트 파일 없음: {text_path}")
            return {"records": []}
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # ADMET 관련 문장만 필터링
        admet_sentences = self._filter_admet_sentences(text_content)
        
        prompt = f"""You are extracting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) data from a scientific paper.

**Task**: Extract ALL compounds mentioned in the text and their ADMET-related attributes.

**Text Content (ADMET-related sentences only)**:
{admet_sentences[:15000]}

**Instructions**:
1. Identify ALL compound names mentioned in the text
2. For each compound, extract any ADMET-related attributes (values, units, conditions)
3. Return a JSON array with records containing: compound_name, aliases, and attributes (as key-value pairs)

**Output Format** (JSON only, no markdown):
{{
  "records": [
    {{
      "compound_name": "Compound A",
      "aliases": ["A", "Compound-A"],
      "attributes": {{
        "caco2": "15.8 1e-6 cm/s",
        "ppb": "45.2 %",
        "cyp1a2": "5.4 µM (IC50)"
      }}
    }}
  ]
}}

Return ONLY valid JSON."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical data extractor. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16384,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(f"텍스트에서 {len(result.get('records', []))}개 화합물 추출")
            return result
            
        except Exception as e:
            logger.error(f"텍스트 추출 실패: {e}")
            return {"records": []}
    
    def _add_image_analysis(self, pmc_id: str, context: Dict) -> Dict[str, Any]:
        """Step 2: 이미지 분석 결과를 기존 화합물에 추가"""
        img_path = self.base_dir / "graph_analyzed" / pmc_id / "all_analyses.json"
        if not img_path.exists():
            img_path = self.base_dir / "graph_extracted" / pmc_id / "all_analyses.json"
        
        if not img_path.exists():
            logger.warning(f"이미지 분석 파일 없음: {img_path}")
            return {"records": []}
        
        with open(img_path, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
        
        # 기존에 추출된 화합물 목록
        existing_compounds = list(context.get("accumulated_compounds", {}).keys())
        
        prompt = f"""You are adding ADMET data from image analysis to existing compounds.

**Previously Extracted Compounds** (from text):
{json.dumps(existing_compounds, ensure_ascii=False, indent=2)}

**Image Analysis Data**:
{json.dumps(image_data, ensure_ascii=False, indent=2)[:10000]}

**Task**: 
1. Match compounds found in images to existing compounds (by name or alias)
2. Add new compounds if not found in existing list
3. Add/update ADMET attributes from images
4. Return updated records with all compounds (existing + new) and their attributes

**Output Format** (JSON only):
{{
  "records": [
    {{
      "compound_name": "Compound A",
      "aliases": ["A"],
      "attributes": {{
        "caco2": "15.8 1e-6 cm/s",
        "new_attribute_from_image": "value"
      }}
    }}
  ]
}}

Return ONLY valid JSON."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical data extractor. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16384,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(f"이미지 분석에서 {len(result.get('records', []))}개 화합물 추가/업데이트")
            return result
            
        except Exception as e:
            logger.error(f"이미지 분석 추가 실패: {e}")
            return {"records": []}
    
    def _add_supplement_excel_word(self, pmc_id: str, context: Dict) -> Dict[str, Any]:
        """Step 3: 보충자료 Excel/Word 데이터 추가"""
        supp_dir = self.base_dir / "supp_extracted" / pmc_id
        
        excel_data = []
        word_data = []
        
        # Excel 로드
        excel_dir = supp_dir / "excel"
        if excel_dir.exists():
            for json_file in excel_dir.glob("*compounds_from_excel.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    excel_data.extend(json.load(f) if isinstance(json.load(f), list) else [])
        
        # Word 로드
        word_dir = supp_dir / "word"
        if word_dir.exists():
            for json_file in word_dir.glob("*compounds_from_word.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    word_data.extend(json.load(f) if isinstance(json.load(f), list) else [])
        
        if not excel_data and not word_data:
            logger.warning("보충자료 Excel/Word 데이터 없음")
            return {"records": []}
        
        # 기존 화합물 목록
        existing_compounds = list(context.get("accumulated_compounds", {}).keys())
        
        prompt = f"""You are adding ADMET data from supplementary Excel/Word files to existing compounds.

**Previously Extracted Compounds** (from text + images):
{json.dumps(existing_compounds, ensure_ascii=False, indent=2)}

**Supplementary Excel Data**:
{json.dumps(excel_data[:100], ensure_ascii=False, indent=2)}

**Supplementary Word Data**:
{json.dumps(word_data[:100], ensure_ascii=False, indent=2)}

**Task**:
1. Match compounds from supplements to existing compounds
2. Add new compounds if not found
3. Add/update ADMET attributes (supplement data has priority)
4. Return updated records

**Output Format** (JSON only):
{{
  "records": [
    {{
      "compound_name": "Compound A",
      "aliases": ["A"],
      "attributes": {{
        "caco2": "15.8 1e-6 cm/s",
        "new_from_supplement": "value"
      }}
    }}
  ]
}}

Return ONLY valid JSON."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical data extractor. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16384,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(f"보충자료 Excel/Word에서 {len(result.get('records', []))}개 화합물 추가/업데이트")
            return result
            
        except Exception as e:
            logger.error(f"보충자료 Excel/Word 추가 실패: {e}")
            return {"records": []}
    
    def _add_supplement_pdf_images(self, pmc_id: str, context: Dict) -> Dict[str, Any]:
        """Step 4: 보충자료 PDF 이미지 분석 결과 추가"""
        supp_dir = self.base_dir / "supp_extracted" / pmc_id
        pdf_info_dir = supp_dir / "pdf_info"
        
        pdf_image_data = []
        if pdf_info_dir.exists():
            for json_file in pdf_info_dir.glob("*_yolo_gpt_analysis.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    pdf_image_data.append(data)
        
        if not pdf_image_data:
            logger.warning("보충자료 PDF 이미지 데이터 없음")
            return {"records": []}
        
        # 기존 화합물 목록
        existing_compounds = list(context.get("accumulated_compounds", {}).keys())
        
        prompt = f"""You are adding ADMET data from supplementary PDF images to existing compounds.

**Previously Extracted Compounds** (from text + images + supplements):
{json.dumps(existing_compounds, ensure_ascii=False, indent=2)}

**Supplementary PDF Image Analysis Data**:
{json.dumps(pdf_image_data[:5], ensure_ascii=False, indent=2)[:10000]}

**Task**:
1. Match compounds from PDF images to existing compounds
2. Add new compounds if not found
3. Add/update ADMET attributes
4. Return updated records

**Output Format** (JSON only):
{{
  "records": [
    {{
      "compound_name": "Compound A",
      "aliases": ["A"],
      "attributes": {{
        "caco2": "15.8 1e-6 cm/s"
      }}
    }}
  ]
}}

Return ONLY valid JSON."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical data extractor. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16384,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(f"보충자료 PDF 이미지에서 {len(result.get('records', []))}개 화합물 추가/업데이트")
            return result
            
        except Exception as e:
            logger.error(f"보충자료 PDF 이미지 추가 실패: {e}")
            return {"records": []}
    
    def _final_integration(self, pmc_id: str, context: Dict) -> Dict[str, Any]:
        """Step 5: 최종 통합 및 정제 (Coreference 적용, 중복 제거, 정규화)"""
        coref_path = self.base_dir / "entity_analyzed" / pmc_id / "global_coreference.json"
        
        coreference = {}
        if coref_path.exists():
            with open(coref_path, 'r', encoding='utf-8') as f:
                coreference = json.load(f)
        
        # 누적된 모든 화합물 정보
        accumulated = context.get("accumulated_compounds", {})
        
        prompt = f"""You are performing final integration and refinement of ADMET data.

**Accumulated Compounds from All Sources**:
{json.dumps(accumulated, ensure_ascii=False, indent=2, default=str)[:20000]}

**Coreference Dictionary** (for alias merging):
{json.dumps(coreference.get("coreference_groups", {}), ensure_ascii=False, indent=2)[:5000]}

**Task**:
1. Merge compounds with aliases using coreference dictionary
2. Resolve conflicts (priority: supplement > image > text)
3. Normalize units and values
4. Create final structured ADMET records with standard schema
5. Return complete ADMET table

**Output Format** (use the standard ADMET schema with all fields):
{{
  "schema_version": "2.0",
  "created_at": "{datetime.now().isoformat()}",
  "pmc_id": "{pmc_id}",
  "records": [
    {{
      "compound_name": "Canonical Name",
      "aliases": ["alias1", "alias2"],
      "smiles": "SMILES string or null",
      "caco2": {{"value": 15.8, "unit": "1e-6 cm/s", "source": "supplement"}},
      ...
    }}
  ]
}}

Return ONLY valid JSON."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical data extractor. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16384,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(f"최종 통합 완료: {len(result.get('records', []))}개 화합물")
            return result
            
        except Exception as e:
            logger.error(f"최종 통합 실패: {e}")
            return {"records": []}
    
    def _filter_admet_sentences(self, text: str) -> str:
        """ADMET 관련 문장만 필터링"""
        admet_keywords = [
            'absorption', 'distribution', 'metabolism', 'excretion', 'toxicity',
            'caco2', 'mdck', 'pampa', 'permeability', 'lipinski',
            'logd', 'logs', 'pka', 'ppb', 'protein binding',
            'bbb', 'blood-brain', 'vd', 'volume of distribution',
            'cyp', 'cytochrome', 'clearance', 'cl', 'half-life', 't1/2',
            'herg', 'dili', 'ames', 'carcinogenicity', 'mutagenicity'
        ]
        
        sentences = text.split('.')
        admet_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in admet_keywords):
                admet_sentences.append(sentence.strip())
        
        return '. '.join(admet_sentences)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="맥락 기반 ADMET 추출")
    parser.add_argument("--pmc_id", required=True, help="PMC ID")
    parser.add_argument("--base_dir", default="data_test", help="Base directory")
    
    args = parser.parse_args()
    
    extractor = ContextualADMETExtractor(base_dir=args.base_dir)
    result = extractor.extract_with_contextual_accumulation(args.pmc_id)
    
    print(f"✅ 완료: {len(result.get('records', []))}개 화합물 추출")


if __name__ == "__main__":
    main()


