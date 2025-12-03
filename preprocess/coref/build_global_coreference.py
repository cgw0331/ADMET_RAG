#!/usr/bin/env python3
"""
Global Coreference Dictionary Builder
- 보충자료, 이미지 분석, 텍스트 추출 결과를 통합
- Entity 간 동의어/관계 매핑 생성
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import ollama
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GlobalCoreferenceBuilder:
    def __init__(self, model_name="gpt-4o", use_gpt=True):
        self.use_gpt = use_gpt
        self.model_name = model_name
        
        if use_gpt:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=api_key)
            logger.info(f"GPT 모드 사용: {model_name}")
        else:
            self.client = ollama.Client()
            logger.info(f"Ollama 모드 사용: {model_name}")
        
        self.entity_mapping = defaultdict(set)
        self.relationships = []
        
    def load_supplement_data(self, supplement_json_path: Path):
        """보충자료 Excel 추출 결과 로드 - 화합물 이름만 반환"""
        logger.info(f"보충자료 데이터 로딩: {supplement_json_path}")
        
        with open(supplement_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # compound_name 열 추출
        compounds = set()
        if isinstance(data, list):
            for item in data:
                if 'compound_name' in item and item['compound_name']:
                    compounds.add(item['compound_name'])
        elif isinstance(data, dict):
            for item in data.get('results', []):
                if 'compounds' in item:
                    for compound in item['compounds']:
                        name = compound.get('compound_name', '')
                        if name:
                            compounds.add(name)
        
        logger.info(f"  발견된 화합물: {len(compounds)}개")
        return list(compounds)
    
    def load_supplement_with_attributes(self, supplement_json_path: Path) -> Dict[str, Any]:
        """보충자료 Excel 추출 결과 로드 - 화합물-속성 정보 포함"""
        logger.info(f"보충자료 상세 데이터 로딩: {supplement_json_path}")
        
        with open(supplement_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        compounds = set()
        compound_attributes = defaultdict(set)  # compound -> set of attribute names
        compound_samples = defaultdict(dict)  # compound -> {attribute: value}
        
        if isinstance(data, list):
            for item in data:
                comp_name = item.get('compound_name', '')
                attr_name = item.get('attribute_name', '')
                value = item.get('value', '')
                
                if comp_name:
                    compounds.add(comp_name)
                    if attr_name:
                        compound_attributes[comp_name].add(attr_name)
                        # 샘플 값 저장 (최대 3개 속성만)
                        if len(compound_samples[comp_name]) < 3:
                            compound_samples[comp_name][attr_name] = str(value)[:100]  # 값은 100자로 제한
        
        return {
            "compounds": list(compounds),
            "compound_attributes": {k: list(v) for k, v in compound_attributes.items()},
            "sample_values": {k: v for k, v in compound_samples.items()}
        }
    
    def load_image_analysis(self, image_analysis_json_path: Path):
        """이미지 분석 결과 로드 - 다양한 구조 지원"""
        logger.info(f"이미지 분석 데이터 로딩: {image_analysis_json_path}")
        
        with open(image_analysis_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        compounds = set()
        if isinstance(data, list):
            for item in data:
                # analysis 필드가 있는 경우 (구버전)
                analysis = item.get('analysis', item)  # analysis가 없으면 item 자체가 analysis
                
                # compound_name (singular) 처리
                if 'compound_name' in analysis:
                    name = analysis['compound_name']
                    if isinstance(name, list):
                        compounds.update(name)
                    elif name and name != "":
                        compounds.add(name)
                
                # compound_names (plural) 처리
                if 'compound_names' in analysis:
                    names = analysis['compound_names']
                    if isinstance(names, list):
                        compounds.update(names)
                    elif names and names != "":
                        compounds.add(names)
                
                # compounds 배열 처리 (table 분석 결과)
                if 'compounds' in analysis and isinstance(analysis['compounds'], list):
                    for comp in analysis['compounds']:
                        if isinstance(comp, dict):
                            # compound_name 필드 찾기
                            if 'compound_name' in comp:
                                name = comp['compound_name']
                                if name and name != "":
                                    compounds.add(name)
                            # name 필드도 체크
                            elif 'name' in comp:
                                name = comp['name']
                                if name and name != "":
                                    compounds.add(name)
                
                # GPT 분석 결과의 자유로운 구조에서 화합물 찾기
                # analysis가 dict인 경우, 재귀적으로 화합물 이름 패턴 찾기
                if isinstance(analysis, dict):
                    self._extract_compounds_recursive(analysis, compounds)
        
        logger.info(f"  발견된 화합물: {len(compounds)}개")
        return list(compounds)
    
    def _extract_compounds_recursive(self, obj, compounds: set, depth=0):
        """재귀적으로 화합물 이름 추출 (GPT의 자유로운 구조 지원)"""
        if depth > 5:  # 깊이 제한
            return
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                # 화합물 관련 키워드 찾기
                if any(kw in key.lower() for kw in ['compound', 'drug', 'molecule', 'chemical']):
                    if isinstance(value, str) and value:
                        compounds.add(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, str) and v:
                                compounds.add(v)
                            elif isinstance(v, dict):
                                self._extract_compounds_recursive(v, compounds, depth+1)
                
                # 재귀 탐색
                self._extract_compounds_recursive(value, compounds, depth+1)
        
        elif isinstance(obj, list):
            for item in obj:
                self._extract_compounds_recursive(item, compounds, depth+1)
    
    def load_image_analysis_full(self, image_analysis_json_path: Path) -> Dict[str, Any]:
        """이미지 분석 결과 전체 로드 (상세 내용 포함)"""
        try:
            with open(image_analysis_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"image_analyses": data, "source": str(image_analysis_json_path)}
        except Exception as e:
            logger.error(f"이미지 분석 전체 로드 실패: {e}")
            return {}
    
    def summarize_image_analysis(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """이미지 분석 결과 요약 생성 (GPT/Llama 모두 지원)"""
        if not image_data or 'image_analyses' not in image_data:
            return {}
        
        analyses = image_data.get('image_analyses', [])
        if not analyses:
            return {}
        
        # 화합물 추출 (재귀적 방식 사용)
        compounds_in_images = set()
        admet_findings = defaultdict(dict)
        compound_sources = defaultdict(list)  # 화합물 → [이미지 파일명]
        
        for item in analyses:
            analysis = item.get('analysis', {})
            filename = item.get('filename', 'unknown')
            
            # 1. 구버전 Llama 구조 (compound_name, compound_names)
            if 'compound_name' in analysis:
                name = analysis['compound_name']
                if isinstance(name, list):
                    for n in name:
                        if n:
                            compounds_in_images.add(n)
                            compound_sources[n].append(filename)
                elif name:
                    compounds_in_images.add(name)
                    compound_sources[name].append(filename)
            
            if 'compound_names' in analysis:
                names = analysis['compound_names']
                if isinstance(names, list):
                    for n in names:
                        if n:
                            compounds_in_images.add(n)
                            compound_sources[n].append(filename)
                elif names:
                    compounds_in_images.add(names)
                    compound_sources[names].append(filename)
            
            # 2. GPT 구조화된 결과 처리 (analysis.figure.panels 등)
            if isinstance(analysis, dict):
                # GPT 결과에서 화합물 찾기
                found_in_this_analysis = set()
                self._extract_compounds_from_gpt_analysis(analysis, found_in_this_analysis)
                
                for comp in found_in_this_analysis:
                    compounds_in_images.add(comp)
                    compound_sources[comp].append(filename)
            
            # 3. ADMET 지표 추출 (구버전)
            admet_indicators = analysis.get('admet_indicators', {})
            for ind_name, ind_data in admet_indicators.items():
                if isinstance(ind_data, dict) and ind_data.get('found', False):
                    for comp in compounds_in_images:
                        if ind_name not in admet_findings[comp]:
                            admet_findings[comp][ind_name] = {
                                "found": True,
                                "description": ind_data.get('description', '')[:200],
                                "source_image": filename
                            }
            
            # 4. GPT 결과에서 ADMET 정보 찾기 (재귀적)
            self._extract_admet_from_gpt_analysis(analysis, compounds_in_images, admet_findings, filename)
        
        # 전체 요약
        return {
            "total_figures": len(analyses),
            "compounds_in_images": list(compounds_in_images),
            "compound_sources": {k: list(set(v)) for k, v in compound_sources.items()},  # 출처 추가
            "admet_findings": {k: v for k, v in admet_findings.items()},
            "sample_analyses": analyses[:3]  # 처음 3개만 샘플로
        }
    
    def _extract_compounds_from_gpt_analysis(self, analysis: Dict, compounds: set, depth=0):
        """GPT 분석 결과에서 화합물 추출 (구조화된 데이터 지원)"""
        if depth > 5:
            return
        
        if isinstance(analysis, dict):
            # GPT는 figure.panels[].title, summary 등에 화합물 이름이 있을 수 있음
            for key, value in analysis.items():
                # title, summary, compound 등 키워드 확인
                if any(kw in key.lower() for kw in ['title', 'summary', 'compound', 'drug', 'molecule', 'label']):
                    if isinstance(value, str) and 2 < len(value) < 100:
                        # 짧은 문자열이면 화합물일 가능성
                        compounds.add(value)
                
                # 재귀 탐색
                self._extract_compounds_from_gpt_analysis(value, compounds, depth+1)
        elif isinstance(analysis, list):
            for item in analysis:
                self._extract_compounds_from_gpt_analysis(item, compounds, depth+1)
    
    def _extract_admet_from_gpt_analysis(self, analysis: Dict, compounds: set, 
                                         admet_findings: Dict, filename: str, depth=0):
        """GPT 분석 결과에서 ADMET 정보 추출"""
        if depth > 5:
            return
        
        if isinstance(analysis, dict):
            # ADMET 관련 키워드 찾기
            admet_keywords = ['caco', 'ppb', 'bbb', 'cyp', 'herg', 'dili', 'clearance', 
                            'half-life', 'permeability', 'protein binding', 'toxicity']
            
            for key, value in analysis.items():
                key_lower = key.lower()
                if any(kw in key_lower for kw in admet_keywords):
                    # ADMET 지표 발견
                    desc = str(value)[:200] if value else ""
                    for comp in compounds:
                        if key_lower not in str(admet_findings.get(comp, {})):
                            if comp not in admet_findings:
                                admet_findings[comp] = {}
                            admet_findings[comp][key_lower] = {
                                "found": True,
                                "description": desc,
                                "source_image": filename
                            }
                
                # 재귀 탐색
                self._extract_admet_from_gpt_analysis(value, compounds, admet_findings, filename, depth+1)
        elif isinstance(analysis, list):
            for item in analysis:
                self._extract_admet_from_gpt_analysis(item, compounds, admet_findings, filename, depth+1)
    
    def load_text_data(self, text_path: Path):
        """원본 텍스트 로드 (일부만)"""
        logger.info(f"텍스트 데이터 로딩: {text_path}")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 처음 5000자만 사용
        return text[:5000]
    
    def extract_text_contexts(self, text_path: Path, compounds: List[str], 
                             max_chunks: int = 10, chunk_size: int = 2000) -> List[Dict[str, str]]:
        """텍스트에서 화합물 등장 섹션 선별 추출"""
        logger.info(f"텍스트 맥락 선별 추출: {text_path}")
        
        if not text_path.exists():
            return []
        
        with open(text_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # 섹션 분리 (간단한 휴리스틱: 제목 패턴)
        sections = []
        current_section = {"name": "Introduction", "text": ""}
        
        section_keywords = ["abstract", "introduction", "methods", "results", "discussion", 
                           "conclusion", "materials and methods", "experimental"]
        
        lines = full_text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            # 섹션 제목 탐지 (대문자 시작, 짧은 라인)
            if len(line.strip()) < 100 and any(kw in line_lower for kw in section_keywords):
                if current_section["text"]:
                    sections.append(current_section)
                current_section = {"name": line.strip()[:50], "text": ""}
            else:
                current_section["text"] += line + "\n"
        
        if current_section["text"]:
            sections.append(current_section)
        
        # 화합물 등장하는 섹션만 선별
        relevant_chunks = []
        for section in sections:
            section_text = section["text"].lower()
            mentioned_compounds = [c for c in compounds if c.lower() in section_text]
            
            if mentioned_compounds:
                # 섹션을 chunk_size로 나누기
                text = section["text"]
                for i in range(0, len(text), chunk_size):
                    chunk_text = text[i:i+chunk_size]
                    chunk_lower = chunk_text.lower()
                    chunk_compounds = [c for c in compounds if c.lower() in chunk_lower]
                    
                    if chunk_compounds:
                        relevant_chunks.append({
                            "section": section["name"],
                            "text": chunk_text[:chunk_size],
                            "mentioned_entities": chunk_compounds[:5]  # 최대 5개만
                        })
                        if len(relevant_chunks) >= max_chunks:
                            break
            
            if len(relevant_chunks) >= max_chunks:
                break
        
        # Abstract와 첫 몇 개 섹션도 포함 (화합물 없어도)
        if not relevant_chunks and sections:
            for section in sections[:3]:
                if section["text"]:
                    relevant_chunks.append({
                        "section": section["name"],
                        "text": section["text"][:chunk_size],
                        "mentioned_entities": []
                    })
                if len(relevant_chunks) >= 3:
                    break
        
        logger.info(f"  선별된 텍스트 청크: {len(relevant_chunks)}개")
        return relevant_chunks
    
    def build_coreference_with_llama(self, all_compounds: List[str], 
                                     text_contexts: List[Dict[str, str]],
                                     supplement_summary: Optional[Dict] = None, 
                                     image_summary: Optional[Dict] = None) -> Dict[str, Any]:
        """LLM을 사용해 Coreference dictionary 생성 (Llama 또는 GPT)"""
        model_type = "GPT" if self.use_gpt else "Llama"
        logger.info(f"{model_type}로 coreference 생성 중... ({len(all_compounds)}개 엔티티)")
        
        # 보충자료 정보 요약 (길이 제한 확대: 2000 → 4000)
        supplement_str = ""
        if supplement_summary:
            supp_data = {
                "total_compounds": len(supplement_summary.get('compounds', [])),
                "compound_attributes": supplement_summary.get('compound_attributes', {}),
                "sample_values": supplement_summary.get('sample_values', {})
            }
            supplement_str = f"""
**Supplementary Material Data:**
{json.dumps(supp_data, ensure_ascii=False, indent=2)[:4000]}
"""
        
        # 이미지 분석 정보 요약 (길이 제한 확대: 2000 → 4000)
        image_str = ""
        if image_summary:
            image_str = f"""
**Image Analysis Summary:**
Total figures analyzed: {image_summary.get('total_figures', 0)}
Compounds found in images: {', '.join(image_summary.get('compounds_in_images', [])[:20])}
Compound sources (which image): {json.dumps(image_summary.get('compound_sources', {}), ensure_ascii=False, indent=2)[:2000]}
ADMET findings: {json.dumps(image_summary.get('admet_findings', {}), ensure_ascii=False, indent=2)[:2000]}
Sample analyses (first 2): {json.dumps(image_summary.get('sample_analyses', [])[:2], ensure_ascii=False, indent=2)[:1000]}
"""
        
        # 텍스트 맥락 (청크 크기 확대: 800 → 1500, 개수 확대: 5 → 10)
        text_str = ""
        if text_contexts:
            text_str = "**Relevant Text Contexts (where compounds are mentioned):**\n"
            for i, ctx in enumerate(text_contexts[:10], 1):  # 최대 10개로 확대
                text_str += f"""
[{i}] Section: {ctx.get('section', 'Unknown')}
Entities mentioned: {', '.join(ctx.get('mentioned_entities', []))}
Text excerpt: {ctx.get('text', '')[:1500]}...
"""
        
        prompt = f"""You are analyzing multiple data sources from a scientific article to build a comprehensive coreference dictionary.

**ALL Entities Found (compounds, proteins, genes, cell lines, etc.):**
{json.dumps(all_compounds, ensure_ascii=False, indent=2)}
{supplement_str}
{image_str}
{text_str}

**TASK:**
Create a comprehensive coreference dictionary that maps entity names (compounds, proteins, genes, cell lines, etc.) to their aliases, abbreviations, and related terms based on ALL sources above.

**Required Output Format (JSON only):**
{{
  "coreference_groups": {{
    "canonical_name_1": {{
      "aliases": ["alias1", "alias2", "abbreviation1"],
      "entity_type": "compound/protein/gene/cell_line/etc",
      "attributes": ["Caco2", "PPB", ...]  // if compound, list ADMET attributes found
    }},
    "canonical_name_2": {{
      "aliases": ["alias3", "alias4"],
      "entity_type": "protein",
      "attributes": []
    }}
  }},
  "relationships": [
    {{
      "entity1": "name1",
      "entity2": "name2", 
      "relation": "inhibits/binds_to/tested_in/synonym/part_of/related_to",
      "evidence": "brief explanation from text",
      "source": "text/supplement/image"
    }}
  ]
}}

**Instructions:**
1. **Entity Types**: Identify ALL entity types (compounds, proteins like "FPGS", "DHFR", genes like "SLC19A1", cell lines, organoids, etc.)
2. **Coreference Groups**: Group aliases/abbreviations that refer to the same entity
3. **Attributes**: For compounds, include ADMET attributes found (Caco2, PPB, BBB, CYP isoforms, etc.)
4. **Relationships**: Extract relationships between entities:
   - Compound-Protein: "inhibits", "binds_to", "targets"
   - Compound-Cell: "tested_in", "effective_in"
   - Compound-Compound: "synonym", "derivative_of"
   - Other: "part_of", "regulates", etc.
5. **Evidence**: Always cite which source (text excerpt, supplement, or image) provides the evidence
6. Be comprehensive - include all variations, abbreviations, and related terms you find

**Examples:**
- If you see "HLO" and "human liver organoid" → same entity (cell_line type)
- If you see "C1" and "novel antifolate" in the same context → same compound
- If supplement shows "Methotrexate" has "Caco2" attribute → include "Caco2" in attributes
- If text says "C1 inhibits FPGS" → relationship: entity1="C1", entity2="FPGS", relation="inhibits"

Output JSON only:
"""
        
        try:
            if self.use_gpt:
                # GPT 사용
                response = self.client.chat.completions.create(
                    model=self.model_name if self.model_name.startswith("gpt") else "gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=8000
                )
                generated_text = response.choices[0].message.content
                logger.info(f"GPT 응답 받음 (길이: {len(generated_text)}자)")
            else:
                # Ollama 사용
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1, "num_predict": 16384}
                )
                generated_text = response.get('message', {}).get('content', '')
                logger.info(f"Llama 응답 받음 (길이: {len(generated_text)}자)")
            
            # JSON 파싱
            if "```json" in generated_text:
                json_start = generated_text.find("```json") + 7
                json_end = generated_text.find("```", json_start)
                json_text = generated_text[json_start:json_end].strip()
            elif "```" in generated_text:
                # 다른 코드 블록 제거
                json_start = generated_text.find("```") + 3
                json_end = generated_text.find("```", json_start)
                json_text = generated_text[json_start:json_end].strip()
            else:
                json_text = generated_text.strip()
            
            result = json.loads(json_text)
            return result
            
        except Exception as e:
            model_type = "GPT" if self.use_gpt else "Llama"
            logger.error(f"{model_type} coreference 생성 실패: {e}")
            return {"coreference_groups": {}, "relationships": []}
    
    def load_global_coreference(self, suffix="", base_dir: Path = Path("data_test")) -> Dict[str, Any]:
        """전역 coreference 로드"""
        global_file = base_dir / "entity_analyzed" / f"global_coreference_accumulated{suffix}.json"
        
        if global_file.exists():
            logger.info(f"전역 coreference 로드: {global_file}")
            with open(global_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.info(f"전역 coreference 없음, 새로 생성: {global_file}")
            return {"coreference_groups": {}, "relationships": []}
    
    def merge_coreference(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """기존과 새로운 coreference 병합 (개선된 구조 지원)"""
        merged_groups = existing.get("coreference_groups", {}).copy()
        
        # 새로운 그룹 병합
        for canonical, group_data in new.get("coreference_groups", {}).items():
            # 하위 호환: group_data가 리스트면 (구버전), dict면 (신버전)
            if isinstance(group_data, list):
                # 구버전: aliases 리스트만
                if canonical in merged_groups:
                    if isinstance(merged_groups[canonical], list):
                        existing_aliases = set(merged_groups[canonical])
                        new_aliases = set(group_data)
                        merged_groups[canonical] = list(existing_aliases | new_aliases)
                    else:
                        # 신버전과 구버전 혼합: aliases만 업데이트
                        existing_aliases = set(merged_groups[canonical].get("aliases", []))
                        new_aliases = set(group_data)
                        merged_groups[canonical]["aliases"] = list(existing_aliases | new_aliases)
                else:
                    merged_groups[canonical] = group_data
            else:
                # 신버전: {aliases, entity_type, attributes}
                if canonical in merged_groups:
                    if isinstance(merged_groups[canonical], dict):
                        # 기존 aliases와 새로운 aliases 합침
                        existing_aliases = set(merged_groups[canonical].get("aliases", []))
                        new_aliases = set(group_data.get("aliases", []))
                        merged_groups[canonical]["aliases"] = list(existing_aliases | new_aliases)
                        
                        # entity_type 업데이트 (새로운 것이 있으면)
                        if "entity_type" in group_data:
                            existing_type = merged_groups[canonical].get("entity_type", "")
                            if not existing_type or existing_type == "unknown":
                                merged_groups[canonical]["entity_type"] = group_data["entity_type"]
                        
                        # attributes 병합
                        existing_attrs = set(merged_groups[canonical].get("attributes", []))
                        new_attrs = set(group_data.get("attributes", []))
                        merged_groups[canonical]["attributes"] = list(existing_attrs | new_attrs)
                    else:
                        # 구버전에서 신버전으로 변환
                        existing_aliases = set(merged_groups[canonical] if isinstance(merged_groups[canonical], list) else [])
                        new_aliases = set(group_data.get("aliases", []))
                        merged_groups[canonical] = {
                            "aliases": list(existing_aliases | new_aliases),
                            "entity_type": group_data.get("entity_type", "unknown"),
                            "attributes": group_data.get("attributes", [])
                        }
                else:
                    merged_groups[canonical] = group_data
        
        # Relationships 병합 (중복 제거)
        existing_rels = existing.get("relationships", [])
        new_rels = new.get("relationships", [])
        
        # entity1, entity2, relation 기준으로 중복 제거
        existing_keys = set()
        for rel in existing_rels:
            key = (rel.get("entity1"), rel.get("entity2"), rel.get("relation"))
            existing_keys.add(key)
        
        merged_rels = existing_rels.copy()
        for rel in new_rels:
            key = (rel.get("entity1"), rel.get("entity2"), rel.get("relation"))
            if key not in existing_keys:
                merged_rels.append(rel)
                existing_keys.add(key)
        
        return {
            "coreference_groups": merged_groups,
            "relationships": merged_rels
        }
    
    def save_global_coreference(self, data: Dict[str, Any], suffix="", base_dir: Path = Path("data_test")):
        """전역 coreference 저장"""
        output_dir = base_dir / "entity_analyzed"
        output_dir.mkdir(parents=True, exist_ok=True)
        global_file = output_dir / f"global_coreference_accumulated{suffix}.json"
        
        with open(global_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"전역 coreference 저장: {global_file}")
    
    def find_supplement_files(self, pmc_id: str, base_dir: Path = Path("data_test")) -> List[Path]:
        """보충자료 파일들을 적응적으로 찾기"""
        found_files = []
        
        # 경로 1: data_test/supp_extracted/PMC###/excel/, word/ (신버전)
        supp_base = base_dir / "supp_extracted" / pmc_id
        if supp_base.exists():
            excel_dir = supp_base / "excel"
            if excel_dir.exists():
                for json_file in excel_dir.glob("*compounds_from_excel.json"):
                    found_files.append(json_file)
                for csv_file in excel_dir.glob("*compounds_from_excel.csv"):
                    found_files.append(csv_file)
            
            word_dir = supp_base / "word"
            if word_dir.exists():
                for json_file in word_dir.glob("*compounds_from_word.json"):
                    found_files.append(json_file)
                for csv_file in word_dir.glob("*compounds_from_word.csv"):
                    found_files.append(csv_file)
        
        # 경로 2: supp_extracted/ 또는 supplement_extracted/ (구버전 경로, 호환성)
        supp_base_old1 = Path("supp_extracted") / pmc_id
        supp_base_old2 = Path("supplement_extracted") / pmc_id
        for supp_base_old in [supp_base_old1, supp_base_old2]:
            if supp_base_old.exists():
                for json_file in supp_base_old.glob("*compounds_from_excel.json"):
                    found_files.append(json_file)
                for csv_file in supp_base_old.glob("*compounds_from_excel.csv"):
                    found_files.append(csv_file)
                for json_file in supp_base_old.glob("*compounds_from_word.json"):
                    found_files.append(json_file)
                for csv_file in supp_base_old.glob("*compounds_from_word.csv"):
                    found_files.append(csv_file)
        
        return found_files
    
    def load_supplement_from_csv(self, csv_path: Path):
        """CSV 파일에서 화합물 이름 추출"""
        try:
            df = pd.read_csv(csv_path)
            compounds = set()
            if 'compound_name' in df.columns:
                compounds.update(df['compound_name'].dropna().astype(str).tolist())
            logger.info(f"  CSV에서 발견된 화합물: {len(compounds)}개")
            return list(compounds)
        except Exception as e:
            logger.error(f"CSV 로드 실패 {csv_path}: {e}")
            return []
    
    def load_supplement_from_csv_with_attributes(self, csv_path: Path) -> Dict[str, Any]:
        """CSV 파일에서 화합물-속성 정보 포함하여 로드"""
        try:
            df = pd.read_csv(csv_path)
            compounds = set()
            compound_attributes = defaultdict(set)
            compound_samples = defaultdict(dict)
            
            if 'compound_name' in df.columns:
                for _, row in df.iterrows():
                    comp_name = str(row.get('compound_name', '')).strip()
                    attr_name = str(row.get('attribute_name', '')).strip()
                    value = str(row.get('value', '')).strip()
                    
                    if comp_name and comp_name.lower() not in {'nan', 'na', ''}:
                        compounds.add(comp_name)
                        if attr_name and attr_name.lower() not in {'nan', 'na', ''}:
                            compound_attributes[comp_name].add(attr_name)
                            if len(compound_samples[comp_name]) < 3:
                                compound_samples[comp_name][attr_name] = value[:100]
            
            logger.info(f"  CSV에서 발견된 화합물: {len(compounds)}개")
            return {
                "compounds": list(compounds),
                "compound_attributes": {k: list(v) for k, v in compound_attributes.items()},
                "sample_values": {k: v for k, v in compound_samples.items()}
            }
        except Exception as e:
            logger.error(f"CSV 상세 로드 실패 {csv_path}: {e}")
            return {"compounds": [], "compound_attributes": {}, "sample_values": {}}
    
    def build_for_pmc(self, pmc_id: str) -> Dict[str, Any]:
        """특정 PMC ID에 대한 통합 coreference 생성 (개선 버전)"""
        logger.info(f"통합 coreference 생성: {pmc_id}")
        
        all_entities = []  # 화합물 + 다른 엔티티들
        supplement_summary = {}
        image_summary = {}
        text_contexts = []
        
        # 경로 설정 (data_test 기준)
        base_dir = Path(os.getenv("BASE_DIR", "data_test"))
        
        # 1. 보충자료 로드 (속성 정보 포함)
        supplement_files = self.find_supplement_files(pmc_id, base_dir=base_dir)
        if supplement_files:
            logger.info(f"보충자료 파일 {len(supplement_files)}개 발견")
            all_supp_compounds = []
            all_compound_attrs = defaultdict(set)
            all_sample_values = defaultdict(dict)
            
            for supp_file in supplement_files:
                if supp_file.suffix == ".csv":
                    # CSV에서 속성 정보 포함 로드
                    supp_data = self.load_supplement_from_csv_with_attributes(supp_file)
                    all_supp_compounds.extend(supp_data.get('compounds', []))
                    for comp, attrs in supp_data.get('compound_attributes', {}).items():
                        all_compound_attrs[comp].update(attrs)
                    for comp, samples in supp_data.get('sample_values', {}).items():
                        all_sample_values[comp].update(samples)
                    
                    # 단순 화합물 이름도 추가 (하위 호환)
                    compounds = self.load_supplement_from_csv(supp_file)
                    all_entities.extend(compounds)
                else:
                    # JSON에서 속성 정보 포함 로드
                    try:
                        supp_data = self.load_supplement_with_attributes(supp_file)
                        all_supp_compounds.extend(supp_data.get('compounds', []))
                        for comp, attrs in supp_data.get('compound_attributes', {}).items():
                            all_compound_attrs[comp].update(attrs)
                        for comp, samples in supp_data.get('sample_values', {}).items():
                            all_sample_values[comp].update(samples)
                        
                        # 단순 화합물 이름도 추가
                        compounds = self.load_supplement_data(supp_file)
                        all_entities.extend(compounds)
                    except Exception as e:
                        logger.warning(f"보충자료 로드 실패 {supp_file}: {e}")
            
            supplement_summary = {
                "compounds": list(set(all_supp_compounds)),
                "compound_attributes": {k: list(v) for k, v in all_compound_attrs.items()},
                "sample_values": {k: v for k, v in all_sample_values.items()},
                "source": f"supp_extracted/{pmc_id}"
            }
            all_entities.extend(supplement_summary['compounds'])
        else:
            logger.warning(f"보충자료 파일 없음: supp_extracted/{pmc_id}")
        
        # 2. 이미지 분석 로드 및 요약
        # 여러 경로 시도: graph_analyzed (GPT 분석 결과), graph_extracted (Llama 분석 결과)
        image_analysis_paths = [
            base_dir / "graph_analyzed" / pmc_id / "all_analyses.json",  # GPT 분석 결과
            base_dir / "graph_analyzed" / pmc_id / "figure_analyses.json",  # GPT 분석 결과 (구버전)
            base_dir / "graph_extracted" / pmc_id / "all_analyses.json",  # graph_extracted
            base_dir / "supp_extracted" / pmc_id / "pdf_graph_info_gpt" / "all_analyses.json",  # 보충자료 GPT 분석 (구버전)
            # 구버전 경로 (호환성)
            Path("graph_analyzed") / pmc_id / "all_analyses.json",
            Path("graph_extracted") / pmc_id / "all_analyses.json",
            Path("supp_extracted") / pmc_id / "pdf_graph_info_gpt" / "all_analyses.json",
        ]
        image_found = False
        for image_path in image_analysis_paths:
            if image_path.exists():
                try:
                    # 화합물 이름 추출
                    image_compounds = self.load_image_analysis(image_path)
                    all_entities.extend(image_compounds)
                    
                    # 전체 이미지 분석 데이터 로드
                    image_data = self.load_image_analysis_full(image_path)
                    
                    # 요약 생성
                    image_summary = self.summarize_image_analysis(image_data)
                    image_summary['source'] = str(image_path)
                    
                    image_found = True
                    logger.info(f"이미지 분석 데이터 로드 성공: {image_path}")
                    break
                except Exception as e:
                    logger.warning(f"이미지 분석 로드 실패 {image_path}: {e}")
        if not image_found:
            logger.warning(f"이미지 분석 파일 없음 (시도 경로: {[str(p) for p in image_analysis_paths]})")
        
        # 3. 텍스트 로드 및 선별
        text_path = Path("text_extracted") / pmc_id / "extracted_text.txt"
        if text_path.exists():
            # 화합물 등장 섹션 선별 추출
            text_contexts = self.extract_text_contexts(text_path, all_entities, max_chunks=10)
        else:
            logger.warning(f"텍스트 파일 없음: {text_path}")
        
        # 중복 제거
        all_entities = list(set(all_entities))
        logger.info(f"총 고유 엔티티: {len(all_entities)}개")
        
        # Llama로 coreference 생성 (개선된 데이터 구조 전달)
        coreference_result = self.build_coreference_with_llama(
            all_entities, text_contexts,
            supplement_summary=supplement_summary if supplement_summary else None,
            image_summary=image_summary if image_summary else None
        )
        
        # 개별 PMC 결과 저장 (Llama/GPT 구분)
        # data_test 기준으로 entity_analyzed/ 사용
        base_dir = Path(os.getenv("BASE_DIR", "data_test"))
        output_dir = base_dir / "entity_analyzed" / pmc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명에 모델 타입 구분 (GPT는 기본이므로 suffix 없음, Llama만 _llama suffix)
        suffix = "_llama" if not self.use_gpt else ""
        output_file = output_dir / f"global_coreference{suffix}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coreference_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"개별 PMC coreference 저장: {output_file}")
        
        # 전역 coreference 업데이트 (모델별로 분리)
        global_suffix = "_llama" if not self.use_gpt else ""
        existing_global = self.load_global_coreference(suffix=global_suffix, base_dir=base_dir)
        merged_global = self.merge_coreference(existing_global, coreference_result)
        self.save_global_coreference(merged_global, suffix=global_suffix, base_dir=base_dir)
        
        return coreference_result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="통합 Coreference Dictionary 생성")
    parser.add_argument("--pmc_id", required=True, help="PMC ID (예: PMC7878295)")
    parser.add_argument("--model", default="gpt-4o", help="모델 이름 (기본값: gpt-4o)")
    parser.add_argument("--use-gpt", action="store_true", default=True, help="GPT-4o 사용 (기본값: True)")
    parser.add_argument("--use-llama", action="store_true", help="Ollama 사용 (GPT 대신)")
    
    args = parser.parse_args()
    
    # --use-llama이 명시되면 Ollama 사용, 아니면 GPT 사용 (기본값)
    use_gpt = not args.use_llama if args.use_llama else args.use_gpt
    
    builder = GlobalCoreferenceBuilder(model_name=args.model, use_gpt=use_gpt)
    result = builder.build_for_pmc(args.pmc_id)
    
    model_type = "Llama" if not use_gpt else "GPT"
    print(f"✅ {model_type}로 통합 coreference 생성 완료!")
    print(f"  그룹 수: {len(result.get('coreference_groups', {}))}")
    print(f"  관계 수: {len(result.get('relationships', []))}")

if __name__ == "__main__":
    main()


