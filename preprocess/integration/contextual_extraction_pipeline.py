#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual Extraction Pipeline
- 각 단계에서 추출한 정보를 누적하여 다음 단계에 전달
- 정보 추출 시 이전 단계의 맥락을 참조하여 더 정확하게 추출
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContextualExtractionPipeline:
    """
    맥락을 유지하면서 단계별로 정보를 추출하는 파이프라인
    """
    
    def __init__(self, base_dir: str = "data_test"):
        self.base_dir = Path(base_dir)
        self.context_dir = self.base_dir / "extraction_context"
        self.context_dir.mkdir(parents=True, exist_ok=True)
    
    def get_accumulated_context(self, pmc_id: str) -> Dict[str, Any]:
        """누적된 컨텍스트 로드"""
        context_file = self.context_dir / f"{pmc_id}_accumulated.json"
        
        if context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 초기 컨텍스트
        return {
            "pmc_id": pmc_id,
            "created_at": datetime.now().isoformat(),
            "compounds": {},  # compound_name -> {aliases: set, attributes: dict, sources: list}
            "step_history": []
        }
    
    def save_accumulated_context(self, pmc_id: str, context: Dict[str, Any]):
        """누적된 컨텍스트 저장 (aliases를 list로 변환)"""
        context_file = self.context_dir / f"{pmc_id}_accumulated.json"
        
        # aliases를 set에서 list로 변환
        for comp_name, comp_data in context["compounds"].items():
            if isinstance(comp_data.get("aliases"), set):
                comp_data["aliases"] = list(comp_data["aliases"])
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
    
    def update_context_from_extraction(self, context: Dict, step_name: str, 
                                       extracted_data: List[Dict[str, Any]]):
        """추출 결과를 컨텍스트에 누적"""
        for record in extracted_data:
            comp_name = record.get("compound_name", "").strip()
            if not comp_name:
                continue
            
            # 화합물이 없으면 추가
            if comp_name not in context["compounds"]:
                context["compounds"][comp_name] = {
                    "aliases": set(),
                    "attributes": defaultdict(list),  # attribute_name -> [values]
                    "sources": []
                }
            
            # Aliases 추가
            for alias in record.get("aliases", []):
                if alias and alias != comp_name:
                    context["compounds"][comp_name]["aliases"].add(alias)
            
            # 속성 추가
            attributes = record.get("attributes", {})
            if isinstance(attributes, dict):
                for attr_name, attr_value in attributes.items():
                    if attr_value:
                        context["compounds"][comp_name]["attributes"][attr_name].append({
                            "value": attr_value,
                            "source": step_name
                        })
            
            # 출처 추가
            if step_name not in context["compounds"][comp_name]["sources"]:
                context["compounds"][comp_name]["sources"].append(step_name)
        
        # 단계 기록
        context["step_history"].append({
            "step": step_name,
            "compounds_found": len([r for r in extracted_data if r.get("compound_name")]),
            "completed_at": datetime.now().isoformat()
        })
    
    def format_context_for_prompt(self, context: Dict) -> str:
        """프롬프트에 포함할 컨텍스트 포맷팅"""
        compounds = context.get("compounds", {})
        
        if not compounds:
            return "**Previously Extracted Compounds:** None (this is the first extraction step)."
        
        # 화합물별로 요약
        summary_lines = []
        summary_lines.append(f"**Previously Extracted Compounds ({len(compounds)} total):**")
        summary_lines.append("")
        
        # 각 화합물의 정보 요약
        for comp_name, comp_data in list(compounds.items())[:20]:  # 최대 20개만
            aliases = list(comp_data.get("aliases", set()))[:5]  # 최대 5개 별칭
            attributes = comp_data.get("attributes", {})
            sources = comp_data.get("sources", [])
            
            summary_lines.append(f"- **{comp_name}**")
            if aliases:
                summary_lines.append(f"  - Aliases: {', '.join(aliases)}")
            if attributes:
                attr_summary = []
                for attr_name, attr_values in list(attributes.items())[:5]:  # 최대 5개 속성
                    if attr_values:
                        latest_value = attr_values[-1].get("value", "")
                        attr_summary.append(f"{attr_name}={latest_value}")
                if attr_summary:
                    summary_lines.append(f"  - Attributes: {', '.join(attr_summary)}")
            if sources:
                summary_lines.append(f"  - Found in: {', '.join(sources)}")
            summary_lines.append("")
        
        if len(compounds) > 20:
            summary_lines.append(f"... and {len(compounds) - 20} more compounds")
        
        return "\n".join(summary_lines)
    
    def get_compound_list_for_prompt(self, context: Dict) -> str:
        """프롬프트에 포함할 화합물 목록 (간단한 리스트)"""
        compounds = list(context.get("compounds", {}).keys())
        if not compounds:
            return "None"
        
        # 최대 100개만
        if len(compounds) <= 100:
            return ", ".join(compounds)
        else:
            return ", ".join(compounds[:100]) + f" ... and {len(compounds) - 100} more"


def modify_analyze_yolo_with_context():
    """
    analyze_yolo_extracted_images.py의 analyze_image 메서드를 수정하여
    이전 컨텍스트를 받도록 함
    """
    # 이 함수는 analyze_yolo_extracted_images.py를 직접 수정하는 대신
    # wrapper 함수를 제공하거나, 해당 파일을 수정해야 함
    pass


def modify_excel_extraction_with_context():
    """
    extract_excel_supplements.py를 수정하여
    이전 컨텍스트를 받도록 함
    """
    pass


if __name__ == "__main__":
    # 테스트
    pipeline = ContextualExtractionPipeline()
    context = pipeline.get_accumulated_context("PMC6989674")
    
    # 예시: 텍스트에서 추출한 정보
    text_extraction = [
        {
            "compound_name": "Compound A",
            "aliases": ["A", "Comp-A"],
            "attributes": {
                "caco2": "15.8 1e-6 cm/s",
                "ppb": "45.2 %"
            }
        }
    ]
    
    pipeline.update_context_from_extraction(context, "text_extraction", text_extraction)
    pipeline.save_accumulated_context("PMC6989674", context)
    
    # 컨텍스트 포맷팅 확인
    formatted = pipeline.format_context_for_prompt(context)
    print(formatted)


