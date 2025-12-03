#!/usr/bin/env python3
"""
5개 데이터 통합 및 정리
- GPT-4o 추출 결과
- 보충자료 (Excel)
- 이미지 분석
- 텍스트 분석
- Coreference dictionary
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import openai
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIntegrator:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def load_all_data(self, pmc_id: str) -> Dict[str, Any]:
        """5개 데이터 소스 로드"""
        logger.info(f"데이터 로딩: {pmc_id}")
        
        data = {}
        
        # 1. GPT-4o 추출 결과
        gpt_path = Path("final_extracted") / pmc_id / f"{pmc_id}_final_admet.json"
        if gpt_path.exists():
            with open(gpt_path, 'r', encoding='utf-8') as f:
                data["gpt_extraction"] = json.load(f)
            logger.info(f"  GPT 추출: {len(data['gpt_extraction'].get('records', []))}개 화합물")
        
        # 2. 보충자료 (Excel)
        supplement_path = Path("supplement_extracted") / pmc_id / f"{pmc_id}_compounds_from_excel.json"
        if supplement_path.exists():
            with open(supplement_path, 'r', encoding='utf-8') as f:
                supplement_data = json.load(f)
                # 첫 20개만 샘플링 (전체는 너무 큼)
                if isinstance(supplement_data, list) and len(supplement_data) > 20:
                    data["supplement"] = supplement_data[:20]
                else:
                    data["supplement"] = supplement_data
            logger.info(f"  보충자료: {len(data['supplement'])}개 화합물 (샘플)")
        
        # 3. 이미지 분석
        figure_path = Path("graph_extracted") / pmc_id / "analysis_llama" / "figure_analyses.json"
        if figure_path.exists():
            with open(figure_path, 'r', encoding='utf-8') as f:
                figures = json.load(f)
                # 처음 10개만
                data["images"] = figures[:10] if isinstance(figures, list) else figures
            logger.info(f"  이미지 분석: {len(data['images'])}개")
        
        # 4. 텍스트 분석
        text_path = Path("text_analyzed") / pmc_id / "text_analysis_result.json"
        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                data["text_analysis"] = json.load(f)
            logger.info("  텍스트 분석: 로드됨")
        
        # 5. Coreference dictionary
        coreference_path = Path("text_analyzed") / pmc_id / "global_coreference.json"
        if coreference_path.exists():
            with open(coreference_path, 'r', encoding='utf-8') as f:
                data["coreference"] = json.load(f)
            logger.info("  Coreference: 로드됨")
        
        return data
    
    def integrate_with_llama(self, all_data: Dict[str, Any], pmc_id: str) -> Dict[str, Any]:
        """Llama로 5개 데이터 통합 및 정리"""
        
        logger.info("Llama로 데이터 통합 중...")
        
        prompt = f"""You are integrating ADMET data from multiple sources for a scientific paper.

**Input Data from 5 Sources:**

1. **GPT-4o PDF Extraction:**
{json.dumps(all_data.get('gpt_extraction', {}), ensure_ascii=False, indent=2)[:3000]}

2. **Supplemental Data (Excel):**
{json.dumps(all_data.get('supplement', []), ensure_ascii=False, indent=2)[:3000]}

3. **Image/Figure Analysis:**
{json.dumps(all_data.get('images', []), ensure_ascii=False, indent=2)[:3000]}

4. **Text Analysis:**
{json.dumps(all_data.get('text_analysis', {}), ensure_ascii=False, indent=2)[:3000]}

5. **Coreference Dictionary:**
{json.dumps(all_data.get('coreference', {}), ensure_ascii=False, indent=2)}

**TASK:**
Create a comprehensive, unified ADMET dataset by:
1. Merging all compounds from all 5 sources
2. Using coreference dictionary to identify synonyms (e.g., "Bosentan" = "Tracleer")
3. Combining data from different sources for the same compound
4. Filling missing values when available from other sources
5. Prioritizing: Supplemental data > Images > Text > GPT extraction

**Required Output Format (JSON only):**
```json
{{
  "schema_version": "1.0",
  "created_at": "ISO 8601 timestamp",
  "source_bundle": {{
    "pmc_id": "{pmc_id}",
    "sources": ["gpt_extraction", "supplement", "images", "text", "coreference"]
  }},
  "records": [
    {{
      "compound_name": "Chemical name (canonical from coreference)",
      "smiles": "SMILES string or null (ONLY if explicitly found in sources)",
      "admet": {{
        "caco2": {{ "value": number or null, "unit": "1e-6 cm/s" }},
        "ppb": {{ "value": number or null, "unit": "%" }},
        "bbb": {{ "value": number or null, "unit": "logBB" }},
        "cyp1a2": {{ "value": number or null, "unit": "µM", "assay": "IC50 or null" }},
        "t_half": {{ "value": number or null, "unit": "h" }},
        "herg": {{ "value": number or null, "unit": "µM", "assay": "IC50 or null" }},
        "dili": {{ "label": "High/Medium/Low risk or null", "score": number or null }},
        "carcinogenicity": {{ "label": "Positive/Negative or null", "score": number or null }}
      }},
      "summary": "Brief summary of the compound's ADMET profile from all sources",
      "notes": "Additional information or null",
      "provenance": {{
        "sources": ["gpt", "supplement", "image", etc.],
        "conflicts": "if any"
      }}
    }}
  ]
}}
```

**Instructions:**
1. Extract ALL compounds from ALL sources
2. Use coreference dictionary to resolve synonyms
3. When same compound appears in multiple sources, merge the data
4. Fill missing values from other sources when available
5. Keep track of data sources (provenance)
6. Prioritize most reliable sources (supplement > images > text > gpt)

Output JSON only:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a specialized assistant for integrating ADMET data from multiple sources. Extract accurate information and preserve provenance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16384
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"GPT-4o 응답 받음 (길이: {len(generated_text)}자)")
            
            # JSON 파싱
            if "```json" in generated_text:
                json_start = generated_text.find("```json") + 7
                json_end = generated_text.find("```", json_start)
                json_text = generated_text[json_start:json_end].strip()
            elif "```" in generated_text:
                json_start = generated_text.find("```") + 3
                json_end = generated_text.find("```", json_start)
                json_text = generated_text[json_start:json_end].strip()
            else:
                json_text = generated_text.strip()
            
            result = json.loads(json_text)
            return result
            
        except Exception as e:
            logger.error(f"GPT-4o 통합 실패: {e}")
            return {
                "schema_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "source_bundle": {"pmc_id": pmc_id, "sources": []},
                "records": []
            }
    
    def integrate_pmc(self, pmc_id: str) -> Dict[str, Any]:
        """특정 PMC ID 통합 처리"""
        logger.info(f"데이터 통합 시작: {pmc_id}")
        
        # 모든 데이터 로드
        all_data = self.load_all_data(pmc_id)
        
        # Llama로 통합
        result = self.integrate_with_llama(all_data, pmc_id)
        
        # 결과 저장
        output_dir = Path("final_integrated") / pmc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{pmc_id}_integrated.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"통합 결과 저장: {output_file}")
        
        return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="5개 데이터 통합")
    parser.add_argument("--pmc_id", required=True, help="PMC ID (예: PMC7878295)")
    
    args = parser.parse_args()
    
    integrator = DataIntegrator()
    result = integrator.integrate_pmc(args.pmc_id)
    
    print(f"✅ 데이터 통합 완료!")
    print(f"  PMC ID: {result.get('source_bundle', {}).get('pmc_id', 'N/A')}")
    print(f"  레코드 수: {len(result.get('records', []))}")

if __name__ == "__main__":
    main()

