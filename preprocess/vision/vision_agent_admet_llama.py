#!/usr/bin/env python3
"""
ADMET/Organoid 연구를 위한 Vision Agent (Llama 4 멀티모달)
- Llama 4로 Figure/Table 분석
- OpenAI 대신 Ollama 사용
- 이미지 인식 및 ADMET 정보 추출
"""

import os
import io
import json
import logging
from pathlib import Path
from time import time
from PIL import Image
import ollama
import base64
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADMETVisionAgentLlama:
    def __init__(self, model_name="llama4:latest", coreference_dict_path=None, use_gpt=False):
        """
        ADMET/Organoid 연구를 위한 Vision Agent 초기화
        
        Args:
            model_name: Ollama 모델 이름 또는 GPT 모델 이름 (default: llama4:latest)
            coreference_dict_path: Coreference dictionary JSON 파일 경로 (선택적)
            use_gpt: True면 GPT-4o 사용, False면 Ollama 사용 (default: False)
        """
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
        
        self.coreference_dict = {}
        
        # Coreference dictionary 로드
        if coreference_dict_path and os.path.exists(coreference_dict_path):
            logger.info(f"Coreference dictionary 로딩: {coreference_dict_path}")
            with open(coreference_dict_path, 'r', encoding='utf-8') as f:
                self.coreference_dict = json.load(f)
            logger.info(f"동의어 그룹 {len(self.coreference_dict)} 개 로드됨")
        else:
            logger.info("Coreference dictionary 미사용")
    
    def load_extracted_figures_tables(self, graph_extracted_dir):
        """
        이미 추출된 figure와 table 이미지들을 로드
        
        Args:
            graph_extracted_dir: graph_extracted 폴더 경로
            
        Returns:
            list: 로드된 figure/table 이미지 리스트
        """
        extracted_images = []
        graph_dir = Path(graph_extracted_dir)
        
        if not graph_dir.exists():
            logger.error(f"Graph extracted directory not found: {graph_dir}")
            return extracted_images
        
        # Figures 로드
        figures_dir = graph_dir / "figures"
        if figures_dir.exists():
            figure_files = sorted(figures_dir.glob("*.png"))
            for fig_file in figure_files:
                try:
                    image = Image.open(fig_file)
                    extracted_images.append({
                        'image': image,
                        'filename': fig_file.name,
                        'class': 'figure',
                        'confidence': 1.0,
                        'file_path': str(fig_file)
                    })
                    logger.info(f"Figure 로드: {fig_file.name}")
                except Exception as e:
                    logger.error(f"Figure 로드 실패 {fig_file.name}: {e}")
        
        # Tables 로드
        tables_dir = graph_dir / "tables"
        if tables_dir.exists():
            table_files = sorted(tables_dir.glob("*.png"))
            for table_file in table_files:
                try:
                    image = Image.open(table_file)
                    extracted_images.append({
                        'image': image,
                        'filename': table_file.name,
                        'class': 'table',
                        'confidence': 1.0,
                        'file_path': str(table_file)
                    })
                    logger.info(f"Table 로드: {table_file.name}")
                except Exception as e:
                    logger.error(f"Table 로드 실패 {table_file.name}: {e}")
        
        logger.info(f"총 {len(extracted_images)}개의 figure/table 로드 완료")
        return extracted_images
    
    def analyze_figure_table_llama(self, image_data):
        """
        Figure/Table 분석 (GPT-4o 또는 Llama)
        
        Args:
            image_data: 이미지 데이터 (dict)
            
        Returns:
            dict: 분석 결과
        """
        image = image_data['image']
        obj_class = image_data['class']
        
        # ADMET 분석 프롬프트
        prompt = self._get_admet_analysis_prompt(obj_class)
        
        try:
            if self.use_gpt:
                # GPT-4o 사용
                logger.info(f"GPT-4o 분석 중: {image_data['filename']}")
                
                # 이미지를 base64로 인코딩
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                response = self.client.chat.completions.create(
                    model=self.model_name if self.model_name.startswith("gpt") else "gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=8000,
                    temperature=0.1
                )
                
                generated_text = response.choices[0].message.content
                logger.info(f"GPT 응답 받음 (길이: {len(generated_text)}자)")
                
            else:
                # Ollama 사용
                logger.info(f"Llama 분석 중: {image_data['filename']}")
                
                # 이미지 경로
                image_path = image_data['file_path']
                
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_path]
                        }
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": 32000
                    }
                )
                
                generated_text = response.get('message', {}).get('content', '')
                logger.info(f"Llama 응답 받음 (길이: {len(generated_text)}자)")
            
            # JSON 파싱 시도
            try:
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
                
                parsed_json = json.loads(json_text)
                return parsed_json
                
            except json.JSONDecodeError as je:
                logger.warning(f"JSON 파싱 실패, 원본 텍스트 반환: {je}")
                return {"raw_response": generated_text}
            
        except Exception as e:
            logger.error(f"분석 실패: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _get_admet_analysis_prompt(self, obj_class):
        """Generic 데이터 추출 프롬프트 - 모든 정보 추출"""
        
        # Coreference dictionary 정보
        coref_info = ""
        if self.coreference_dict:
            coref_info = f"""
            
**ENTITY ALIASES FROM TEXT ANALYSIS (동의어 사전):**
{json.dumps(self.coreference_dict, ensure_ascii=False, indent=2)}
"""
        
        if obj_class == 'figure':
            return f"""Analyze this figure from a scientific article and extract ALL available information.

{coref_info}

**YOUR TASK:**
Extract ALL information you can see in this figure. Be as comprehensive and detailed as possible.

**WHAT TO EXTRACT (everything you see):**
- Compound/drug names (use aliases from coreference if available)
- Chemical structures (SMILES, molecular formulas, etc.)
- Numerical values with units (IC50, EC50, AUC, Cmax, concentrations, percentages, time values, etc.)
- Measurements and experimental results
- Biological/chemical processes
- Relationships between entities
- Trends, patterns, comparisons
- Any ADMET-related information (absorption, distribution, metabolism, excretion, toxicity)
- Organoid-related information
- Experimental conditions
- Statistical information (p-values, error bars, etc.)
- Labels, legends, annotations

**OUTPUT FORMAT:**
Return a JSON object with ANY structure that best represents the information you extracted.
Organize the data logically - use nested objects, arrays, or flat key-value pairs as appropriate.
Include a "summary" field with a brief description of what the figure shows.

**EXAMPLES OF VALID OUTPUT STRUCTURES:**
- If it's a single compound with properties: {{"compound_name": "...", "properties": {{...}}, "summary": "..."}}
- If it's multiple compounds: {{"compounds": [{{...}}, {{...}}], "summary": "..."}}
- If it's experimental data: {{"experiments": [...], "results": {{...}}, "summary": "..."}}
- If it's a complex figure: Create the structure that best fits the data

**INSTRUCTIONS:**
1. Extract EVERYTHING - don't leave out any information
2. Use the structure that best fits the data (don't force a rigid format)
3. Include all numerical values with their units
4. Include all compound names and use aliases from coreference when applicable
5. Be thorough and comprehensive - more detail is better
6. Return valid JSON only
"""
        
        elif obj_class == 'table':
            return f"""Analyze this table from a scientific article and extract ALL available information.

{coref_info}

**YOUR TASK:**
Extract ALL information from this table. Parse it systematically and capture every piece of data.

**WHAT TO EXTRACT (everything in the table):**
- All column headers
- All row data
- All compound/drug names (use aliases from coreference if available)
- All numerical values with units
- All measurements, properties, attributes
- Relationships between columns/rows
- Any ADMET-related information
- Any patterns or trends in the data
- Statistical values
- Footnotes or annotations

**OUTPUT FORMAT:**
Return a JSON object with ANY structure that best represents the table data.
You can organize it as:
- An array of row objects
- A nested structure with columns and rows
- A compound-centric structure with properties
- Or any other structure that fits the data best
Include a "summary" field describing what the table contains.

**EXAMPLES OF VALID OUTPUT STRUCTURES:**
- Row-based: {{"columns": [...], "rows": [{{...}}, {{...}}], "summary": "..."}}
- Compound-based: {{"compounds": [{{"name": "...", "properties": {{...}}}}, ...], "summary": "..."}}
- Matrix-style: {{"headers": [...], "data": [[...], [...], ...], "summary": "..."}}
- Custom structure: Use whatever best fits your specific table

**INSTRUCTIONS:**
1. Parse the ENTIRE table - don't skip any rows or columns
2. Extract ALL values with their units
3. Preserve the table structure in a logical way
4. Include all compound names and use aliases from coreference when applicable
5. Be thorough - extract EVERYTHING
6. Return valid JSON only
"""
        
        else:
            return "Describe this image and identify any ADMET or organoid-related content."
    
    def analyze_extracted_images(self, graph_extracted_dir):
        """
        이미 추출된 figure/table 분석
        
        Args:
            graph_extracted_dir: graph_extracted 폴더 경로
            
        Returns:
            dict: 분석 결과
        """
        logger.info(f"추출된 이미지 분석 시작: {graph_extracted_dir}")
        
        # 이미지 로드
        extracted_images = self.load_extracted_figures_tables(graph_extracted_dir)
        
        if not extracted_images:
            logger.warning("로드된 figure/table이 없습니다.")
            return {"error": "No figures or tables found"}
        
        # PMC ID 추출
        pmc_id = None
        graph_dir = Path(graph_extracted_dir)
        
        # 1. 폴더 이름에서 PMC 찾기
        for part in reversed(graph_dir.parts):
            if part.startswith("PMC"):
                pmc_id = part
                break
        
        # 2. 못 찾으면 상위 폴더에서 찾기
        if not pmc_id:
            for parent_dir in graph_dir.parents:
                for part in parent_dir.parts:
                    if part.startswith("PMC"):
                        pmc_id = part
                        break
                if pmc_id:
                    break
        
        # 3. 기본값 설정
        if not pmc_id:
            pmc_id = graph_dir.name
        
        # 분석 결과 저장 디렉토리
        # graph_extracted 폴더 안에 저장 (상대 경로 또는 절대 경로)
        analysis_dir = graph_dir / "analysis_llama"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"저장 위치: {analysis_dir}")
        
        # 각 이미지 분석
        analysis_results = {}
        figure_analyses = []
        table_analyses = []
        
        for i, image_data in enumerate(extracted_images):
            logger.info(f"이미지 {i+1}/{len(extracted_images)} 분석 중... ({image_data['class']}: {image_data['filename']})")
            
            try:
                analysis = self.analyze_figure_table_llama(image_data)
                analysis_result = {
                    "filename": image_data['filename'],
                    "class": image_data['class'],
                    "confidence": image_data['confidence'],
                    "analysis": analysis,
                    "file_path": image_data['file_path']
                }
                
                analysis_results[f"image_{i+1}"] = analysis_result
                
                if image_data['class'] == 'figure':
                    figure_analyses.append(analysis_result)
                elif image_data['class'] == 'table':
                    table_analyses.append(analysis_result)
                
            except Exception as e:
                logger.error(f"이미지 {i+1} 분석 실패: {e}")
                analysis_results[f"image_{i+1}"] = {
                    "filename": image_data['filename'],
                    "class": image_data['class'],
                    "error": str(e)
                }
        
        # 클래스별 분석 결과 저장
        with open(analysis_dir / "figure_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(figure_analyses, f, ensure_ascii=False, indent=2)
        
        with open(analysis_dir / "table_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(table_analyses, f, ensure_ascii=False, indent=2)
        
        # 요약 생성
        summary = self._generate_summary(analysis_results)
        
        with open(analysis_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        result = {
            "graph_extracted_dir": str(graph_extracted_dir),
            "total_images": len(extracted_images),
            "analysis_results": analysis_results,
            "summary": summary,
            "timestamp": time(),
            "analysis_dir": str(analysis_dir)
        }
        
        logger.info(f"이미지 분석 완료: {len(extracted_images)}개 이미지 분석")
        logger.info(f"분석 결과 저장 위치: {analysis_dir}")
        return result
    
    def _generate_summary(self, analysis_results):
        """분석 결과 요약"""
        figures = [r for r in analysis_results.values() if r.get('class') == 'figure']
        tables = [r for r in analysis_results.values() if r.get('class') == 'table']
        
        total_compounds = 0
        total_attributes = 0
        total_numerical_data = 0
        
        for r in analysis_results.values():
            analysis = r.get('analysis', {})
            if isinstance(analysis, dict):
                if 'data_found' in analysis and analysis.get('data_found'):
                    # General compound data
                    if 'compound_name' in analysis and analysis['compound_name']:
                        total_compounds += 1
                    if 'attributes' in analysis:
                        total_attributes += len(analysis['attributes'])
                    if 'numerical_data' in analysis:
                        total_numerical_data += len(analysis['numerical_data'])
                    
                    # Table compounds
                    if 'compounds' in analysis and isinstance(analysis['compounds'], list):
                        total_compounds += len(analysis['compounds'])
        
        return {
            "total_figures": len(figures),
            "total_tables": len(tables),
            "total_images": len(analysis_results),
            "total_compounds": total_compounds,
            "total_attributes": total_attributes,
            "total_numerical_data": total_numerical_data,
            "analysis_timestamp": time()
        }

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ADMET Vision Agent (Llama)")
    parser.add_argument("--graph_extracted_dir", required=True, help="graph_extracted 폴더 경로")
    parser.add_argument("--model", default="llama4:latest", help="Ollama 모델 이름")
    parser.add_argument("--coreference_dict", help="Coreference dictionary JSON 파일 경로")
    parser.add_argument("--output", help="전체 결과 저장 경로")
    
    args = parser.parse_args()
    
    # Vision Agent 초기화
    agent = ADMETVisionAgentLlama(
        model_name=args.model,
        coreference_dict_path=args.coreference_dict
    )
    
    # 분석 실행
    results = agent.analyze_extracted_images(args.graph_extracted_dir)
    
    # 선택적으로 전체 결과 저장
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"전체 결과 저장: {args.output}")
    
    print(f"✅ 분석 완료!")
    print(f"  분석 이미지: {results['total_images']}개")
    print(f"  총 화합물: {results['summary']['total_compounds']}개")
    print(f"  총 속성: {results['summary']['total_attributes']}개")
    print(f"  결과 위치: {results['analysis_dir']}")

if __name__ == "__main__":
    main()

