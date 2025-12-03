#!/usr/bin/env python3
"""
ADMET/Organoid 연구를 위한 Vision Agent
- nanoMINER 구조 기반
- Figure/Table에서 ADMET 관련 정보 추출
- YOLO로 감지된 figure/table 분석
"""

import os
import io
import fitz
import torch
import tempfile
import json
import base64
from pathlib import Path
from time import time
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
import logging
import openai

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ADMETVisionAgent:
    def __init__(self, yolo_model_path=None, coreference_dict_path=None):
        """
        ADMET/Organoid 연구를 위한 Vision Agent 초기화
        
        Args:
            yolo_model_path: YOLO 모델 경로 (선택적)
            coreference_dict_path: Coreference dictionary JSON 파일 경로 (선택적)
        """
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.coreference_dict = {}
        
        # OpenAI API 키 설정
        openai.api_key = OPENAI_API_KEY
        
        # Coreference dictionary 로드
        if coreference_dict_path and os.path.exists(coreference_dict_path):
            logger.info(f"Coreference dictionary 로딩: {coreference_dict_path}")
            with open(coreference_dict_path, 'r', encoding='utf-8') as f:
                self.coreference_dict = json.load(f)
            logger.info(f"동의어 그룹 {len(self.coreference_dict)} 개 로드됨")
        else:
            logger.info("Coreference dictionary 미사용")
        
        if yolo_model_path and os.path.exists(yolo_model_path):
            logger.info(f"YOLO 모델 로딩: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path)
    
    def load_extracted_figures_tables(self, graph_extracted_dir):
        """
        이미 추출된 figure와 table 이미지들을 로드
        
        Args:
            graph_extracted_dir: graph_extracted 폴더 경로 (예: ./graph_extracted/test_pdf)
            
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
            for i, fig_file in enumerate(figure_files, 1):
                try:
                    image = Image.open(fig_file)
                    extracted_images.append({
                        'image': image,
                        'filename': fig_file.name,
                        'class': 'figure',
                        'confidence': 1.0,  # 이미 추출된 것이므로 높은 신뢰도
                        'file_path': str(fig_file)
                    })
                    logger.info(f"Figure 로드: {fig_file.name}")
                except Exception as e:
                    logger.error(f"Figure 로드 실패 {fig_file.name}: {e}")
        
        # Tables 로드
        tables_dir = graph_dir / "tables"
        if tables_dir.exists():
            table_files = sorted(tables_dir.glob("*.png"))
            for i, table_file in enumerate(table_files, 1):
                try:
                    image = Image.open(table_file)
                    extracted_images.append({
                        'image': image,
                        'filename': table_file.name,
                        'class': 'table',
                        'confidence': 1.0,  # 이미 추출된 것이므로 높은 신뢰도
                        'file_path': str(table_file)
                    })
                    logger.info(f"Table 로드: {table_file.name}")
                except Exception as e:
                    logger.error(f"Table 로드 실패 {table_file.name}: {e}")
        
        logger.info(f"총 {len(extracted_images)}개의 figure/table 로드 완료")
        return extracted_images
    
    def _extract_all_pages_as_images(self, pdf_path):
        """YOLO 없이 모든 페이지를 이미지로 변환"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = io.BytesIO(pix.tobytes())
            page_image = Image.open(img_data)
            
            images.append({
                'image': page_image,
                'page_num': page_num + 1,
                'class': 'page',
                'confidence': 1.0,
                'bbox': [0, 0, page_image.width, page_image.height]
            })
        
        doc.close()
        return images
    
    def analyze_figure_table(self, image_data):
        """
        Figure/Table을 ADMET/Organoid 관점에서 분석
        
        Args:
            image_data: 이미지 데이터 (dict)
            
        Returns:
            str: 분석 결과 텍스트
        """
        image = image_data['image']
        obj_class = image_data['class']
        confidence = image_data['confidence']
        
        # 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # ADMET/Organoid 특화 프롬프트
        query = self._get_admet_analysis_prompt(obj_class, 1)  # 페이지 번호는 임시로 1
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.0
            )
            
            # JSON 응답 파싱 시도
            response_text = response.choices[0].message.content
            try:
                # JSON 부분만 추출 (```json ... ``` 형태일 수 있음)
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text.strip()
                
                parsed_json = json.loads(json_text)
                return parsed_json
                
            except json.JSONDecodeError as je:
                logger.warning(f"JSON 파싱 실패, 원본 텍스트 반환: {je}")
                return response_text
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _get_admet_analysis_prompt(self, obj_class, page_num):
        """ADMET 분석을 위한 프롬프트 생성 - 8개 핵심 지표 중심"""
        
        # Coreference dictionary 정보 추가
        coref_info = ""
        if self.coreference_dict:
            coref_info = f"""
            
            **ENTITY ALIASES FROM TEXT ANALYSIS (동의어 사전):**
            {json.dumps(self.coreference_dict, ensure_ascii=False, indent=2)}
            
            When you see any of these aliases in the image, recognize them as the same entity.
            For example, if you see "HLO" or "human liver organoid" in the figure, they refer to "Liver Organoid".
            """
        
        if obj_class == 'figure':
            return f"""
            Analyze this figure (Page {page_num}) from a scientific article and extract the following information in JSON format:
            {coref_info}
            
            **REQUIRED OUTPUT FORMAT:**
            {{
                "admet_indicators": {{
                    "caco2_permeability": {{
                        "found": true/false,
                        "value": "numerical value with units",
                        "description": "brief description"
                    }},
                    "ppb": {{
                        "found": true/false,
                        "value": "percentage or binding value",
                        "description": "plasma protein binding data"
                    }},
                    "bbb": {{
                        "found": true/false,
                        "value": "permeability or penetration value",
                        "description": "blood-brain barrier data"
                    }},
                    "cyp1a2": {{
                        "found": true/false,
                        "value": "IC50 or inhibition value",
                        "description": "CYP1A2 inhibition data"
                    }},
                    "t_half": {{
                        "found": true/false,
                        "value": "half-life value with units",
                        "description": "elimination half-life data"
                    }},
                    "herg": {{
                        "found": true/false,
                        "value": "IC50 or inhibition value",
                        "description": "hERG channel inhibition data"
                    }},
                    "dili": {{
                        "found": true/false,
                        "value": "toxicity indicators",
                        "description": "liver injury markers"
                    }},
                    "carcinogenicity": {{
                        "found": true/false,
                        "value": "genotoxicity indicators",
                        "description": "carcinogenic potential data"
                    }}
                }},
                "smiles_structures": [
                    "SMILES string 1",
                    "SMILES string 2"
                ],
                "summary": "Brief summary of the figure content and key findings"
            }}
            
            **KEY TERMS TO SEARCH FOR:**
            - Caco-2, Papp, permeability, absorption
            - PPB, plasma protein binding, protein binding
            - BBB, blood-brain barrier, brain penetration
            - CYP1A2, cytochrome P450, metabolism
            - half-life, t1/2, elimination
            - hERG, QT prolongation, cardiac toxicity
            - DILI, liver injury, hepatotoxicity
            - carcinogenic, genotoxic, mutagenic
            - SMILES, molecular structure, chemical formula
            
            **INSTRUCTIONS:**
            1. Only mark "found": true if the specific indicator is clearly present in the figure
            2. Extract EXACT numerical values with units (e.g., "IC50 = 2.5 μM", "t1/2 = 3.2 hours", "AUC = 125 ng·h/mL")
            3. Include all relevant numerical data (mean ± SD, error bars, confidence intervals, etc.)
            4. Look for SMILES strings or molecular structures
            5. Use the entity aliases from text analysis to recognize the same concepts
            6. Provide a concise summary of the figure content and key findings
            7. If no ADMET data is found, set all "found" to false and explain what the figure shows instead
            """
        
        elif obj_class == 'table':
            return f"""
            Analyze this table (Page {page_num}) from a scientific article and extract the following information in JSON format:
            {coref_info}
            
            **REQUIRED OUTPUT FORMAT:**
            {{
                "admet_indicators": {{
                    "caco2_permeability": {{
                        "found": true/false,
                        "value": "numerical value with units",
                        "description": "brief description"
                    }},
                    "ppb": {{
                        "found": true/false,
                        "value": "percentage or binding value",
                        "description": "plasma protein binding data"
                    }},
                    "bbb": {{
                        "found": true/false,
                        "value": "permeability or penetration value",
                        "description": "blood-brain barrier data"
                    }},
                    "cyp1a2": {{
                        "found": true/false,
                        "value": "IC50 or inhibition value",
                        "description": "CYP1A2 inhibition data"
                    }},
                    "t_half": {{
                        "found": true/false,
                        "value": "half-life value with units",
                        "description": "elimination half-life data"
                    }},
                    "herg": {{
                        "found": true/false,
                        "value": "IC50 or inhibition value",
                        "description": "hERG channel inhibition data"
                    }},
                    "dili": {{
                        "found": true/false,
                        "value": "toxicity indicators",
                        "description": "liver injury markers"
                    }},
                    "carcinogenicity": {{
                        "found": true/false,
                        "value": "genotoxicity indicators",
                        "description": "carcinogenic potential data"
                    }}
                }},
                "smiles_structures": [
                    "SMILES string 1",
                    "SMILES string 2"
                ],
                "summary": "Brief summary of the table content and key findings"
            }}
            
            **KEY TERMS TO SEARCH FOR:**
            - Caco-2, Papp, permeability, absorption
            - PPB, plasma protein binding, protein binding
            - BBB, blood-brain barrier, brain penetration
            - CYP1A2, cytochrome P450, metabolism
            - half-life, t1/2, elimination
            - hERG, QT prolongation, cardiac toxicity
            - DILI, liver injury, hepatotoxicity
            - carcinogenic, genotoxic, mutagenic
            - SMILES, molecular structure, chemical formula
            
            **INSTRUCTIONS:**
            1. Only mark "found": true if the specific indicator is clearly present in the table
            2. Extract EXACT numerical values with units (e.g., "IC50 = 2.5 μM", "t1/2 = 3.2 hours", "AUC = 125 ng·h/mL")
            3. Include all relevant numerical data from tables (mean ± SD, error bars, confidence intervals, etc.)
            4. Look for SMILES strings or molecular structures
            5. Use the entity aliases from text analysis to recognize the same concepts
            6. Provide a concise summary of the table content and key findings
            7. If no ADMET data is found, set all "found" to false and explain what the table shows instead
            """
        
        else:  # page or other
            return f"""
            Analyze this page (Page {page_num}) for any ADMET or organoid-related content.
            
            Look for:
            - Figures showing drug effects, toxicity, or organoid studies
            - Tables with pharmacokinetic, pharmacodynamic, or toxicity data
            - Graphs showing dose-response, time-course, or viability data
            - Any visual content related to drug metabolism, distribution, or organoid models
            
            If ADMET/organoid content is found, provide a brief analysis.
            If not relevant, state that this page does not contain ADMET/organoid information.
            """
    
    def analyze_extracted_images(self, graph_extracted_dir):
        """
        이미 추출된 figure/table 이미지들을 ADMET/Organoid 관점에서 분석
        
        Args:
            graph_extracted_dir: graph_extracted 폴더 경로
            
        Returns:
            dict: 분석 결과
        """
        logger.info(f"추출된 이미지 분석 시작: {graph_extracted_dir}")
        
        # 이미 추출된 Figure/Table 로드
        extracted_images = self.load_extracted_figures_tables(graph_extracted_dir)
        
        if not extracted_images:
            logger.warning("로드된 figure/table이 없습니다.")
            return {"error": "No figures or tables found"}
        
        # 분석 결과 저장 디렉토리 생성
        analysis_dir = Path(graph_extracted_dir) / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # 각 이미지 분석
        analysis_results = {}
        figure_analyses = []
        table_analyses = []
        
        for i, image_data in enumerate(extracted_images):
            logger.info(f"이미지 {i+1}/{len(extracted_images)} 분석 중... ({image_data['class']}: {image_data['filename']})")
            
            try:
                analysis = self.analyze_figure_table(image_data)
                analysis_result = {
                    "filename": image_data['filename'],
                    "class": image_data['class'],
                    "confidence": image_data['confidence'],
                    "analysis": analysis,
                    "file_path": image_data['file_path']
                }
                
                analysis_results[f"image_{i+1}"] = analysis_result
                
                # 클래스별로 분리 저장
                if image_data['class'] == 'figure':
                    figure_analyses.append(analysis_result)
                elif image_data['class'] == 'table':
                    table_analyses.append(analysis_result)
                
            except Exception as e:
                logger.error(f"이미지 {i+1} 분석 실패: {e}")
                analysis_result = {
                    "filename": image_data['filename'],
                    "class": image_data['class'],
                    "confidence": image_data['confidence'],
                    "analysis": f"Analysis failed: {str(e)}",
                    "file_path": image_data['file_path']
                }
                analysis_results[f"image_{i+1}"] = analysis_result
        
        # 클래스별 분석 결과 저장
        with open(analysis_dir / "figure_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(figure_analyses, f, ensure_ascii=False, indent=2)
        
        with open(analysis_dir / "table_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(table_analyses, f, ensure_ascii=False, indent=2)
        
        # 전체 요약 생성
        summary = self._generate_summary(analysis_results)
        
        # 요약 저장
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
        """분석 결과 요약 생성"""
        figures = [r for r in analysis_results.values() if r['class'] == 'figure']
        tables = [r for r in analysis_results.values() if r['class'] == 'table']
        
        # ADMET 관련 콘텐츠 카운트
        admet_relevant = 0
        found_indicators = {
            "caco2_permeability": 0,
            "ppb": 0,
            "bbb": 0,
            "cyp1a2": 0,
            "t_half": 0,
            "herg": 0,
            "dili": 0,
            "carcinogenicity": 0
        }
        
        for r in analysis_results.values():
            analysis = r.get('analysis', {})
            if isinstance(analysis, dict):
                # JSON 형태의 응답인 경우
                admet_indicators = analysis.get('admet_indicators', {})
                for indicator in found_indicators.keys():
                    if indicator in admet_indicators:
                        indicator_data = admet_indicators[indicator]
                        if isinstance(indicator_data, dict) and indicator_data.get('found', False):
                            found_indicators[indicator] += 1
                            admet_relevant += 1
            elif isinstance(analysis, str):
                # 문자열 형태의 응답인 경우
                analysis_text = analysis.lower()
                if any(keyword in analysis_text for keyword in ['admet', 'organoid', 'pharmacokinetic', 'toxicity', 'viability', 'ic50', 'ec50']):
                    admet_relevant += 1
        
        summary = {
            "total_figures": len(figures),
            "total_tables": len(tables),
            "total_images": len(analysis_results),
            "admet_relevant_content": admet_relevant,
            "found_indicators": found_indicators,
            "analysis_timestamp": time()
        }
        
        return summary
    
    def save_results(self, results, output_path):
        """분석 결과 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과 저장 완료: {output_file}")

def main():
    """테스트용 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ADMET/Organoid Vision Agent")
    parser.add_argument("--graph_extracted_dir", required=True, help="graph_extracted 폴더 경로 (예: ./graph_extracted/test_pdf)")
    parser.add_argument("--yolo_model", help="YOLO 모델 경로 (선택적, 현재는 사용하지 않음)")
    parser.add_argument("--coreference_dict", help="Coreference dictionary JSON 파일 경로 (선택적)")
    parser.add_argument("--output", help="전체 결과 저장 경로 (선택적)")
    
    args = parser.parse_args()
    
    # Vision Agent 초기화
    agent = ADMETVisionAgent(
        yolo_model_path=args.yolo_model,
        coreference_dict_path=args.coreference_dict
    )
    
    # 추출된 이미지 분석
    results = agent.analyze_extracted_images(args.graph_extracted_dir)
    
    # 선택적으로 전체 결과 저장
    if args.output:
        agent.save_results(results, args.output)
        print(f"전체 결과 저장: {args.output}")
    
    print(f"분석 완료!")
    print(f"분석 결과 위치: {results.get('analysis_dir', 'N/A')}")
    print(f"총 {results['total_images']}개 이미지 분석")
    print(f"ADMET 관련 콘텐츠: {results['summary']['admet_relevant_content']}개")

if __name__ == "__main__":
    main()
