#!/usr/bin/env python3
"""
Final ADMET Extraction - ReAct Agent 버전
- 기존 스크립트들을 도구(tools)로 래핑
- LangChain ReAct Agent로 구조화된 출력 생성
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import os

# LangChain imports
from langchain.agents import AgentType, initialize_agent, tool
from langchain_openai import ChatOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReactADMETExtractor:
    """ReAct 패턴 기반 ADMET 추출기"""
    
    def __init__(self, pmc_id: str, test_mode: bool = False):
        self.pmc_id = pmc_id
        if test_mode:
            # 테스트 모드: data_test 폴더 사용
            self.base_dir = Path("./data_test/raws")
            self.supp_base_dir = Path("./data_test/supp")
            self.text_dir = Path("./data_test/text_extracted")
            self.graph_dir = Path("./data_test/graph_extracted")
            self.graph_analyzed_dir = Path("./data_test/graph_analyzed")
            self.supp_extracted_dir = Path("./data_test/supp_extracted")
            self.text_analyzed_dir = Path("./data_test/text_analyzed")
            self.final_dir = Path("./data_test/final_extracted")
        else:
            # 프로덕션 모드: 기본 경로 사용
            self.base_dir = Path("./raws_v1")
            self.supp_base_dir = Path("./supp_raws_v1")
            self.text_dir = Path("./text_extracted")
            self.graph_dir = Path("./graph_extracted")
            self.graph_analyzed_dir = Path("./graph_analyzed")
            self.supp_extracted_dir = Path("./supp_extracted")
            self.text_analyzed_dir = Path("./text_analyzed")
            self.final_dir = Path("./final_extracted")
        
        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # LangChain LLM 초기화
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            openai_api_key=api_key,
        )
        
        # 도구들 정의
        self.tools = self._create_tools()
        
        # Agent 초기화
        self.agent = self._create_agent()
    
    def _create_tools(self):
        """기존 스크립트들을 도구로 래핑"""
        pmc_id = self.pmc_id
        
        @tool("extract_text")
        def extract_text(query: str) -> str:
            """Extracts text from the main article PDF. Returns the extracted text content.
            
            Args:
                query: Unused, but required by LangChain tool format.
            
            Returns:
                JSON string containing extracted text or error message.
            """
            try:
                pdf_path = self.base_dir / pmc_id / "article.pdf"
                if not pdf_path.exists():
                    return json.dumps({"error": f"PDF not found: {pdf_path}", "text": ""})
                
                output_dir = self.text_dir / pmc_id
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # pdf_to_text.py 실행
                result = subprocess.run(
                    [
                        sys.executable, "pdf_to_text.py",
                        "--pmc_id", pmc_id,
                        "--base_dir", str(self.base_dir),
                        "--output_dir", str(self.text_dir)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    text_file = output_dir / "extracted_text.txt"
                    if text_file.exists():
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        return json.dumps({
                            "success": True,
                            "text_length": len(text_content),
                            "text_preview": text_content[:5000]  # 처음 5000자만 반환
                        })
                    else:
                        return json.dumps({"error": "Text extraction completed but file not found", "text": ""})
                else:
                    return json.dumps({
                        "error": f"Text extraction failed: {result.stderr}",
                        "text": ""
                    })
            except Exception as e:
                return json.dumps({"error": str(e), "text": ""})
        
        @tool("extract_images")
        def extract_images(query: str) -> str:
            """Extracts figures and tables from the main article PDF using YOLO.
            
            Args:
                query: Unused, but required by LangChain tool format.
            
            Returns:
                JSON string containing extraction results.
            """
            try:
                pdf_path = self.base_dir / pmc_id / "article.pdf"
                if not pdf_path.exists():
                    return json.dumps({"error": f"PDF not found: {pdf_path}"})
                
                output_dir = self.graph_dir / pmc_id / "article"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # inference_yolo.py 실행
                result = subprocess.run(
                    [
                        sys.executable, "inference_yolo.py",
                        "--pdf", str(pdf_path),
                        "--output", str(output_dir),
                        "--confidence", "0.25"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    # figures와 tables 폴더 확인
                    figures_dir = output_dir / "figures"
                    tables_dir = output_dir / "tables"
                    
                    figure_count = len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0
                    table_count = len(list(tables_dir.glob("*.png"))) if tables_dir.exists() else 0
                    
                    return json.dumps({
                        "success": True,
                        "figures_extracted": figure_count,
                        "tables_extracted": table_count,
                        "output_path": str(output_dir)
                    })
                else:
                    return json.dumps({
                        "error": f"Image extraction failed: {result.stderr}",
                        "stdout": result.stdout
                    })
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool("analyze_images")
        def analyze_images(query: str) -> str:
            """Analyzes extracted figures and tables using GPT-4o to extract compound and ADMET information.
            
            Args:
                query: Unused, but required by LangChain tool format.
            
            Returns:
                JSON string containing analysis results.
            """
            try:
                figures_dir = self.graph_dir / pmc_id / "article" / "figures"
                tables_dir = self.graph_dir / pmc_id / "article" / "tables"
                
                if not figures_dir.exists() and not tables_dir.exists():
                    return json.dumps({"error": "No images found. Extract images first using extract_images tool."})
                
                output_dir = self.graph_analyzed_dir / pmc_id
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # batch_analyze_main_images.py 실행
                result = subprocess.run(
                    [
                        sys.executable, "batch_analyze_main_images.py",
                        "--pmc_id", pmc_id,
                        "--input_dir", str(self.graph_dir),
                        "--output_dir", str(self.graph_analyzed_dir),
                        "--use-gpt"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    analysis_file = output_dir / "all_analyses.json"
                    if analysis_file.exists():
                        with open(analysis_file, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                        
                        # 요약 정보 반환
                        total_images = len(analysis_data) if isinstance(analysis_data, list) else 0
                        return json.dumps({
                            "success": True,
                            "total_images_analyzed": total_images,
                            "analysis_preview": json.dumps(analysis_data[:3] if isinstance(analysis_data, list) else [], ensure_ascii=False)[:2000]
                        })
                    else:
                        return json.dumps({"error": "Analysis completed but file not found"})
                else:
                    return json.dumps({
                        "error": f"Image analysis failed: {result.stderr}",
                        "stdout": result.stdout
                    })
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool("extract_supplements")
        def extract_supplements(query: str) -> str:
            """Extracts data from supplementary materials (Excel, Word, PDF).
            
            Args:
                query: Unused, but required by LangChain tool format.
            
            Returns:
                JSON string containing supplement extraction results.
            """
            try:
                supp_dir = self.supp_base_dir / pmc_id
                if not supp_dir.exists():
                    return json.dumps({
                        "success": True,
                        "message": "No supplementary materials found",
                        "excel_count": 0,
                        "word_count": 0,
                        "pdf_count": 0
                    })
                
                output_dir = self.supp_extracted_dir / pmc_id
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # batch_process_supplements.py 실행
                result = subprocess.run(
                    [
                        sys.executable, "batch_process_supplements.py",
                        "--input_dir", str(self.supp_base_dir),
                        "--output_dir", str(self.supp_extracted_dir),
                        "--pmc_id", pmc_id
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30분
                )
                
                if result.returncode == 0:
                    # 결과 확인
                    excel_dir = output_dir / "excel"
                    word_dir = output_dir / "word"
                    pdf_text_dir = output_dir / "pdf_text"
                    
                    excel_count = len(list(excel_dir.glob("*.json"))) if excel_dir.exists() else 0
                    word_count = len(list(word_dir.glob("*.json"))) if word_dir.exists() else 0
                    pdf_count = len(list(pdf_text_dir.glob("*.txt"))) if pdf_text_dir.exists() else 0
                    
                    return json.dumps({
                        "success": True,
                        "excel_files": excel_count,
                        "word_files": word_count,
                        "pdf_text_files": pdf_count,
                        "output_path": str(output_dir)
                    })
                else:
                    return json.dumps({
                        "error": f"Supplement extraction failed: {result.stderr}",
                        "stdout": result.stdout
                    })
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool("build_coreference")
        def build_coreference(query: str) -> str:
            """Builds a coreference dictionary by integrating data from supplements, images, and text.
            
            Args:
                query: Unused, but required by LangChain tool format.
            
            Returns:
                JSON string containing coreference dictionary.
            """
            try:
                output_dir = self.text_analyzed_dir / pmc_id
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # build_global_coreference.py 실행
                result = subprocess.run(
                    [
                        sys.executable, "build_global_coreference.py",
                        "--pmc_id", pmc_id,
                        "--use-gpt"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    coref_file = output_dir / "global_coreference_gpt.json"
                    if coref_file.exists():
                        with open(coref_file, 'r', encoding='utf-8') as f:
                            coref_data = json.load(f)
                        
                        group_count = len(coref_data.get('coreference_groups', {}))
                        rel_count = len(coref_data.get('relationships', []))
                        
                        return json.dumps({
                            "success": True,
                            "coreference_groups": group_count,
                            "relationships": rel_count,
                            "preview": json.dumps({k: v for k, v in list(coref_data.get('coreference_groups', {}).items())[:5]}, ensure_ascii=False)[:1000]
                        })
                    else:
                        return json.dumps({"error": "Coreference built but file not found"})
                else:
                    return json.dumps({
                        "error": f"Coreference building failed: {result.stderr}",
                        "stdout": result.stdout
                    })
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool("get_extracted_data")
        def get_extracted_data(query: str) -> str:
            """Gets all previously extracted data (text, images, supplements, coreference) for final extraction.
            
            Args:
                query: Unused, but required by LangChain tool format.
            
            Returns:
                JSON string containing summaries of all extracted data.
            """
            try:
                data_summary = {
                    "text": {},
                    "images": {},
                    "supplements": {},
                    "coreference": {}
                }
                
                # 텍스트 확인
                text_file = self.text_dir / pmc_id / "extracted_text.txt"
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    data_summary["text"] = {
                        "exists": True,
                        "length": len(text),
                        "preview": text[:2000]
                    }
                else:
                    data_summary["text"] = {"exists": False}
                
                # 이미지 분석 확인
                img_file = self.graph_analyzed_dir / pmc_id / "all_analyses.json"
                if img_file.exists():
                    with open(img_file, 'r', encoding='utf-8') as f:
                        img_data = json.load(f)
                    data_summary["images"] = {
                        "exists": True,
                        "count": len(img_data) if isinstance(img_data, list) else 0,
                        "preview": json.dumps(img_data[:2] if isinstance(img_data, list) else {}, ensure_ascii=False)[:1000]
                    }
                else:
                    data_summary["images"] = {"exists": False}
                
                # 보충자료 확인
                supp_dir = self.supp_extracted_dir / pmc_id
                if supp_dir.exists():
                    excel_files = list((supp_dir / "excel").glob("*.json")) if (supp_dir / "excel").exists() else []
                    word_files = list((supp_dir / "word").glob("*.json")) if (supp_dir / "word").exists() else []
                    
                    data_summary["supplements"] = {
                        "exists": True,
                        "excel_count": len(excel_files),
                        "word_count": len(word_files)
                    }
                else:
                    data_summary["supplements"] = {"exists": False}
                
                # Coreference 확인
                coref_file = self.text_analyzed_dir / pmc_id / "global_coreference_gpt.json"
                if coref_file.exists():
                    with open(coref_file, 'r', encoding='utf-8') as f:
                        coref_data = json.load(f)
                    data_summary["coreference"] = {
                        "exists": True,
                        "groups": len(coref_data.get('coreference_groups', {})),
                        "relationships": len(coref_data.get('relationships', []))
                    }
                else:
                    data_summary["coreference"] = {"exists": False}
                
                return json.dumps(data_summary, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        return [
            extract_text,
            extract_images,
            analyze_images,
            extract_supplements,
            build_coreference,
            get_extracted_data
        ]
    
    def _create_agent(self):
        """ReAct Agent 생성"""
        prompt = """You are a specialized assistant for extracting and structuring comprehensive ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) data from scientific papers.

Your task is to:
1. Extract text from the main article PDF
2. Extract and analyze figures/tables from the article
3. Extract data from supplementary materials (Excel, Word, PDF)
4. Build a coreference dictionary to identify compound aliases
5. Finally, extract all compounds and their attributes/ADMET indicators

Use the available tools to:
- extract_text: Extract text from the article PDF
- extract_images: Extract figures and tables using YOLO
- analyze_images: Analyze extracted images with GPT-4o
- extract_supplements: Process supplementary materials
- build_coreference: Create coreference dictionary
- get_extracted_data: Get summaries of all extracted data

After gathering all necessary data, extract compounds and their attributes. Do NOT filter by ADMET keywords - extract everything. ADMET filtering will be done manually later.

Output your final answer as structured JSON with all compounds and their attributes found across all sources."""
        
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={"prefix": prompt, "seed": 42},
        )
        
        return agent
    
    def extract(self) -> Dict[str, Any]:
        """ReAct Agent를 사용하여 ADMET 데이터 추출"""
        logger.info(f"ReAct Agent 시작: {self.pmc_id}")
        
        user_prompt = f"""
Extract all compounds and their attributes from paper {self.pmc_id}.

Steps:
1. Extract text from the article PDF
2. Extract and analyze images (figures/tables)
3. Extract data from supplementary materials
4. Build coreference dictionary
5. Extract all compounds with all their attributes (do not filter by ADMET keywords)

Return the final structured JSON result with all compounds found.
"""
        
        try:
            response = self.agent.run(user_prompt)
            
            # 결과 저장
            output_dir = self.final_dir / self.pmc_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 응답을 JSON으로 파싱 시도
            try:
                # JSON 부분 추출
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_text = response[json_start:json_end].strip()
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    json_text = response[json_start:json_end].strip()
                else:
                    json_text = response.strip()
                
                result = json.loads(json_text)
            except:
                # JSON 파싱 실패 시 전체 응답 저장
                result = {
                    "schema_version": "2.0",
                    "created_at": datetime.now().isoformat(),
                    "pmc_id": self.pmc_id,
                    "raw_response": response,
                    "records": []
                }
            
            output_file = output_dir / f"{self.pmc_id}_final_admet_react.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과 저장: {output_file}")
            return result
            
        except Exception as e:
            logger.error(f"ReAct Agent 실행 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "schema_version": "2.0",
                "created_at": datetime.now().isoformat(),
                "pmc_id": self.pmc_id,
                "error": str(e),
                "records": []
            }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ReAct Agent 기반 최종 ADMET 데이터 추출")
    parser.add_argument("--pmc_id", required=True, help="PMC ID (예: PMC7066191)")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (data_test 폴더 사용)")
    
    args = parser.parse_args()
    
    extractor = ReactADMETExtractor(args.pmc_id, test_mode=args.test)
    result = extractor.extract()
    
    print(f"✅ ReAct Agent 추출 완료!")
    print(f"  PMC ID: {result.get('pmc_id', 'N/A')}")
    print(f"  레코드 수: {len(result.get('records', []))}")
    print(f"  스키마 버전: {result.get('schema_version', 'N/A')}")


if __name__ == "__main__":
    main()

