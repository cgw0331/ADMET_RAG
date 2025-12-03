#!/usr/bin/env python3
"""
YOLO로 추출한 표/그림 이미지를 GPT-4o Vision으로 분석하여 화합물 정보 추출
- supp_extracted/PMC###/pdf_graph/###/figures/, tables/ 이미지 분석
- 화합물 ID (CBK), SMILES, Well Position 등 추출
- JSON Lines 형식으로 통합 저장
"""

import json
import logging
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os
import io

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOImageAnalyzer:
    """YOLO로 추출한 이미지를 GPT-4o Vision으로 분석"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
    
    def find_extracted_images(self, pdf_graph_dir: Path) -> List[Dict[str, Any]]:
        """YOLO로 추출된 figures/와 tables/ 폴더의 이미지 찾기"""
        images = []
        
        # figures 폴더 찾기
        figures_dir = pdf_graph_dir / "figures"
        if figures_dir.exists() and figures_dir.is_dir():
            for png_file in sorted(figures_dir.glob("*.png")):
                try:
                    img = Image.open(png_file)
                    images.append({
                        'image': img,
                        'filename': png_file.name,
                        'class': 'figure',
                        'file_path': str(png_file)
                    })
                    logger.debug(f"Figure 발견: {png_file.name}")
                except Exception as e:
                    logger.warning(f"Figure 로드 실패 {png_file}: {e}")
        
        # tables 폴더 찾기
        tables_dir = pdf_graph_dir / "tables"
        if tables_dir.exists() and tables_dir.is_dir():
            for png_file in sorted(tables_dir.glob("*.png")):
                try:
                    img = Image.open(png_file)
                    images.append({
                        'image': img,
                        'filename': png_file.name,
                        'class': 'table',
                        'file_path': str(png_file)
                    })
                    logger.debug(f"Table 발견: {png_file.name}")
                except Exception as e:
                    logger.warning(f"Table 로드 실패 {png_file}: {e}")
        
        logger.info(f"총 {len(images)}개 이미지 발견 (Figures: {sum(1 for x in images if x['class']=='figure')}, Tables: {sum(1 for x in images if x['class']=='table')})")
        return images
    
    def analyze_image(self, image_data: Dict[str, Any], 
                     previous_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """단일 이미지를 GPT-4o Vision으로 분석하여 화합물 정보 추출
        
        Args:
            image_data: 이미지 데이터 (image, class, filename)
            previous_context: 이전 단계에서 누적된 컨텍스트 (선택적)
        """
        image = image_data['image']
        obj_class = image_data['class']
        filename = image_data['filename']
        
        # 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 이전 컨텍스트 포맷팅
        context_section = ""
        if previous_context and previous_context.get("compounds"):
            compounds = previous_context["compounds"]
            compound_list = list(compounds.keys())[:50]  # 최대 50개
            context_section = f"""

**이전 단계에서 발견된 화합물들 (참고용):**
{', '.join(compound_list) if compound_list else "None"}
- 이 화합물들이 이미지에 나타나면 기존 정보와 매칭하세요
- 새로운 화합물도 추가로 추출하세요
"""
        
        prompt = f"""이 이미지는 과학 논문 보충자료의 {obj_class}입니다 (파일명: {filename}).
{context_section}
**작업:**
이 {obj_class}에서 **모든 화합물 정보**를 추출하세요.

**추출 형식 (JSON Lines):**
각 레코드를 한 줄로 출력:
{{"compound_name": "화합물ID/이름", "indicator_name": "속성명", "value": "값", "unit": "", "source": "{filename}"}}

**중요 지시사항:**
1. {obj_class}의 **모든 행(row)**을 빠짐없이 분석
2. 화합물 ID (예: CBK037537, CBK093726, CBK074456 등) 추출
3. Well 위치 (예: "2 B01", "2 C01", "1 A01" 등) 추출
4. SMILES 구조식 추출
5. ADMET 필터링하지 말고 **모든 속성** 포함
6. 표 캡션/제목도 source에 포함 가능하면 포함

**예시:**
표에 "2 B01 CBK037537 NC1=CC(=CC=C1)C1=NC=CN=C1NCC1=CC=CS1" 이 있으면:
{{"compound_name": "CBK037537", "indicator_name": "Well Position", "value": "2 B01", "unit": "", "source": "{filename}"}}
{{"compound_name": "CBK037537", "indicator_name": "SMILES", "value": "NC1=CC(=CC=C1)C1=NC=CN=C1NCC1=CC=CS1", "unit": "", "source": "{filename}"}}

JSON Lines 형식으로만 출력 (각 레코드가 한 줄).
화합물이 없으면 빈 결과를 반환하세요."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                temperature=0,
                max_tokens=4000
            )
            
            generated_text = response.choices[0].message.content
            records = self._parse_jsonl(generated_text, filename)
            
            if not records:
                logger.debug(f"{filename}: 화합물 정보 없음")
            
            return records
                    
        except Exception as e:
            logger.error(f"{filename} 분석 실패: {e}")
            return []
    
    def _analyze_image_with_history(self, image_data: Dict[str, Any], 
                                   messages: List[Dict[str, Any]],
                                   image_index: int, total_images: int) -> List[Dict[str, Any]]:
        """대화 히스토리를 유지하면서 이미지 분석 (ChatGPT처럼 컨텍스트 이어짐)"""
        image = image_data['image']
        obj_class = image_data['class']
        filename = image_data['filename']
        
        # 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 사용자 메시지 (이미지 포함)
        user_prompt = f"""이미지 {image_index}/{total_images}: 과학 논문 보충자료의 {obj_class} (파일명: {filename})

**작업:**
이 {obj_class}에서 **모든 화합물 정보**를 추출하세요.

**중요:**
- 이전 이미지들에서 발견된 화합물과 일치하는 경우, 동일한 이름을 사용하세요
- 새로운 화합물도 추가로 추출하세요
- {obj_class}의 **모든 행(row)**을 빠짐없이 분석

**추출 형식 (JSON Lines):**
각 레코드를 한 줄로 출력:
{{"compound_name": "화합물ID/이름", "indicator_name": "속성명", "value": "값", "unit": "", "source": "{filename}"}}

**예시:**
표에 "2 B01 CBK037537 NC1=CC(=CC=C1)C1=NC=CN=C1NCC1=CC=CS1" 이 있으면:
{{"compound_name": "CBK037537", "indicator_name": "Well Position", "value": "2 B01", "unit": "", "source": "{filename}"}}
{{"compound_name": "CBK037537", "indicator_name": "SMILES", "value": "NC1=CC(=CC=C1)C1=NC=CN=C1NCC1=CC=CS1", "unit": "", "source": "{filename}"}}

JSON Lines 형식으로만 출력 (각 레코드가 한 줄).
화합물이 없으면 빈 결과를 반환하세요."""
        
        # 사용자 메시지 추가
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        })
        
        try:
            # 대화 히스토리를 포함하여 API 호출
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,  # 이전 대화 포함
                temperature=0,
                max_tokens=4000
            )
            
            generated_text = response.choices[0].message.content
            records = self._parse_jsonl(generated_text, filename)
            
            # Assistant 응답을 대화 히스토리에 추가 (다음 이미지 분석 시 참조)
            messages.append({
                "role": "assistant",
                "content": generated_text
            })
            
            if not records:
                logger.debug(f"{filename}: 화합물 정보 없음")
            
            return records
                    
        except Exception as e:
            logger.error(f"{filename} 분석 실패: {e}")
            return []
    
    def _parse_jsonl(self, text: str, source: str) -> List[Dict[str, Any]]:
        """JSON Lines 파싱"""
        records = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            if line.startswith('```'):
                continue
            if '```json' in line:
                line = line.replace('```json', '').strip()
            if line.endswith('```'):
                line = line[:-3].strip()
            
            try:
                record = json.loads(line)
                # source 필드 보정
                if 'source' not in record or not record['source']:
                    record['source'] = source
                elif source not in record['source']:
                    record['source'] = f"{source}/{record['source']}"
                
                # 필수 필드 확인
                if "compound_name" in record and "indicator_name" in record:
                    records.append(record)
            except json.JSONDecodeError:
                continue
        
        return records
    
    def analyze_pdf_graph(self, pdf_graph_dir: Path, 
                         previous_context: Optional[Dict[str, Any]] = None,
                         use_conversation_history: bool = True) -> Dict[str, Any]:
        """PDF 그래프 폴더의 모든 이미지 분석 (대화 히스토리 유지)
        
        Args:
            pdf_graph_dir: PDF 그래프 디렉토리
            previous_context: 이전 단계에서 누적된 컨텍스트 (선택적)
            use_conversation_history: 대화 히스토리 유지 여부 (True면 ChatGPT처럼 컨텍스트 이어짐)
        """
        logger.info(f"이미지 분석 시작: {pdf_graph_dir}")
        
        # 1. 이미지 찾기
        images = self.find_extracted_images(pdf_graph_dir)
        
        if not images:
            logger.warning("분석할 이미지가 없습니다.")
            return {
                "compounds": [],
                "summary": {
                    "total_compounds": 0,
                    "total_attributes": 0,
                    "total_records": 0,
                    "images_processed": 0
                },
                "raw_jsonl": []
            }
        
        # 2. 대화 히스토리 초기화 (시스템 메시지 + 이전 컨텍스트)
        messages = []
        
        # 시스템 메시지: 이전 단계에서 발견된 화합물 정보 포함
        system_content = """You are an expert data extractor for biomedical ADMET compounds.
Extract ALL compounds and their attributes from each image.
Output in JSON Lines format: {"compound_name": "...", "indicator_name": "...", "value": "...", "unit": "", "source": "..."}"""
        
        if previous_context and previous_context.get("compounds"):
            compounds = previous_context["compounds"]
            compound_list = list(compounds.keys())[:50]  # 최대 50개
            if compound_list:
                system_content += f"""

**Previously discovered compounds from other sources (for reference):**
{', '.join(compound_list)}
- If these compounds appear in the images, match them with existing information
- Also extract any NEW compounds found in the images
- Maintain consistency in compound naming across all images"""
        
        messages.append({"role": "system", "content": system_content})
        
        # 3. 각 이미지 분석 (대화 히스토리 유지)
        all_records = []
        logger.info(f"총 {len(images)}개 이미지 분석 시작...")
        if previous_context:
            prev_compounds = len(previous_context.get("compounds", {}))
            logger.info(f"  이전 단계에서 발견된 화합물: {prev_compounds}개 (맥락 참조)")
        if use_conversation_history:
            logger.info(f"  💬 대화 히스토리 유지 모드: 각 이미지 분석이 이전 대화를 기억합니다")
        
        for i, image_data in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}] {image_data['class']} 분석 중: {image_data['filename']}...")
            try:
                # 대화 히스토리를 사용하는 경우
                if use_conversation_history:
                    records = self._analyze_image_with_history(image_data, messages, i, len(images))
                else:
                    # 기존 방식 (독립적 호출)
                    records = self.analyze_image(image_data, previous_context=previous_context)
                
                all_records.extend(records)
                if records:
                    logger.info(f"  ✅ {len(records)}개 레코드 추출")
                else:
                    logger.info(f"  ⏭️  화합물 정보 없음 (건너뜀)")
            except Exception as e:
                logger.error(f"  ❌ 분석 실패: {e}")
                continue
            
            # 진행률 표시 (10개마다)
            if i % 10 == 0:
                logger.info(f"진행률: {i}/{len(images)} ({i*100//len(images)}%), 현재까지 {len(all_records)}개 레코드 추출됨")
        
        # 3. 화합물별로 그룹화 및 중복제거
        compounds_dict = {}
        seen = set()
        
        for record in all_records:
            comp_name = record.get("compound_name", "").strip()
            indicator = record.get("indicator_name", "").strip()
            value = record.get("value", "").strip()
            
            if not comp_name or not indicator or not value:
                continue
            
            dup_key = (comp_name.lower(), indicator.lower(), value.lower())
            if dup_key in seen:
                continue
            seen.add(dup_key)
            
            if comp_name not in compounds_dict:
                compounds_dict[comp_name] = {
                    "compound_name": comp_name,
                    "attributes": {},
                    "aliases": []
                }
            
            if indicator in compounds_dict[comp_name]["attributes"]:
                existing = compounds_dict[comp_name]["attributes"][indicator]
                if isinstance(existing.get("value"), list):
                    if value not in existing["value"]:
                        existing["value"].append(value)
                        existing["source"] = f"{existing['source']}, {record.get('source', '')}"
                else:
                    if existing.get("value") != value:
                        compounds_dict[comp_name]["attributes"][indicator] = {
                            "value": [existing["value"], value],
                            "unit": existing.get("unit", "") or record.get("unit", ""),
                            "source": f"{existing['source']}, {record.get('source', '')}"
                        }
                    else:
                        existing["source"] = f"{existing['source']}, {record.get('source', '')}"
            else:
                compounds_dict[comp_name]["attributes"][indicator] = {
                    "value": value,
                    "unit": record.get("unit", ""),
                    "source": record.get("source", "")
                }
        
        compounds = list(compounds_dict.values())
        total_attrs = sum(len(c.get("attributes", {})) for c in compounds)
        
        result = {
            "compounds": compounds,
            "summary": {
                "total_compounds": len(compounds),
                "total_attributes": total_attrs,
                "total_records": len(all_records),
                "images_processed": len(images)
            },
            "raw_jsonl": all_records
        }
        
        logger.info(f"추출 완료: 화합물 {len(compounds)}개, 총 속성 {total_attrs}개, 레코드 {len(all_records)}개, 이미지 {len(images)}개")
        
        return result
    
    def process_pdf_graph(self, pdf_graph_dir: Path, output_dir: Path, 
                        previous_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """PDF 그래프 분석 및 결과 저장
        
        Args:
            pdf_graph_dir: PDF 그래프 디렉토리
            output_dir: 출력 디렉토리
            previous_context: 이전 단계에서 누적된 컨텍스트 (선택적)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 분석 (이전 컨텍스트 전달)
        result = self.analyze_pdf_graph(pdf_graph_dir, previous_context=previous_context)
        
        # 결과 저장
        pdf_name = pdf_graph_dir.name
        output_file = output_dir / f"{pdf_name}_yolo_gpt_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과 저장: {output_file}")
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO로 추출한 이미지를 GPT-4o Vision으로 분석")
    parser.add_argument("--pdf_graph_dir", required=True, help="pdf_graph 폴더 경로 (예: supp_extracted/PMC7066191/pdf_graph/41467_2020_15111_MOESM1_ESM)")
    parser.add_argument("--output_dir", help="출력 디렉토리 (기본값: pdf_graph_dir 상위 폴더/pdf_info)")
    parser.add_argument("--pmc_id", help="PMC ID (출력 경로 지정용)")
    
    args = parser.parse_args()
    
    pdf_graph_dir = Path(args.pdf_graph_dir)
    if not pdf_graph_dir.exists():
        print(f"❌ PDF 그래프 폴더 없음: {pdf_graph_dir}")
        return
    
    # 출력 디렉토리 결정
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # supp_extracted/PMC###/pdf_info/ 형태로 저장
        if args.pmc_id:
            output_dir = pdf_graph_dir.parent.parent / "pdf_info"
        else:
            output_dir = pdf_graph_dir.parent / "pdf_info"
    
    analyzer = YOLOImageAnalyzer()
    result = analyzer.process_pdf_graph(pdf_graph_dir, output_dir)
    
    compound_count = len(result.get("compounds", []))
    images_processed = result.get("summary", {}).get("images_processed", 0)
    print(f"✅ 이미지 분석 완료!")
    print(f"  폴더: {pdf_graph_dir.name}")
    print(f"  화합물 수: {compound_count}")
    print(f"  처리된 이미지: {images_processed}개")
    print(f"  결과: {output_dir}")


if __name__ == "__main__":
    main()

