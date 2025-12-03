#!/usr/bin/env python3
"""
본문 이미지 분석 배치 스크립트
- graph_extracted/PMC###/article/figures/, tables/ 아래 이미지 분석
- vision_agent_admet_llama.py로 분석 (GPT 모드)
- 결과를 graph_analyzed/ 폴더에 저장
"""

import os
import sys
from pathlib import Path
# Ensure repo root is on sys.path when running from preprocess/
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from vision_agent_admet_llama import ADMETVisionAgentLlama
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_all_images(article_dir: Path) -> List[Dict[str, Any]]:
    """
    article 폴더 아래 figures/와 tables/ 폴더 안의 이미지만 찾기
    """
    images = []
    
    # figures 폴더 찾기
    figures_dir = article_dir / "figures"
    if figures_dir.exists() and figures_dir.is_dir():
        for png_file in figures_dir.glob("*.png"):
            try:
                img = Image.open(png_file)
                images.append({
                    'image': img,
                    'filename': png_file.name,
                    'class': 'figure',
                    'confidence': 1.0,
                    'file_path': str(png_file),
                    'relative_path': str(png_file.relative_to(article_dir.parent))
                })
                logger.debug(f"Figure 발견: {png_file.relative_to(article_dir.parent)}")
            except Exception as e:
                logger.warning(f"이미지 로드 실패 {png_file}: {e}")
    
    # tables 폴더 찾기
    tables_dir = article_dir / "tables"
    if tables_dir.exists() and tables_dir.is_dir():
        for png_file in tables_dir.glob("*.png"):
            try:
                img = Image.open(png_file)
                images.append({
                    'image': img,
                    'filename': png_file.name,
                    'class': 'table',
                    'confidence': 1.0,
                    'file_path': str(png_file),
                    'relative_path': str(png_file.relative_to(article_dir.parent))
                })
                logger.debug(f"Table 발견: {png_file.relative_to(article_dir.parent)}")
            except Exception as e:
                logger.warning(f"이미지 로드 실패 {png_file}: {e}")
    
    return images


def analyze_pmc_main_images(pmc_id: str, 
                           agent: ADMETVisionAgentLlama,
                           base_dir: Path = Path("graph_extracted")) -> Dict[str, Any]:
    """
    특정 PMC의 본문 이미지 분석
    """
    logger.info(f"본문 이미지 분석 시작: {pmc_id}")
    
    article_dir = base_dir / pmc_id / "article"
    if not article_dir.exists():
        logger.warning(f"article 폴더 없음: {article_dir}")
        return {"pmc_id": pmc_id, "status": "no_article", "images_found": 0}
    
    # 모든 이미지 찾기
    images = find_all_images(article_dir)
    
    if not images:
        logger.warning(f"이미지 없음: {article_dir}")
        return {"pmc_id": pmc_id, "status": "no_images", "images_found": 0}
    
    logger.info(f"  발견된 이미지: {len(images)}개")
    
    # 출력 디렉토리
    output_dir = Path("graph_analyzed") / pmc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 분석 결과 저장
    figure_analyses = []
    table_analyses = []
    all_results = []
    errors = []
    
    # 각 이미지 분석
    for i, image_data in enumerate(images):
        logger.info(f"  [{i+1}/{len(images)}] 분석 중: {image_data['filename']} ({image_data['class']})")
        
        try:
            analysis = agent.analyze_figure_table_llama(image_data)
            
            result = {
                "filename": image_data['filename'],
                "relative_path": image_data['relative_path'],
                "class": image_data['class'],
                "confidence": image_data['confidence'],
                "analysis": analysis,
                "file_path": image_data['file_path']
            }
            
            all_results.append(result)
            
            if image_data['class'] == 'figure':
                figure_analyses.append(result)
            elif image_data['class'] == 'table':
                table_analyses.append(result)
            
        except Exception as e:
            logger.error(f"  이미지 {i+1} 분석 실패: {e}")
            errors.append({
                "filename": image_data['filename'],
                "relative_path": image_data['relative_path'],
                "error": str(e)
            })
    
    # 결과 저장
    results = {
        "pmc_id": pmc_id,
        "status": "completed",
        "total_images": len(images),
        "figures": len(figure_analyses),
        "tables": len(table_analyses),
        "errors": len(errors)
    }
    
    # figure_analyses.json
    if figure_analyses:
        with open(output_dir / "figure_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(figure_analyses, f, ensure_ascii=False, indent=2)
        logger.info(f"  Figure 분석 결과 저장: {len(figure_analyses)}개")
    
    # table_analyses.json
    if table_analyses:
        with open(output_dir / "table_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(table_analyses, f, ensure_ascii=False, indent=2)
        logger.info(f"  Table 분석 결과 저장: {len(table_analyses)}개")
    
    # all_analyses.json (전체)
    if all_results:
        with open(output_dir / "all_analyses.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # errors.json (에러 기록)
    if errors:
        with open(output_dir / "errors.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        logger.warning(f"  에러 발생: {len(errors)}개")
    
    # summary.json
    results["summary"] = agent._generate_summary({f"image_{i}": r for i, r in enumerate(all_results)})
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ {pmc_id} 분석 완료: Figures {len(figure_analyses)}개, Tables {len(table_analyses)}개")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="본문 이미지 분석 배치 스크립트")
    parser.add_argument("--pmc_id", help="특정 PMC ID만 분석 (없으면 전체)")
    parser.add_argument("--base_dir", default="graph_extracted", help="기본 디렉토리")
    parser.add_argument("--model", default="gpt-4o", help="모델 이름 (기본값: gpt-4o)")
    parser.add_argument("--use-gpt", action="store_true", default=True, help="GPT-4o 사용 (기본값: True)")
    parser.add_argument("--coreference_dict", help="Coreference dictionary 경로 (선택적)")
    parser.add_argument("--limit", type=int, help="처리할 최대 PMC 개수")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스")
    parser.add_argument("--skip-completed", action="store_true", help="이미 완료된 PMC 건너뛰기")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Vision Agent 초기화 (GPT 기본값)
    agent = ADMETVisionAgentLlama(
        model_name=args.model,
        coreference_dict_path=args.coreference_dict,
        use_gpt=args.use_gpt if hasattr(args, 'use_gpt') else True
    )
    
    # PMC 폴더 찾기
    if args.pmc_id:
        pmc_folders = [base_dir / args.pmc_id]
    else:
        pmc_folders = sorted([d for d in base_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('PMC')])
    
    total = len(pmc_folders)
    logger.info(f"총 {total}개 PMC 폴더 발견")
    
    # 시작 인덱스부터
    pmc_folders = pmc_folders[args.start:]
    if args.limit:
        pmc_folders = pmc_folders[:args.limit]
    
    success_count = 0
    skip_count = 0
    
    for i, pmc_folder in enumerate(pmc_folders, args.start + 1):
        pmc_id = pmc_folder.name
        
        # 이미 완료되었는지 확인
        if args.skip_completed:
            info_dir = Path("graph_analyzed") / pmc_id
            if info_dir.exists() and (info_dir / "summary.json").exists():
                logger.info(f"[{i}/{total}] {pmc_id}: 이미 완료됨, 건너뜀")
                skip_count += 1
                continue
        
        logger.info(f"[{i}/{total}] {pmc_id} 처리 중...")
        
        try:
            result = analyze_pmc_main_images(pmc_id, agent, base_dir)
            if result.get("status") == "completed":
                success_count += 1
        except Exception as e:
            logger.error(f"[{i}/{total}] {pmc_id} 처리 실패: {e}")
    
    logger.info(f"\n✅ 전체 분석 완료!")
    logger.info(f"  성공: {success_count}개")
    logger.info(f"  건너뜀: {skip_count}개")
    logger.info(f"  실패: {len(pmc_folders) - success_count - skip_count}개")


if __name__ == "__main__":
    main()



