#!/usr/bin/env python3
"""
YOLO로 추출한 보충자료 이미지를 배치로 GPT-4o Vision으로 분석
- supp_extracted/PMC###/pdf_graph/ 아래 모든 PDF 폴더 처리
- 각 PDF별로 figures/와 tables/ 이미지 분석
- pdf_gpt_yolo/{pdf_name}_yolo_gpt_analysis.json 형태로 저장
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from analyze_yolo_extracted_images import YOLOImageAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_pdf_graph_folders(base_dir: Path, pmc_id: str = None) -> List[Path]:
    """pdf_graph 폴더 아래의 모든 PDF 폴더 찾기"""
    pdf_folders = []
    
    if pmc_id:
        pdf_graph_dir = base_dir / pmc_id / "pdf_graph"
    else:
        pdf_graph_dir = base_dir
    
    if not pdf_graph_dir.exists():
        return []
    
    # pdf_graph 아래의 모든 폴더 (PDF별)
    for folder in pdf_graph_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith('page_'):
            # figures/ 또는 tables/ 폴더가 있는지 확인
            if (folder / "figures").exists() or (folder / "tables").exists():
                pdf_folders.append(folder)
    
    return sorted(pdf_folders)


def process_pmc(pmc_id: str, base_dir: Path = Path("supp_extracted"), 
                analyzer: YOLOImageAnalyzer = None, 
                skip_completed: bool = False) -> Dict[str, Any]:
    """특정 PMC의 모든 PDF 그래프 처리"""
    logger.info(f"PMC {pmc_id} 처리 시작...")
    
    pdf_graph_dir = base_dir / pmc_id / "pdf_graph"
    if not pdf_graph_dir.exists():
        logger.warning(f"pdf_graph 폴더 없음: {pdf_graph_dir}")
        return {"pmc_id": pmc_id, "status": "no_pdf_graph", "processed": 0}
    
    # PDF 폴더 찾기
    pdf_folders = find_pdf_graph_folders(base_dir, pmc_id)
    
    if not pdf_folders:
        logger.warning(f"처리할 PDF 폴더 없음: {pdf_graph_dir}")
        return {"pmc_id": pmc_id, "status": "no_pdfs", "processed": 0}
    
    logger.info(f"  발견된 PDF 폴더: {len(pdf_folders)}개")
    
    # 출력 디렉토리
    output_dir = base_dir / pmc_id / "pdf_gpt_yolo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyzer 초기화
    if analyzer is None:
        analyzer = YOLOImageAnalyzer()
    
    processed = 0
    skipped = 0
    errors = []
    
    for pdf_folder in pdf_folders:
        pdf_name = pdf_folder.name
        output_file = output_dir / f"{pdf_name}_yolo_gpt_analysis.json"
        
        # 이미 완료되었는지 확인
        if skip_completed and output_file.exists():
            logger.info(f"  [{processed+skipped+1}/{len(pdf_folders)}] {pdf_name}: 이미 완료됨, 건너뜀")
            skipped += 1
            continue
        
        logger.info(f"  [{processed+skipped+1}/{len(pdf_folders)}] {pdf_name} 분석 중...")
        
        try:
            result = analyzer.process_pdf_graph(pdf_folder, output_dir)
            processed += 1
            logger.info(f"  ✅ {pdf_name}: 화합물 {len(result.get('compounds', []))}개, 이미지 {result.get('summary', {}).get('images_processed', 0)}개")
        except Exception as e:
            logger.error(f"  ❌ {pdf_name} 처리 실패: {e}")
            errors.append({"pdf_name": pdf_name, "error": str(e)})
    
    return {
        "pmc_id": pmc_id,
        "status": "completed",
        "total_pdfs": len(pdf_folders),
        "processed": processed,
        "skipped": skipped,
        "errors": errors
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO로 추출한 보충자료 이미지 배치 분석")
    parser.add_argument("--pmc_id", help="특정 PMC ID만 처리 (없으면 전체)")
    parser.add_argument("--base_dir", default="supp_extracted", help="기본 디렉토리")
    parser.add_argument("--limit", type=int, help="처리할 최대 PMC 개수")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스")
    parser.add_argument("--skip-completed", action="store_true", help="이미 완료된 PDF 건너뛰기")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Analyzer 초기화
    analyzer = YOLOImageAnalyzer()
    
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
    total_processed = 0
    total_skipped = 0
    
    for i, pmc_folder in enumerate(pmc_folders, args.start + 1):
        pmc_id = pmc_folder.name
        
        logger.info(f"\n[{i}/{total}] PMC {pmc_id} 처리 중...")
        
        try:
            result = process_pmc(pmc_id, base_dir, analyzer, args.skip_completed)
            
            if result.get("status") == "completed":
                success_count += 1
                total_processed += result.get("processed", 0)
                total_skipped += result.get("skipped", 0)
                
                if result.get("errors"):
                    logger.warning(f"  {len(result['errors'])}개 PDF 처리 실패")
        except Exception as e:
            logger.error(f"[{i}/{total}] PMC {pmc_id} 처리 실패: {e}")
    
    logger.info(f"\n✅ 전체 분석 완료!")
    logger.info(f"  성공한 PMC: {success_count}개")
    logger.info(f"  처리된 PDF: {total_processed}개")
    logger.info(f"  건너뛴 PDF: {total_skipped}개")


if __name__ == "__main__":
    main()



