#!/usr/bin/env python3
"""
보충자료 PDF를 YOLO로 figure/table 추출하고 GPT-4o Vision으로 분석
- data_test/supp/PMC###/ 아래 PDF 파일 처리
- YOLO로 figure/table 추출 → supp_extracted/PMC###/pdf_graph/{pdf_name}/
- GPT-4o Vision으로 분석 → supp_extracted/PMC###/pdf_gpt_yolo/{pdf_name}_yolo_gpt_analysis.json
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_pdf_files(supp_dir: Path, pmc_id: str) -> List[Path]:
    """보충자료 폴더에서 PDF 파일 찾기"""
    pmc_supp_dir = supp_dir / pmc_id
    if not pmc_supp_dir.exists():
        return []
    
    pdf_files = list(pmc_supp_dir.glob("*.pdf"))
    return sorted(pdf_files)


def extract_figures_tables(pdf_path: Path, output_dir: Path, 
                           model_path: Optional[str] = None) -> bool:
    """YOLO로 PDF에서 figure/table 추출"""
    logger.info(f"YOLO 추출 시작: {pdf_path.name}")
    
    # inference_yolo.py 실행
    cmd = [
        sys.executable, "inference_yolo.py",
        "--pdf", str(pdf_path),
        "--output", str(output_dir),
        "--confidence", "0.25"
    ]
    
    if model_path:
        cmd.extend(["--model", model_path])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode == 0:
            logger.info(f"✅ YOLO 추출 완료: {pdf_path.name}")
            return True
        else:
            logger.error(f"❌ YOLO 추출 실패: {pdf_path.name}")
            logger.error(f"  stderr: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ YOLO 추출 타임아웃: {pdf_path.name}")
        return False
    except Exception as e:
        logger.error(f"❌ YOLO 추출 오류: {pdf_path.name} - {e}")
        return False


def analyze_extracted_images(pdf_graph_dir: Path, output_file: Path) -> bool:
    """추출된 figure/table을 GPT-4o Vision으로 분석"""
    logger.info(f"GPT-4o Vision 분석 시작: {output_file.name}")
    
    from analyze_yolo_extracted_images import YOLOImageAnalyzer
    
    try:
        analyzer = YOLOImageAnalyzer()
        result = analyzer.process_pdf_graph(pdf_graph_dir, output_file.parent)
        
        compounds_count = len(result.get('compounds', []))
        images_count = result.get('summary', {}).get('images_processed', 0)
        
        logger.info(f"✅ 분석 완료: 화합물 {compounds_count}개, 이미지 {images_count}개")
        return True
    except Exception as e:
        logger.error(f"❌ GPT-4o Vision 분석 실패: {e}")
        return False


def process_pdf(pdf_path: Path, pmc_id: str, base_dir: Path = Path("data_test"),
                skip_completed: bool = False) -> Dict[str, Any]:
    """단일 PDF 처리"""
    pdf_name = pdf_path.stem  # 확장자 제거
    
    # 출력 디렉토리
    supp_extracted_dir = base_dir / "supp_extracted" / pmc_id
    pdf_graph_dir = supp_extracted_dir / "pdf_graph" / pdf_name
    pdf_gpt_yolo_dir = supp_extracted_dir / "pdf_gpt_yolo"
    output_file = pdf_gpt_yolo_dir / f"{pdf_name}_yolo_gpt_analysis.json"
    
    # 이미 완료되었는지 확인
    if skip_completed and output_file.exists():
        logger.info(f"  ⏭️  {pdf_name}: 이미 완료됨, 건너뜀")
        return {
            "pdf_name": pdf_name,
            "status": "skipped",
            "extracted": True,
            "analyzed": True
        }
    
    logger.info(f"  📄 {pdf_name} 처리 시작...")
    
    # 1. YOLO로 figure/table 추출
    extracted = False
    if pdf_graph_dir.exists() and any((pdf_graph_dir / "figures").iterdir()) or any((pdf_graph_dir / "tables").iterdir()):
        logger.info(f"  ✅ 이미 추출됨: {pdf_name}")
        extracted = True
    else:
        pdf_graph_dir.mkdir(parents=True, exist_ok=True)
        extracted = extract_figures_tables(pdf_path, pdf_graph_dir)
    
    if not extracted:
        return {
            "pdf_name": pdf_name,
            "status": "extraction_failed",
            "extracted": False,
            "analyzed": False
        }
    
    # 2. GPT-4o Vision으로 분석
    pdf_gpt_yolo_dir.mkdir(parents=True, exist_ok=True)
    analyzed = analyze_extracted_images(pdf_graph_dir, output_file)
    
    return {
        "pdf_name": pdf_name,
        "status": "completed" if analyzed else "analysis_failed",
        "extracted": extracted,
        "analyzed": analyzed
    }


def process_pmc(pmc_id: str, base_dir: Path = Path("data_test"),
                skip_completed: bool = False) -> Dict[str, Any]:
    """특정 PMC의 모든 보충자료 PDF 처리"""
    logger.info(f"PMC {pmc_id} 처리 시작...")
    
    supp_dir = base_dir / "supp"
    pdf_files = find_pdf_files(supp_dir, pmc_id)
    
    if not pdf_files:
        logger.warning(f"  ⚠️  PDF 파일 없음: {supp_dir / pmc_id}")
        return {
            "pmc_id": pmc_id,
            "status": "no_pdfs",
            "processed": 0,
            "total": 0
        }
    
    logger.info(f"  발견된 PDF: {len(pdf_files)}개")
    
    results = []
    for pdf_path in pdf_files:
        result = process_pdf(pdf_path, pmc_id, base_dir, skip_completed)
        results.append(result)
    
    processed = sum(1 for r in results if r.get("status") == "completed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    
    return {
        "pmc_id": pmc_id,
        "status": "completed",
        "total": len(pdf_files),
        "processed": processed,
        "skipped": skipped,
        "results": results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="보충자료 PDF YOLO 추출 및 GPT-4o Vision 분석")
    parser.add_argument("--pmc_id", help="특정 PMC ID만 처리")
    parser.add_argument("--base_dir", default="data_test", help="기본 디렉토리")
    parser.add_argument("--skip-completed", action="store_true", help="이미 완료된 PDF 건너뛰기")
    parser.add_argument("--limit", type=int, help="처리할 최대 PMC 개수")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    supp_dir = base_dir / "supp"
    
    if not supp_dir.exists():
        logger.error(f"보충자료 폴더 없음: {supp_dir}")
        sys.exit(1)
    
    # PMC 폴더 찾기
    if args.pmc_id:
        pmc_ids = [args.pmc_id]
    else:
        pmc_ids = sorted([d.name for d in supp_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('PMC')])
    
    total = len(pmc_ids)
    logger.info(f"총 {total}개 PMC 폴더 발견")
    
    # 시작 인덱스부터
    pmc_ids = pmc_ids[args.start:]
    if args.limit:
        pmc_ids = pmc_ids[:args.limit]
    
    success_count = 0
    total_processed = 0
    total_skipped = 0
    
    for i, pmc_id in enumerate(pmc_ids, args.start + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{total}] PMC {pmc_id} 처리 중...")
        logger.info(f"{'='*80}")
        
        try:
            result = process_pmc(pmc_id, base_dir, args.skip_completed)
            
            if result.get("status") == "completed":
                success_count += 1
                total_processed += result.get("processed", 0)
                total_skipped += result.get("skipped", 0)
                
                logger.info(f"  ✅ 완료: {result.get('processed', 0)}개 처리, {result.get('skipped', 0)}개 건너뜀")
            else:
                logger.warning(f"  ⚠️  상태: {result.get('status')}")
        except Exception as e:
            logger.error(f"  ❌ 처리 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 전체 처리 완료!")
    logger.info(f"  성공한 PMC: {success_count}개")
    logger.info(f"  처리된 PDF: {total_processed}개")
    logger.info(f"  건너뛴 PDF: {total_skipped}개")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()


