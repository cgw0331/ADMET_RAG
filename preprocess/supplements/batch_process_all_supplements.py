#!/usr/bin/env python3
"""
보충자료 통합 처리 스크립트
- data_test/supp/PMC###/ 아래 모든 파일 자동 감지
- Excel/Word/PDF 파일 타입에 따라 동적 처리
- 결과를 supp_extracted/PMC###/ 아래 통합 저장
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_supplement_files(supp_dir: Path, pmc_id: str) -> Dict[str, List[Path]]:
    """보충자료 폴더에서 모든 파일 찾기 (타입별 분류)"""
    pmc_supp_dir = supp_dir / pmc_id
    if not pmc_supp_dir.exists():
        return {}
    
    files = {
        'excel': [],
        'word': [],
        'pdf': []
    }
    
    # Excel 파일
    for ext in ['.xlsx', '.xls']:
        files['excel'].extend(list(pmc_supp_dir.glob(f"*{ext}")))
    
    # Word 파일
    for ext in ['.docx', '.doc']:
        files['word'].extend(list(pmc_supp_dir.glob(f"*{ext}")))
    
    # PDF 파일
    files['pdf'].extend(list(pmc_supp_dir.glob("*.pdf")))
    
    # 정렬
    for file_type in files:
        files[file_type] = sorted(files[file_type])
    
    return files


def process_excel(excel_path: Path, pmc_id: str, output_dir: Path, 
                  llama_normalize: bool = False) -> Dict[str, Any]:
    """Excel 파일 처리 (Llama 사용 안함)"""
    logger.info(f"  📊 Excel 처리: {excel_path.name}")
    
    try:
        # Llama 사용 안함 (기본값 False)
        result = subprocess.run(
            [
                sys.executable, "extract_excel_supplements.py",
                str(excel_path),
                "--output_dir", str(output_dir / "excel")
                # --llama-normalize 옵션 제거 (사용 안함)
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )
        
        if result.returncode == 0:
            logger.info(f"    ✅ Excel 처리 완료: {excel_path.name}")
            return {"status": "success", "file": excel_path.name}
        else:
            logger.error(f"    ❌ Excel 처리 실패: {excel_path.name}")
            logger.error(f"      stderr: {result.stderr[-500:]}")
            return {"status": "failed", "file": excel_path.name, "error": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        logger.error(f"    ❌ Excel 처리 타임아웃: {excel_path.name}")
        return {"status": "timeout", "file": excel_path.name}
    except Exception as e:
        logger.error(f"    ❌ Excel 처리 오류: {excel_path.name} - {e}")
        return {"status": "error", "file": excel_path.name, "error": str(e)}


def process_word(word_path: Path, pmc_id: str, output_dir: Path) -> Dict[str, Any]:
    """Word 파일 처리"""
    logger.info(f"  📝 Word 처리: {word_path.name}")
    
    try:
        result = subprocess.run(
            [
                sys.executable, "extract_word_supplements.py",
                str(word_path),
                "--output_dir", str(output_dir / "word")
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )
        
        if result.returncode == 0:
            logger.info(f"    ✅ Word 처리 완료: {word_path.name}")
            return {"status": "success", "file": word_path.name}
        else:
            logger.error(f"    ❌ Word 처리 실패: {word_path.name}")
            logger.error(f"      stderr: {result.stderr[-500:]}")
            return {"status": "failed", "file": word_path.name, "error": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        logger.error(f"    ❌ Word 처리 타임아웃: {word_path.name}")
        return {"status": "timeout", "file": word_path.name}
    except Exception as e:
        logger.error(f"    ❌ Word 처리 오류: {word_path.name} - {e}")
        return {"status": "error", "file": word_path.name, "error": str(e)}


def process_pdf(pdf_path: Path, pmc_id: str, output_dir: Path,
                skip_completed: bool = False,
                previous_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """PDF 파일 처리 (YOLO 추출 + GPT-4o Vision 분석)
    
    Args:
        pdf_path: PDF 파일 경로
        pmc_id: PMC ID
        output_dir: 출력 디렉토리
        skip_completed: 완료된 파일 건너뛰기
        previous_context: 이전 단계에서 누적된 컨텍스트 (선택적)
    """
    logger.info(f"  📄 PDF 처리: {pdf_path.name}")
    
    pdf_name = pdf_path.stem
    
    # 출력 경로
    pdf_graph_dir = output_dir / "pdf_graph" / pdf_name
    pdf_info_dir = output_dir / "pdf_info"
    output_file = pdf_info_dir / f"{pdf_name}_yolo_gpt_analysis.json"
    
    # 이미 완료되었는지 확인
    if skip_completed and output_file.exists():
        logger.info(f"    ⏭️  이미 완료됨, 건너뜀")
        return {"status": "skipped", "file": pdf_path.name}
    
    # 1. YOLO로 figure/table 추출
    extracted = False
    if pdf_graph_dir.exists() and (
        (pdf_graph_dir / "figures").exists() and any((pdf_graph_dir / "figures").iterdir()) or
        (pdf_graph_dir / "tables").exists() and any((pdf_graph_dir / "tables").iterdir())
    ):
        logger.info(f"    ✅ 이미 추출됨")
        extracted = True
    else:
        pdf_graph_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"    🔍 YOLO 추출 중...")
        
        try:
            result = subprocess.run(
                [
                    sys.executable, "inference_yolo.py",
                    "--pdf", str(pdf_path),
                    "--output", str(pdf_graph_dir),
                    "--confidence", "0.25"
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            if result.returncode == 0:
                extracted = True
                logger.info(f"    ✅ YOLO 추출 완료")
            else:
                logger.error(f"    ❌ YOLO 추출 실패: {result.stderr[-500:]}")
                return {"status": "extraction_failed", "file": pdf_path.name, "error": result.stderr[-500:]}
        except subprocess.TimeoutExpired:
            logger.error(f"    ❌ YOLO 추출 타임아웃")
            return {"status": "extraction_timeout", "file": pdf_path.name}
        except Exception as e:
            logger.error(f"    ❌ YOLO 추출 오류: {e}")
            return {"status": "extraction_error", "file": pdf_path.name, "error": str(e)}
    
    # 2. GPT-4o Vision으로 분석 (이전 컨텍스트 전달)
    if not extracted:
        return {"status": "extraction_failed", "file": pdf_path.name}
    
    pdf_info_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"    🤖 GPT-4o Vision 분석 중...")
    if previous_context:
        prev_compounds = len(previous_context.get("compounds", {}))
        logger.info(f"    📋 이전 단계 화합물 {prev_compounds}개 참조")
    
    try:
        from analyze_yolo_extracted_images import YOLOImageAnalyzer
        analyzer = YOLOImageAnalyzer()
        result = analyzer.process_pdf_graph(pdf_graph_dir, pdf_info_dir, previous_context=previous_context)
        
        compounds_count = len(result.get('compounds', []))
        images_count = result.get('summary', {}).get('images_processed', 0)
        
        logger.info(f"    ✅ 분석 완료: 화합물 {compounds_count}개, 이미지 {images_count}개")
        return {
            "status": "success",
            "file": pdf_path.name,
            "compounds": compounds_count,
            "images": images_count
        }
    except Exception as e:
        logger.error(f"    ❌ GPT-4o Vision 분석 실패: {e}")
        return {"status": "analysis_failed", "file": pdf_path.name, "error": str(e)}


def process_pmc(pmc_id: str, base_dir: Path = Path("data_test"),
                skip_completed: bool = False,
                llama_normalize_excel: bool = False) -> Dict[str, Any]:
    """특정 PMC의 모든 보충자료 처리 (맥락 누적)"""
    logger.info(f"PMC {pmc_id} 처리 시작...")
    
    # 컨텍스트 파이프라인 로드
    from contextual_extraction_pipeline import ContextualExtractionPipeline
    pipeline = ContextualExtractionPipeline(base_dir=str(base_dir))
    context = pipeline.get_accumulated_context(pmc_id)
    
    supp_dir = base_dir / "supp"
    output_dir = base_dir / "supp_extracted" / pmc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 찾기
    files = find_supplement_files(supp_dir, pmc_id)
    
    total_files = sum(len(files[ft]) for ft in files)
    if total_files == 0:
        logger.warning(f"  ⚠️  보충자료 파일 없음")
        return {
            "pmc_id": pmc_id,
            "status": "no_files",
            "total": 0
        }
    
    logger.info(f"  발견된 파일:")
    logger.info(f"    Excel: {len(files['excel'])}개")
    logger.info(f"    Word: {len(files['word'])}개")
    logger.info(f"    PDF: {len(files['pdf'])}개")
    logger.info(f"    총 {total_files}개")
    
    results = {
        "excel": [],
        "word": [],
        "pdf": []
    }
    
    # Excel 처리 (이전 컨텍스트 전달)
    for excel_path in files['excel']:
        result = process_excel(excel_path, pmc_id, output_dir, llama_normalize_excel)
        results['excel'].append(result)
        # TODO: Excel 결과를 컨텍스트에 누적
    
    # Word 처리 (이전 컨텍스트 전달)
    for word_path in files['word']:
        result = process_word(word_path, pmc_id, output_dir)
        results['word'].append(result)
        # TODO: Word 결과를 컨텍스트에 누적
    
    # PDF 처리 (이전 컨텍스트 전달 및 누적)
    for pdf_path in files['pdf']:
        result = process_pdf(pdf_path, pmc_id, output_dir, skip_completed, previous_context=context)
        results['pdf'].append(result)
        
        # PDF 결과를 컨텍스트에 누적
        if result.get('status') == 'success':
            # PDF 분석 결과 로드하여 컨텍스트에 추가
            pdf_name = pdf_path.stem
            pdf_info_file = output_dir / "pdf_info" / f"{pdf_name}_yolo_gpt_analysis.json"
            if pdf_info_file.exists():
                try:
                    with open(pdf_info_file, 'r', encoding='utf-8') as f:
                        pdf_result = json.load(f)
                    compounds = pdf_result.get('compounds', [])
                    # 컨텍스트에 누적
                    for comp in compounds:
                        comp_name = comp.get('compound_name', '').strip()
                        if comp_name:
                            if comp_name not in context['compounds']:
                                context['compounds'][comp_name] = {
                                    'aliases': set(),
                                    'attributes': defaultdict(list),
                                    'sources': []
                                }
                            # 속성 추가
                            for attr_name, attr_data in comp.get('attributes', {}).items():
                                if isinstance(attr_data, dict):
                                    value = attr_data.get('value', '')
                                    if value:
                                        context['compounds'][comp_name]['attributes'][attr_name].append({
                                            'value': value,
                                            'source': f'pdf_{pdf_name}'
                                        })
                            if f'pdf_{pdf_name}' not in context['compounds'][comp_name]['sources']:
                                context['compounds'][comp_name]['sources'].append(f'pdf_{pdf_name}')
                except Exception as e:
                    logger.warning(f"  컨텍스트 누적 실패: {e}")
        
        # 컨텍스트 저장
        pipeline.save_accumulated_context(pmc_id, context)
    
    # 통계
    excel_success = sum(1 for r in results['excel'] if r.get('status') == 'success')
    word_success = sum(1 for r in results['word'] if r.get('status') == 'success')
    pdf_success = sum(1 for r in results['pdf'] if r.get('status') == 'success')
    
    total_success = excel_success + word_success + pdf_success
    
    logger.info(f"  ✅ 처리 완료:")
    logger.info(f"    Excel: {excel_success}/{len(files['excel'])}")
    logger.info(f"    Word: {word_success}/{len(files['word'])}")
    logger.info(f"    PDF: {pdf_success}/{len(files['pdf'])}")
    logger.info(f"    총: {total_success}/{total_files}")
    
    return {
        "pmc_id": pmc_id,
        "status": "completed",
        "total_files": total_files,
        "success": total_success,
        "results": results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="보충자료 통합 처리 (Excel/Word/PDF 자동 감지)")
    parser.add_argument("--pmc_id", help="특정 PMC ID만 처리")
    parser.add_argument("--base_dir", default="data_test", help="기본 디렉토리")
    parser.add_argument("--skip-completed", action="store_true", help="이미 완료된 PDF 건너뛰기")
    parser.add_argument("--llama-normalize", action="store_true", help="Excel 속성 정규화에 Llama 사용")
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
    total_files = 0
    total_success = 0
    
    for i, pmc_id in enumerate(pmc_ids, args.start + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{total}] PMC {pmc_id} 처리 중...")
        logger.info(f"{'='*80}")
        
        try:
            result = process_pmc(pmc_id, base_dir, args.skip_completed, args.llama_normalize)
            
            if result.get("status") == "completed":
                success_count += 1
                total_files += result.get("total_files", 0)
                total_success += result.get("success", 0)
            else:
                logger.warning(f"  ⚠️  상태: {result.get('status')}")
        except Exception as e:
            logger.error(f"  ❌ 처리 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 전체 처리 완료!")
    logger.info(f"  성공한 PMC: {success_count}개")
    logger.info(f"  처리된 파일: {total_success}/{total_files}개")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

