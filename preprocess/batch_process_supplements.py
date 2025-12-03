#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch process supplementary files downloaded under supp_raws_v1/ and organize
outputs into supp_extracted/ per PMC ID.

Rules:
- Excel (.xlsx, .xls): use extract_excel_supplements.py
- Word (.docx): use extract_word_supplements.py
- PDF (.pdf): use analyze_pdf_supplement_gpt.py (GPT-4o로 직접 분석)

Outputs per PMC under supp_extracted/PMCXXXX/ in subfolders:
- excel/, word/, pdf_gpt/
- manifest.json with summary
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

BASE_SUPP = Path("supp_raws_v1")
OUT_ROOT = Path("supp_extracted")


def find_pmc_id(path: Path) -> str:
    for part in path.parts:
        if part.startswith("PMC"):
            return part
    return ""


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    try:
        print(f"      $ {' '.join(cmd)}")
        sys.stdout.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # echo brief output tails for visibility
        if proc.stdout:
            print("      stdout:", proc.stdout.strip().splitlines()[-1:][0] if proc.stdout.strip().splitlines() else "")
        if proc.stderr:
            print("      stderr:", proc.stderr.strip().splitlines()[-1:][0] if proc.stderr.strip().splitlines() else "")
        sys.stdout.flush()
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)


def process_excel(file_path: Path, pmc_id: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # extract_excel_supplements.py를 호출하되 --output_dir 지정
    code = [sys.executable, "extract_excel_supplements.py", str(file_path), "--output_dir", str(out_dir)]
    rc, out, err = run_cmd(code)
    return {"rc": rc, "stdout": out[-2000:], "stderr": err[-2000:]}


def process_word(file_path: Path, pmc_id: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # extract_word_supplements.py를 호출하되 --output_dir 지정
    code = [sys.executable, "extract_word_supplements.py", str(file_path), "--output_dir", str(out_dir)]
    rc, out, err = run_cmd(code)
    return {"rc": rc, "stdout": out[-2000:], "stderr": err[-2000:]}


def process_pdf(file_path: Path, pmc_id: str, gpt_out_dir: Path, use_vision: bool = False) -> dict:
    """PDF 보충자료를 GPT-4o로 분석
    
    Args:
        use_vision: True면 이미지 변환 방식 (그림 포함), False면 Assistants API (빠름)
    """
    gpt_out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    if use_vision:
        # Vision API 방식: PDF를 이미지로 변환하여 분석 (그림 포함)
        code_gpt = [
            sys.executable, "analyze_pdf_supplement_gpt_vision.py",
            "--pdf", str(file_path),
            "--pmc_id", pmc_id,
            "--output_dir", str(gpt_out_dir.parent.parent)
        ]
        rc, out, err = run_cmd(code_gpt)
        results["pdf_gpt_vision"] = {"rc": rc, "stdout": out[-2000:], "stderr": err[-2000:]}
    else:
        # Assistants API 방식: PDF 직접 분석 (빠름)
        code_gpt = [
            sys.executable, "analyze_pdf_supplement_gpt.py",
            "--pdf", str(file_path),
            "--pmc_id", pmc_id,
            "--output_dir", str(gpt_out_dir.parent.parent)
        ]
        rc, out, err = run_cmd(code_gpt)
        results["pdf_gpt"] = {"rc": rc, "stdout": out[-2000:], "stderr": err[-2000:]}
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch process supplementary files")
    parser.add_argument("--start", type=int, default=1, help="Start from this index (1-based)")
    parser.add_argument("--skip-completed", action="store_true", help="Skip PMC folders that already have manifest.json")
    parser.add_argument("--use-vision", action="store_true", help="Use Vision API (PDF to image conversion) instead of Assistants API")
    args = parser.parse_args()

    base = BASE_SUPP
    if not base.exists():
        print(f"[ERR] Base directory not found: {base}")
        sys.exit(1)

    pmc_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("PMC")], key=lambda x: x.name)
    total = len(pmc_dirs)
    processed = 0
    skipped = 0
    print(f"Found {total} PMC folders under {base}.")
    if args.start > 1:
        print(f"Starting from index {args.start}.")
    if args.skip_completed:
        print(f"Skipping already completed PMCs.")
    sys.stdout.flush()

    for idx, pmc_dir in enumerate(pmc_dirs, start=1):
        if idx < args.start:
            continue

        pmc_id = pmc_dir.name
        out_pmc = OUT_ROOT / pmc_id
        
        # Skip if already processed
        if args.skip_completed and (out_pmc / "manifest.json").exists():
            print(f"[{idx}/{total}] {pmc_id}: already processed, skipping...")
            sys.stdout.flush()
            skipped += 1
            continue

        print(f"[{idx}/{total}] {pmc_id}: scanning...")
        sys.stdout.flush()
        # Consistent subfolders (created lazily when needed)
        excel_dir = out_pmc / "excel"
        word_dir = out_pmc / "word"
        pdf_gpt_dir = out_pmc / "pdf_gpt"  # GPT로 직접 분석

        manifest = {
            "pmc_id": pmc_id,
            "source_dir": str(pmc_dir),
            "timestamp": datetime.now().isoformat(),
            "files": [],
            "results": {}
        }

        files = [f for f in pmc_dir.iterdir() if f.is_file()]
        print(f"   -> {len(files)} file(s) found")
        sys.stdout.flush()
        if not files:
            print(f"   -> no files")
            sys.stdout.flush()
            continue

        for f in files:
            ext = f.suffix.lower()
            manifest["files"].append(str(f))
            try:
                if ext in [".xlsx", ".xls"]:
                    print(f"   [excel] {f.name}")
                    sys.stdout.flush()
                    res = process_excel(f, pmc_id, excel_dir)
                    manifest["results"].setdefault("excel", []).append({"file": f.name, **res})
                elif ext == ".docx":
                    print(f"   [word] {f.name}")
                    sys.stdout.flush()
                    res = process_word(f, pmc_id, word_dir)
                    manifest["results"].setdefault("word", []).append({"file": f.name, **res})
                elif ext == ".pdf":
                    print(f"   [pdf] {f.name}")
                    sys.stdout.flush()
                    res = process_pdf(f, pmc_id, pdf_gpt_dir, use_vision=args.use_vision)
                    key = "pdf_gpt_vision" if args.use_vision else "pdf_gpt"
                    manifest["results"].setdefault("pdf", []).append({"file": f.name, **res})
                else:
                    print(f"   [skip] {f.name}")
                sys.stdout.flush()
            except Exception as e:
                manifest["results"].setdefault("errors", []).append({"file": f.name, "error": str(e)})

        # Save manifest only if any result keys exist (avoid creating folder when nothing processed)
        if any(k in manifest["results"] for k in ("excel","word","pdf","errors")):
            out_pmc.mkdir(parents=True, exist_ok=True)
            (out_pmc / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            print(f"   -> manifest saved: {out_pmc / 'manifest.json'}")
            sys.stdout.flush()
            processed += 1

    print(f"Done. Processed {processed}/{total} PMC folders into {OUT_ROOT}.")
    if skipped > 0:
        print(f"Skipped {skipped} already completed folders.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()


