#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
멀티 에이전트 오케스트레이터
- 비전 에이전트: 본문/보충자료의 도식·표 분석
- 코어퍼런스 에이전트: 전역/국소 코어퍼런스 사전 생성·병합
- 메인 에이전트(오케스트레이터): 다양한 소스를 통합하여 최종 ADMET JSON/CSV 추출

설계 목표:
- 기존 모듈과의 결합도 최소화: 서브프로세스로 기존 스크립트 호출
- 에이전트 프로파일 명확화(목적, 입력, 출력)
- 견고한 실행(환경 점검, 로그, 오류 표면화)
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------
# 유틸리티
# ---------------------------

def check_environment_or_exit() -> None:
	"""
	가벼운 런타임 가드:
	- 필수 의존 패키지 임포트 가능 여부 점검
	- 누락 시 conda 활성화 또는 설치 안내
	"""
	missing: List[str] = []
	for mod in ["requests", "pandas", "PIL", "openai"]:
		try:
			__import__(mod)
		except Exception:
			missing.append(mod)
	if missing:
		msg = (
			"[ENV] Missing required packages: "
			+ ", ".join(missing)
			+ "\nPlease activate your conda environment first:\n"
			+ "  conda activate extRAG\n"
			+ "or install dependencies:\n"
			+ "  pip install -r requirements.txt\n"
		)
		print(msg, file=sys.stderr)
		sys.exit(1)


def run_command(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
	"""
	지정된 작업 디렉토리에서 서브프로세스를 실행하고 표준 출력/에러를 캡처한다.
	반환값: (returncode, stdout, stderr)
	"""
	try:
		proc = subprocess.run(
			cmd,
			cwd=str(cwd),
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			check=False,
		)
		return proc.returncode, proc.stdout, proc.stderr
	except Exception as e:
		return 1, "", f"Exception while running {' '.join(shlex.quote(c) for c in cmd)}: {e}"


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


# ---------------------------
# 에이전트 프로파일
# ---------------------------

@dataclass
class AgentProfile:
	name: str
	purpose: str
	inputs: List[str]
	outputs: List[str]
	tools: List[str] = field(default_factory=list)


VISION_PROFILE = AgentProfile(
	name="Vision Agent",
	purpose="Analyze figures/tables to extract compound and ADMET cues; update per-PMC image analyses.",
	inputs=[
		"graph_extracted/{PMCID}/article/{figures,tables}/*.png",
		"supp_extracted/{PMCID}/pdf_graph/*/*.png (optional)"
	],
	outputs=[
		"graph_analyzed/{PMCID}/figure_analyses.json",
		"graph_analyzed/{PMCID}/table_analyses.json",
		"graph_analyzed/{PMCID}/all_analyses.json",
		"graph_analyzed/{PMCID}/summary.json"
	],
	tools=[
		"batch_analyze_main_images.py",
		"batch_analyze_supplement_images.py (optional)"
	],
)

COREFERENCE_PROFILE = AgentProfile(
	name="Coreference Agent",
	purpose="Build entity alias mapping (focus on compounds) and relationships from text/supp/image results.",
	inputs=[
		"text_extracted/{PMCID}/extracted_text.txt",
		"graph_analyzed/{PMCID}/all_analyses.json",
		"supp_extracted/{PMCID}/{excel,word,pdf_info}/...json"
	],
	outputs=[
		"entity_analyzed/{PMCID}/global_coreference.json",
		"entity_analyzed/global_coreference_accumulated.json"
	],
	tools=[
		"build_global_coreference.py"
	],
)

MAIN_PROFILE = AgentProfile(
	name="Main Agent (Orchestrator)",
	purpose="Integrate text/image/supp/coref and extract final ADMET JSON; later aggregate CSV.",
	inputs=[
		"text_extracted/{PMCID}/extracted_text.txt",
		"graph_analyzed/{PMCID}/all_analyses.json",
		"supp_extracted/{PMCID}/**/*.json",
		"entity_analyzed/{PMCID}/global_coreference*.json"
	],
	outputs=[
		"final_extracted/{PMCID}/{PMCID}_final_admet.json",
		"final_extracted/all_admet_results.csv (project-level)",
		"final_extracted/admet_indicators_only.csv (project-level)"
	],
	tools=[
		"final_extract_admet.py",
		"integrate_all_data.py (for CSV aggregation, if needed)"
	],
)


# ---------------------------
# Agents (tool-wrapping)
# ---------------------------

class VisionAgent:
	def __init__(self, repo_root: Path, base_dir: Path):
		self.repo_root = repo_root
		self.base_dir = base_dir

	def run_main_images(self, pmc_id: str, model: str = "gpt-4o") -> bool:
		"""
		특정 PMCID의 본문 도식/표를 분석한다.
		실제 처리는 preprocess/batch_analyze_main_images.py에 위임한다.
		"""
		cmd = [
			sys.executable,
			"preprocess/batch_analyze_main_images.py",
			"--pmc_id", pmc_id,
			"--base_dir", str(self.base_dir / "graph_extracted"),
			"--model", model,
			"--use-gpt",
		]
		rc, out, err = run_command(cmd, cwd=self.repo_root)
		print(out)
		if rc != 0:
			print(err, file=sys.stderr)
		return rc == 0

	def run_supp_images(self, pmc_id: str, model: str = "gpt-4o") -> bool:
		"""
		보충자료 PDF에서 추출된 이미지를 분석한다(선택).
		스크립트가 존재할 경우 preprocess/batch_analyze_supplement_images.py에 위임한다.
		"""
		script = self.repo_root / "preprocess/batch_analyze_supplement_images.py"
		if not script.exists():
			return True  # optional; treat as success if script not present
		cmd = [
			sys.executable,
			"preprocess/batch_analyze_supplement_images.py",
			"--pmc_id", pmc_id,
			"--base_dir", str(self.base_dir),
			"--model", model,
			"--use-gpt",
		]
		rc, out, err = run_command(cmd, cwd=self.repo_root)
		print(out)
		if rc != 0:
			print(err, file=sys.stderr)
		return rc == 0


class CoreferenceAgent:
	def __init__(self, repo_root: Path):
		self.repo_root = repo_root

	def build(self, pmc_id: str, model: str = "gpt-4o") -> bool:
		"""
		PMC 단위 코어퍼런스를 생성하고 전역 사전을 갱신한다.
		실제 처리는 preprocess/build_global_coreference.py에 위임한다.
		"""
		cmd = [
			sys.executable,
			"preprocess/build_global_coreference.py",
			"--pmc_id", pmc_id,
			"--model", model,
			"--use-gpt",
		]
		rc, out, err = run_command(cmd, cwd=self.repo_root)
		print(out)
		if rc != 0:
			print(err, file=sys.stderr)
		return rc == 0


class MainAgent:
	def __init__(self, repo_root: Path, base_dir: Path):
		self.repo_root = repo_root
		self.base_dir = base_dir

	def extract_final_admet(self, pmc_id: str) -> bool:
		"""
		PMC 단위 최종 ADMET 추출을 수행한다.
		실제 처리는 preprocess/final_extract_admet.py에 위임한다.
		"""
		cmd = [
			sys.executable,
			"preprocess/final_extract_admet.py",
			"--pmc_id", pmc_id,
			"--base_dir", str(self.base_dir),
		]
		rc, out, err = run_command(cmd, cwd=self.repo_root)
		print(out)
		if rc != 0:
			print(err, file=sys.stderr)
		return rc == 0


# ---------------------------
# 오케스트레이터
# ---------------------------

class Orchestrator:
	def __init__(self, repo_root: Path, base_dir: Path):
		self.repo_root = repo_root
		self.base_dir = base_dir
		self.vision = VisionAgent(repo_root, base_dir)
		self.coref = CoreferenceAgent(repo_root)
		self.main = MainAgent(repo_root, base_dir)

	def run_for_pmc(
		self,
		pmc_id: str,
		run_vision: bool = True,
		run_coref: bool = True,
		run_main: bool = True,
		model: str = "gpt-4o",
	) -> Dict[str, bool]:
		"""
		하나의 PMC에 대해 선택된 단계들을 실행한다.
		반환값은 단계별 성공 여부 딕셔너리이다.
		"""
		results = {"vision": True, "coref": True, "main": True}
		print(f"\n=== Processing {pmc_id} ===")

		if run_vision:
			ok_main = self.vision.run_main_images(pmc_id, model=model)
			ok_supp = self.vision.run_supp_images(pmc_id, model=model)
			results["vision"] = ok_main and ok_supp
			print(f"[Vision] {pmc_id}: {'OK' if results['vision'] else 'FAIL'}")

		if run_coref:
			ok_coref = self.coref.build(pmc_id, model=model)
			results["coref"] = ok_coref
			print(f"[Coref] {pmc_id}: {'OK' if ok_coref else 'FAIL'}")

		if run_main:
			ok_main_extract = self.main.extract_final_admet(pmc_id)
			results["main"] = ok_main_extract
			print(f"[Main] {pmc_id}: {'OK' if ok_main_extract else 'FAIL'}")

		return results


# ---------------------------
# 명령행 인터페이스(CLI)
# ---------------------------

def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Multi-Agent Orchestrator for ADMET extraction")
	ap.add_argument("--pmc_id", action="append", help="PMC ID (repeatable). If omitted, use selected_pmcs.txt")
	ap.add_argument("--base_dir", default="data_test", help="Base directory for inputs/outputs")
	ap.add_argument("--model", default="gpt-4o", help="Model name for Vision/Coref (default: gpt-4o)")
	ap.add_argument("--no-vision", action="store_true", help="Skip Vision Agent step")
	ap.add_argument("--no-coref", action="store_true", help="Skip Coreference Agent step")
	ap.add_argument("--no-main", action="store_true", help="Skip Main Agent step")
	return ap.parse_args()


def load_default_pmcs(repo_root: Path) -> List[str]:
	sel = repo_root / "selected_pmcs.txt"
	if sel.exists():
		with open(sel, "r", encoding="utf-8") as f:
			pmcs = [line.strip() for line in f if line.strip()]
		return pmcs
	return []


def main():
	check_environment_or_exit()

	args = parse_args()
	repo_root = Path(__file__).resolve().parent
	base_dir = Path(args.base_dir)
	ensure_dir(base_dir)

	pmcs: List[str] = args.pmc_id or load_default_pmcs(repo_root)
	if not pmcs:
		print("No PMC IDs provided (use --pmc_id or create selected_pmcs.txt).", file=sys.stderr)
		sys.exit(1)

	orch = Orchestrator(repo_root=repo_root, base_dir=base_dir)

	run_vision = not args.no_vision
	run_coref = not args.no_coref
	run_main = not args.no_main

	total = len(pmcs)
	ok_count = 0

	for idx, pmc_id in enumerate(pmcs, 1):
		print(f"\n[{idx}/{total}] {pmc_id}")
		res = orch.run_for_pmc(
			pmc_id=pmc_id,
			run_vision=run_vision,
			run_coref=run_coref,
			run_main=run_main,
			model=args.model,
		)
		if all(res.values()):
			ok_count += 1

	print(f"\n✅ Done. Success: {ok_count}/{total}")


if __name__ == "__main__":
	main()


