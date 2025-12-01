## Project Overview

LLM + RAG 기반으로 논문 본문·이미지·보충자료에서 ADMET 지표를 자동 추출·정규화하는 파이프라인입니다. PubMed/PMC에서 문서를 수집하고, 텍스트/표/이미지를 처리한 후, 컨텍스트를 누적하여 최종 JSON/CSV 산출물을 생성합니다.

## 현재 진행

- Multi-Agent가 유기적으로 상호작용 할 수 있도록 Tools를 설계하고 있는 단계입니다.


## Key Features

- PMC 본문 PDF 및 보충자료 자동 수집(도메인별 헤더·Referer·재시도 포함)
- 텍스트 추출 및 이미지/표(figure/table) 분석
- 컨텍스트 누적 기반의 다단계 ADMET 통합 추출
- 표준화된 CSV(`admet_indicators_only.csv`, `all_admet_results.csv`) 생성



## Environment

- Python 3.10+
- Conda 권장

```bash
conda create -n extRAG python=3.10 -y
conda activate extRAG
pip install -r requirements.txt
```

환경 변수(선택): 보충자료 수집의 마지막 수단으로 LLM을 사용할 경우

```bash
export OPEN_API_KEY=sk-...
```

## Folder Layout (suggested)

```
repo/
 ├─ github_upload/                 # 이 폴더(핵심 스크립트/문서)
 ├─ raws/                          # 본문 PDF
 ├─ data_test/                     # 실험용 베이스 디렉터리(산출물 위치)
 │   ├─ supp/                      # 보충자료 저장
 │   ├─ text_extracted/            # 텍스트 추출물
 │   ├─ graph_extracted/           # YOLO 등으로 분리된 도식/표
 │   ├─ graph_analyzed/            # 비전 분석 결과(JSON)
 │   ├─ supp_extracted/            # 보충자료(엑셀/워드/PDF) 분석 결과
 │   ├─ entity_analyzed/           # 코어퍼런스/명칭 통합
 │   └─ final_extracted/           # 최종 ADMET JSON/CSV
 └─ selected_pmcs.txt              # 선택 처리 시 사용(옵션)
```

## Core Scripts

- 수집/다운로드
  - `pubmed_to_pdf.py`: PMC 본문 PDF 수집 및 필터링(JATS/키워드)
  - `pmc_direct_collector.py`: PMC/DOI에서 퍼블리셔 랜딩으로 이동
  - `supp_downloader.py`: 보충자료 다운로드(Referer/Accept/재시도/검증)

- 보충자료 처리
  - `extract_excel_supplements.py`, `extract_word_supplements.py`
  - `batch_process_supplement_pdfs.py`, `batch_process_all_supplements.py`

- 이미지/표 분석
  - `analyze_yolo_extracted_images.py`
  - `batch_analyze_main_images.py`, `batch_analyze_supplement_images.py`

- 컨텍스트/통합
  - `contextual_extraction_pipeline.py`
  - `integrate_all_data.py`

- 최종 ADMET 추출
  - `final_extract_admet.py`
  - `final_extract_admet_contextual.py`(맥락 누적형), `final_extract_admet_v2_prompt.py`

## Quick Start

1) 본문 PDF 수집(옵션: 이미 보유 시 생략)

```bash
conda activate extRAG
python pubmed_to_pdf.py --out data_test/raws --max 50
```

2) 보충자료 다운로드(필요한 PMCID만 선택 가능)

```bash
python supp_downloader.py PMC12345678 -o data_test/supp
```

3) 보충자료/이미지 처리 배치

```bash
python batch_process_all_supplements.py --pmc_id PMC12345678 --base_dir data_test
```

4) 최종 ADMET 통합 추출 및 CSV 생성

```bash
python final_extract_admet.py --base_dir data_test
# 결과: data_test/final_extracted/{PMCID}_final_admet.json, CSV 집계 파일들
```

## Tips & Troubleshooting

- 403/429로 보충자료 다운로드 실패
  - `supp_downloader.py`는 브라우저형 헤더·Referer·재시도(백오프)를 적용합니다.
  - 그래도 실패 시 수동 다운로드 또는 대체 리포지터리(저자 저장소, Dryad/Zenodo/OSF) 확인을 권장합니다.
- “HTML 저장” 문제
  - 다운로드 후 `Content-Type`, 파일 시그니처(`%PDF`, `PK`)를 검증합니다. HTML 응답은 자동 폐기합니다.
- 토큰 한도/대용량 논문
  - 통합 추출 스크립트는 배치 처리로 분할 호출합니다. 배치 크기 조정으로 실패율을 낮출 수 있습니다.

## Notes

- 코드 예시는 연구용 템플릿입니다. 실제 배포/운영 시 키 관리, 로깅, 에러 처리, 속도 제한을 강화하세요.

## License

- TBD (조직 정책에 맞게 설정)
*** End Patch***} -->

