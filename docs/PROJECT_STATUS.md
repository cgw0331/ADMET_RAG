# ADMET 데이터 추출 프로젝트 현황 문서

## 📋 프로젝트 개요

### 목적
과학 논문(PubMed Central)에서 **화합물(Compound)의 ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) 데이터**를 자동으로 추출하여 구조화된 데이터셋을 구축하는 것.

### 핵심 목표
- 논문 본문, 그림/표, 보충자료에서 **모든 화합물과 그들의 ADMET 지표**를 완전하게 추출
- 여러 소스의 정보를 **유기적으로 연결**하여 통합된 데이터 생성
- 추출된 데이터를 **구조화된 JSON/CSV 형식**으로 저장

---

## 🔄 현재 진행 방식 (파이프라인)

### 0단계: PDF 수집 및 다운로드
- **입력**: PMC ID 목록 또는 검색 쿼리
- **처리**: 
  - **PMC 직접 다운로드**: `pmc_direct_collector.py`, `gpt_paper_collector.py`
    - PMC 표준 URL 패턴 시도: `/pmc/articles/{pmcid}/pdf/`
    - 여러 fallback URL 시도 (europepmc.org 등)
  - **보충자료 다운로드**: `supp_downloader.py`
    - PMC 페이지에서 보충자료 링크 추출
    - DOI 기반 출판사 페이지 접근
    - Elsevier CDN 휴리스틱 (ars.els-cdn.com)
    - GPT 기반 링크 추론 (최후 수단)
- **출력**: 
  - `data_test/raws/PMC###/article.pdf` (원문)
  - `data_test/supp/PMC###/*.{xlsx,docx,pdf}` (보충자료)

#### 0-1. 논문 필터링 (2단계 검증)

**1단계: JATS 메타데이터 기반 사전 필터링** (PDF 다운로드 전)
- **목적**: PDF 다운로드 전에 논문이 아닌 문서를 제외하여 시간/비용 절약
- **처리**: `is_research_paper_from_jats()` 함수
- **검증 항목**:
  - **문서 유형 필터링**: JATS XML의 `article-categories` 확인
    - 제외: `editorial`, `letter`, `comment`, `correction`, `retraction`
    - 제외: `guideline`, `protocol`, `manual`, `handbook`, `book`
    - 제외: `supplement`, `appendix`, `figure`, `table`, `data`
    - 제외: `review`, `meta-analysis`, `case-report`, `brief-report`
  - **제목 키워드 필터링**: 제목에서 비논문 키워드 확인
    - 제외 키워드: `guideline`, `protocol`, `manual`, `handbook`, `instruction`
    - 제외 키워드: `supplement`, `appendix`, `figure`, `table`, `data`
    - 제외 키워드: `editorial`, `letter`, `comment`, `correction`, `retraction`
- **효과**: PDF 다운로드 전에 약 20-30%의 비논문 문서 제외

**2단계: PDF 내용 기반 필터링** (다운로드 후)
- **라이브러리**: `pdfplumber` - PDF 텍스트 추출
- **목적**: 다운로드한 PDF가 실제 연구 논문인지 최종 검증
- **처리**: `is_research_paper()` 함수
- **검증 방법**: `pdfplumber`로 PDF 텍스트 추출 후 키워드 기반 점수 계산
- **논문 키워드** (포함 점수):
  - `abstract`, `introduction`, `methods`, `results`, `discussion`, `conclusion`
  - `materials and methods`, `experimental`, `data`, `analysis`
  - `peer reviewed`, `manuscript`, `article`, `research`, `study`
- **비논문 키워드** (제외 점수):
  - **튜토리얼/매뉴얼**: `laboratory manual`, `handbook`, `textbook`, `edition`, `chapter`
  - **저널 표지/부록**: `table of contents`, `preface`, `appendix`, `glossary`, `index`
  - **출판사 정보**: `cold spring harbor`, `wiley`, `elsevier`, `springer`
  - **프레젠테이션**: `slide`, `presentation`, `conference`, `symposium`, `workshop`
  - **프레젠테이션 도구**: `powerpoint`, `keynote`, `agenda`, `speaker`, `session`
  - **보고서**: `final report`, `technical report`, `contract`, `submitted to`
  - **규제 문서**: `summary of product characteristics`, `spc`, `package insert`, `label`, `fda`, `ema`
  - **가이드라인**: `guideline`, `protocol`, `manuscript body formatting guidelines`, `formatting guidelines`
  - **지시사항**: `instruction`, `instructions`, `how to`
  - **샘플/템플릿**: `sample`, `template`, `example`, `lorem ipsum`, `dummy text`
- **판단 기준**:
  1. 비논문 키워드 3개 이상 → **제외**
  2. 비논문 키워드 2개 이상 + 논문 키워드 ≤ 비논문 키워드 → **제외**
  3. 논문 키워드 5개 이상 → **포함**
  4. 논문 키워드 3개 이상 + 비논문 키워드 ≤ 1개 → **포함**
  5. 애매한 경우: 텍스트 길이 > 2000자 + `abstract`/`introduction` 포함 + `lorem ipsum` 없음 → **포함**
- **효과**: 다운로드된 PDF 중 약 10-15%의 비논문 문서 추가 제외

**필터링 결과**:
- 전체 검색 결과 중 약 30-40%가 필터링되어 제외됨
- 최종적으로 실제 연구 논문만 저장됨
- 튜토리얼, 저널 표지, 에디토리얼, 리뷰 등이 제외됨

- **어려움**:
  - ⚠️ **403 Forbidden 오류**: MDPI, ACS 등 일부 출판사에서 접근 차단
  - ⚠️ **429 Too Many Requests**: PMC API 호출 제한
  - ⚠️ **보충자료 링크 불명확**: HTML 파싱으로 찾기 어려운 경우 많음
  - ⚠️ **다양한 출판사 형식**: 각 출판사마다 다른 URL 패턴과 접근 방식
  - ⚠️ **필터링 오판**: 일부 실제 논문이 필터링되거나, 비논문이 통과하는 경우
  - ⚠️ **JATS 메타데이터 부재**: 일부 논문은 JATS가 없어 사전 필터링 불가

### 1단계: 원문 PDF 전처리 및 분리

#### 1-1. 텍스트 추출
- **라이브러리**: `pdfplumber` (PDF 텍스트 추출)
- **처리**: `pdf_to_text.py` 스크립트
- **출력**: `text_extracted/PMC###/extracted_text.txt`
- **어려움**:
  - ⚠️ **PDF 인코딩 문제**: 일부 PDF에서 특수문자 깨짐
  - ⚠️ **2-column 레이아웃**: 컬럼 순서가 뒤바뀌는 경우
  - ⚠️ **수식/기호**: LaTeX 수식이 텍스트로 제대로 변환 안 됨
  - ⚠️ **표 내부 텍스트**: 표가 이미지로 되어 있으면 추출 불가

#### 1-2. 이미지/텍스트 분리 (YOLO)
- **라이브러리**: 
  - `PyMuPDF` (fitz) - PDF를 이미지로 변환
  - `ultralytics YOLO` - 객체 감지 모델
  - `PIL` (Pillow) - 이미지 처리
  - `OpenCV` (cv2) - 이미지 리사이즈 및 좌표 변환
- **처리**: `inference_yolo.py` 스크립트
- **모델**: PubLayNet 기반 훈련된 YOLOv8 5-class 모델 (text, title, list, table, figure)
- **과정**:
  1. PDF를 이미지로 변환 (`PyMuPDF` fitz)
  2. 612x792 크기로 리사이즈 (PubLayNet 훈련 크기)
  3. `ultralytics YOLO`로 객체 감지 (confidence threshold: 0.25-0.4)
  4. NMS (Non-Maximum Suppression)로 중복 제거
  5. 원본 해상도로 좌표 변환 (`OpenCV`)
  6. figure/table만 추출하여 PNG로 저장 (`PIL`)
- **출력**: 
  - `graph_extracted/PMC###/figures/*.png`
  - `graph_extracted/PMC###/tables/*.png`
- **어려움**:
  - ⚠️ **YOLO 모델 정확도**: 일부 figure/table을 놓치거나 잘못 분류
  - ⚠️ **복잡한 레이아웃**: figure와 table이 겹쳐있는 경우
  - ⚠️ **작은 객체**: 작은 figure/table이 감지 안 됨 (min_area 설정 필요)
  - ⚠️ **해상도 문제**: 저해상도 PDF에서 객체 경계가 불명확
  - ⚠️ **처리 시간**: 긴 논문(100+ 페이지)은 처리 시간이 오래 걸림
  - ⚠️ **메모리 사용**: 대용량 PDF 처리 시 메모리 부족 가능

#### 1-3. 이미지 분석 (GPT-4o Vision)
- **라이브러리**: 
  - `OpenAI API` (openai 패키지) - GPT-4o Vision 모델
  - `PIL` (Pillow) - 이미지 로드 및 base64 인코딩
- **처리**: `analyze_yolo_extracted_images.py` 스크립트
- **과정**:
  1. 추출된 figure/table 이미지를 base64 인코딩 (`PIL`)
  2. `OpenAI API` GPT-4o Vision 호출 (각 이미지마다)
  3. JSON Lines 형식으로 화합물 정보 추출
  4. 화합물별로 그룹화 및 중복 제거
- **출력**: `graph_analyzed/PMC###/all_analyses.json`
- **어려움**:
  - ⚠️ **API 비용**: 이미지가 많을수록 비용 증가 (예: 50개 이미지 = 50회 API 호출)
  - ⚠️ **이미지 품질**: 저해상도 이미지에서 텍스트 인식 실패
  - ⚠️ **표 구조 복잡성**: 병합된 셀, 다단 구조 등 복잡한 표는 파싱 어려움
  - ⚠️ **컨텍스트 부족**: 각 이미지를 독립적으로 분석하여 이전 이미지 정보를 모름 (→ 대화 히스토리로 해결 시도 중)

### 2단계: 보충자료 처리

#### 2-1. Excel 파일 처리
- **라이브러리**: 
  - `pandas` - Excel 파일 읽기 및 데이터 처리
  - `openpyxl` - Excel 파일 파싱 (pandas 백엔드)
  - `Ollama` (선택적) - Llama 모델로 속성 정규화
- **처리**: `extract_excel_supplements.py` → `admet_extract_llama.py`
- **과정**:
  1. Excel 파일 로드 (`pandas.read_excel()`)
  2. 헤더 자동 감지 (LLM 기반 또는 휴리스틱)
  3. 화합물-속성 쌍 추출 (long format, `pandas`)
  4. (선택적) `Ollama` Llama 모델로 속성 정규화
- **출력**: `supp_extracted/PMC###/excel/PMC###_compounds_from_excel.json`
- **어려움**:
  - ⚠️ **헤더 감지 실패**: 헤더가 여러 행에 걸쳐 있거나 비표준 형식
  - ⚠️ **병합된 셀**: Excel의 병합된 셀 처리 어려움
  - ⚠️ **다중 시트**: 여러 시트에 걸쳐 있는 데이터 통합 필요
  - ⚠️ **데이터 형식 불일치**: 숫자/텍스트 혼재, 단위 불명확

#### 2-2. Word 파일 처리
- **라이브러리**: 
  - `python-docx` (docx) - Word 파일 읽기 및 표 추출
  - `pandas` - 표 데이터를 DataFrame으로 변환
- **처리**: `extract_word_supplements.py` 스크립트
- **과정**:
  1. Word 파일 로드 (`python-docx Document()`)
  2. 표 추출 (`doc.tables`, `pandas DataFrame`으로 변환)
  3. 표가 없으면 텍스트 추출
- **출력**: `supp_extracted/PMC###/word/PMC###_compounds_from_word.json`
- **어려움**:
  - ⚠️ **복잡한 표 구조**: Word 표는 Excel보다 구조가 복잡
  - ⚠️ **이미지 내 표**: 표가 이미지로 삽입된 경우 추출 불가
  - ⚠️ **텍스트만 있는 경우**: 표가 없으면 구조화된 데이터 추출 어려움

#### 2-3. PDF 보충자료 처리
- **라이브러리**: 
  - `PyMuPDF` (fitz) - PDF 처리
  - `ultralytics YOLO` - 객체 감지
  - `OpenAI API` - GPT-4o Vision
  - `pdfplumber` (선택적) - PDF 텍스트 추출
- **처리**: `batch_process_supplement_pdfs.py` 스크립트
- **과정**:
  1. `ultralytics YOLO`로 figure/table 추출 (본문과 동일)
  2. `OpenAI API` GPT-4o Vision으로 이미지 분석
  3. (선택적) `pdfplumber`로 PDF 텍스트 추출
- **출력**: `supp_extracted/PMC###/pdf_info/*_yolo_gpt_analysis.json`
- **어려움**:
  - ⚠️ **보충자료 PDF 품질**: 스캔본이거나 저해상도인 경우 많음
  - ⚠️ **대용량 파일**: 보충자료 PDF가 수백 페이지인 경우 처리 시간 오래 걸림
  - ⚠️ **본문과 동일한 문제**: YOLO 정확도, 이미지 품질 등

### 3단계: 코어퍼런스 분석
- **라이브러리**: 
  - `OpenAI API` (GPT-4o) 또는 `Ollama` (Llama 4)
- **입력**: 텍스트 분석 결과
- **처리**: `batch_build_coreference.py` 스크립트
  - `OpenAI API` GPT-4o 또는 `Ollama` Llama로 텍스트 분석
  - 화합물 이름, 별칭, 약어 추출
  - 동일 화합물 그룹핑
- **출력**: `entity_analyzed/PMC###/global_coreference_gpt.json`
- **목적**: 같은 화합물의 다른 이름/별칭 통합 (예: "APA" = "6-(4-aminophenyl)-N-(3,4,5-trimethoxyphenyl)pyrazin-2-amine")
- **어려움**:
  - ⚠️ **약어 해석**: "Compound 1", "C1" 같은 약어가 어떤 화합물인지 추론 어려움
  - ⚠️ **제형 구분**: "Liposomal APA"와 "APA"를 같은 것으로 볼지 다른 것으로 볼지 애매
  - ⚠️ **SMILES/InChI 부재**: 구조식이 없으면 이름만으로 매칭 어려움

### 4단계: 최종 ADMET 통합 추출
- **라이브러리**: 
  - `OpenAI API` (openai 패키지) - GPT-4o 모델
  - `Pydantic` - 구조화된 출력 스키마 정의 및 검증
- **입력**: 모든 단계의 결과물
  - 텍스트 추출 결과
  - 이미지 분석 결과
  - 보충자료 추출 결과 (Excel/Word/PDF)
  - 코어퍼런스 딕셔너리
- **처리**: `final_extract_admet.py` 스크립트
  - 모든 소스의 데이터를 하나의 프롬프트로 통합
  - `OpenAI API` GPT-4o structured output 사용 (`Pydantic BaseModel`)
  - 배치 처리로 화합물을 그룹으로 나누어 처리
  - 화합물별로 모든 ADMET 지표 추출
- **출력**: `final_extracted/PMC###/PMC###_final_admet.json`
- **목적**: 모든 소스의 정보를 통합하여 완전한 ADMET 테이블 생성
- **어려움**:
  - ⚠️ **토큰 제한**: 입력 토큰은 충분하지만 출력 토큰(16,384) 제한으로 일부 화합물 누락
  - ⚠️ **소스 간 충돌**: 같은 화합물의 같은 지표가 다른 값으로 나타나는 경우
  - ⚠️ **컨텍스트 부족**: 각 단계가 독립적으로 실행되어 정보가 유기적으로 연결 안 됨
  - ⚠️ **배치 크기 조정**: 화합물 수에 따라 배치 크기를 동적으로 조정해야 함

---

## 🎯 해결해야 할 핵심 문제들

### 1. **정보의 유기적 연결 부족** ⚠️ **최우선 문제**
**문제점**:
- 각 처리 단계(텍스트 추출, 이미지 분석, 보충자료 추출)가 **독립적으로 실행**됨
- 각 단계의 LLM이 **이전 단계의 맥락을 모름**
- 예: 텍스트에서 "Compound A"를 발견했는데, 이미지 분석 시 이 정보를 모르는 상태로 분석
- 결과: 같은 화합물이 다른 이름으로 추출되거나, 정보가 제대로 매칭되지 않음

**해결 시도**:
- ✅ `contextual_extraction_pipeline.py` 생성: 각 단계의 결과를 누적하여 다음 단계에 전달
- ✅ `analyze_yolo_extracted_images.py` 수정: `use_conversation_history=True`로 대화 히스토리 유지
- ⚠️ **아직 완전히 해결되지 않음**: Excel/Word 추출 단계에서 컨텍스트 전달 미구현

**필요한 작업**:
- Excel/Word 추출 시에도 이전 컨텍스트를 참조하도록 수정
- 최종 통합 단계에서도 단계별 누적 정보를 활용

### 2. **화합물 추출 완전성 부족**
**문제점**:
- 보충자료에 178개 화합물이 있는데 2-3개만 추출됨
- GPT-4o 출력 토큰 제한(16,384)으로 인한 불완전한 추출

**해결 시도**:
- ✅ 배치 처리 구현: 화합물을 작은 그룹으로 나누어 여러 번 API 호출
- ✅ 프롬프트 개선: "ALL COMPOUND NAMES" 명시, exhaustive extraction 강조
- ✅ 결과: 2개 → 34개 → 162개 (85.4% 포함률)로 개선

**남은 문제**:
- 여전히 100% 추출이 안 되는 경우 있음
- 일부 논문에서 `LengthFinishReasonError` 발생 (출력 토큰 초과)

### 3. **화합물 이름 정규화 및 매칭**
**문제점**:
- 같은 화합물이 다른 이름으로 추출됨 (예: "CBK037537" vs "Compound 1")
- Coreference 분석이 완벽하지 않음

**해결 시도**:
- ✅ Coreference 분석 파이프라인 구축
- ✅ 최종 통합 단계에서 aliases 필드로 별칭 관리
- ⚠️ **아직 완전하지 않음**: SMILES/InChI 기반 매칭 미구현

### 4. **ADMET 지표 표준화**
**문제점**:
- 단위가 일관되지 않음 (µM vs uM vs μM)
- 같은 지표가 다른 이름으로 표현됨

**해결 시도**:
- ✅ 프롬프트에 정규화 규칙 명시
- ✅ Pydantic BaseModel로 스키마 강제
- ⚠️ **부분적 해결**: 여전히 일부 불일치 발생

---

## 🐛 현재 구체적인 문제점들

### 1. **컨텍스트 누적이 완전하지 않음**
- 이미지 분석은 대화 히스토리 유지 ✅
- Excel/Word 추출은 컨텍스트 미전달 ❌
- 최종 통합 단계에서 단계별 맥락 부족 ❌

### 2. **보충자료 다운로드 실패**
- MDPI, ACS 출판사에서 403 Forbidden 발생
- 일부 논문은 보충자료를 다운로드할 수 없음
- **영향**: 해당 논문들은 보충자료 정보 없이 처리됨

### 3. **출력 토큰 제한**
- GPT-4o 최대 출력 토큰: 16,384
- 화합물이 많은 논문(200개 이상)에서 출력이 잘림
- **해결책**: 배치 처리로 나누지만, 여전히 일부 손실 가능

### 4. **정보 소스 우선순위 불명확**
- 같은 화합물의 같은 지표가 여러 소스에서 다른 값으로 나타남
- 현재: supplement > image > text 우선순위
- **문제**: 충돌 해결 로직이 완벽하지 않음

### 5. **Excel/Word 추출 품질**
- Excel: 헤더 자동 감지가 완벽하지 않음
- Word: 표 추출이 복잡한 구조에서 실패
- **영향**: 보충자료 정보 누락

---

## 📊 현재 데이터 구조

### 입력 구조
```
data_test/
├── raws/PMC###/          # 원문 PDF
│   └── article.pdf
└── supp/PMC###/          # 보충자료
    ├── *.xlsx
    ├── *.docx
    └── *.pdf
```

### 중간 처리 결과
```
data_test/
├── text_extracted/PMC###/        # 텍스트 추출
├── graph_extracted/PMC###/      # YOLO 추출 (figures, tables)
├── graph_analyzed/PMC###/       # 이미지 분석 결과
├── supp_extracted/PMC###/       # 보충자료 추출
│   ├── excel/
│   ├── word/
│   ├── pdf_graph/
│   └── pdf_info/
└── entity_analyzed/PMC###/      # 코어퍼런스 분석
```

### 최종 출력
```
data_test/
└── final_extracted/PMC###/
    ├── PMC###_final_admet.json  # 구조화된 ADMET 데이터
    └── admet_indicators_only.csv # CSV 형식 (간소화)
```

---

## 🎯 다음 단계 우선순위

### 1. **컨텍스트 누적 완성** (최우선)
- [ ] Excel 추출 시 이전 컨텍스트 전달
- [ ] Word 추출 시 이전 컨텍스트 전달
- [ ] 최종 통합 단계에서 단계별 맥락 활용
- [ ] 각 단계의 결과를 실시간으로 컨텍스트에 누적

### 2. **추출 완전성 향상**
- [ ] 배치 크기 최적화 (현재 50개 → 동적 조정)
- [ ] 출력 토큰 초과 시 자동 재시도 로직
- [ ] 화합물 목록을 먼저 추출한 후 속성 추출 (2단계 방식)

### 3. **화합물 매칭 개선**
- [ ] SMILES/InChI 기반 자동 매칭
- [ ] Coreference 분석 정확도 향상
- [ ] 화합물 이름 정규화 강화

### 4. **보충자료 다운로드 개선**
- [ ] MDPI/ACS 접근 문제 해결 (User-Agent, Referer 조정)
- [ ] 대체 다운로드 방법 탐색
- [ ] GPT 기반 링크 추론 강화

### 5. **데이터 품질 검증**
- [ ] 추출된 데이터 검증 로직
- [ ] 누락된 화합물 감지
- [ ] 값의 일관성 검사

---

## 📝 기술 스택

### 핵심 라이브러리
- **PDF 처리**:
  - `pdfplumber` - PDF 텍스트 추출
  - `PyMuPDF` (fitz) - PDF를 이미지로 변환
- **이미지 처리**:
  - `ultralytics YOLO` - 객체 감지 (figure/table)
  - `PIL` (Pillow) - 이미지 로드/저장/인코딩
  - `OpenCV` (cv2) - 이미지 리사이즈 및 좌표 변환
- **LLM**:
  - `OpenAI API` (openai 패키지) - GPT-4o, GPT-4o Vision
  - `Ollama` - Llama 4 (코어퍼런스 분석, Excel 속성 정규화, 선택적)
- **데이터 처리**:
  - `pandas` - Excel/CSV 데이터 처리
  - `openpyxl` - Excel 파일 파싱
  - `python-docx` (docx) - Word 파일 처리
  - `Pydantic` - 구조화된 출력 스키마 정의 및 검증
- **기타**:
  - `requests` - HTTP 요청 (PDF/보충자료 다운로드)
  - `BeautifulSoup` - HTML 파싱 (보충자료 링크 추출)
  - `lxml` - XML 파싱 (JATS 메타데이터)
- **언어**: Python 3

---

## 🔍 테스트 현황

### 성공 사례
- **PMC7066191**: 162개 화합물 추출 (85.4% 포함률)
- **PMC7878295**: 보충자료 Excel에서 성공적으로 추출
- **PMC12006413**: 복잡한 보충자료 처리 성공

### 실패/부분 실패 사례
- **PMC10017499**: 보충자료 다운로드 실패
- **PMC10177590**: MDPI 403 오류
- **PMC12077378**: 출력 토큰 초과 (`LengthFinishReasonError`)

---

## 💡 핵심 인사이트

1. **단일 프롬프트로 모든 것을 추출하는 것의 한계**
   - 여러 단계로 나누고, 각 단계의 결과를 누적하는 것이 중요
   - 맥락 유지가 추출 품질에 결정적

2. **배치 처리의 필요성**
   - 화합물이 많은 논문은 반드시 배치로 나눠야 함
   - 배치 크기는 동적으로 조정 필요

3. **소스 간 정보 통합의 어려움**
   - 같은 화합물을 다른 이름으로 부르는 경우가 많음
   - SMILES/InChI 같은 구조식 기반 매칭이 필수

4. **프롬프트 엔지니어링의 중요성**
   - "ALL", "EXHAUSTIVE", "NEVER stop" 같은 키워드가 효과적
   - 구체적인 예시와 형식 명시가 필수

---

## 📌 다음 미팅/리뷰 시 논의할 사항

1. 컨텍스트 누적 완성도 평가
2. 추출 완전성 목표 설정 (현재 85% → 목표 95%+)
3. 보충자료 다운로드 실패 논문 처리 방안
4. 데이터 검증 및 품질 관리 프로세스 수립
5. 대규모 배치 처리 시 비용 최적화 방안

---

**최종 업데이트**: 2025-01-XX
**작성자**: AI Assistant (대화 내용 기반)

