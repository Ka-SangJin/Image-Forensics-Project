## My Contributions

### **Role: Data Preprocessing & Dataset Engineering (데이터 전처리 및 데이터셋 구축 담당)**
본 프로젝트에서 2-Stage 위조 탐지 파이프라인의 핵심 입력 데이터인 **30,000장 규모의 대용량 이미지/마스크 데이터셋 전처리 및 파이프라인 구축**을 전담하였습니다.

#### **주요 담당 업무 및 성과**
* **이종 데이터셋 수집 및 가공 (`step01_data_prep.py`)**:
  * DEFACTO(Splicing 위조 이미지 15,000장 + Probe Mask)와 MS COCO(정상 이미지 15,000장) 데이터셋 수집 및 통합 구조 설계
  * 정상 이미지의 학습을 위해 픽셀 전체가 0인 **'All-Zero Mask' 직접 생성** 및 정답 마스크 구축
* **데이터 누수(Data Leakage) 엄격 통제**:
  * 동일 원본 이미지에서 파생된 이미지들이 Train/Val/Test 셋에 섞여 성과가 오버패키징되는 문제를 막기 위해 **`group_id` 기반의 스플릿 통제 로직 구현** (Split 비율 7:2:1 유지)
* **이미지 비율 및 해상도 표준화 (Letterbox Resize)**:
  * 이미지의 비율 distort(왜곡)를 방지하면서 고정 해상도($256 \times 256$)로 패딩 처리하는 `letterbox_resize` 유틸리티 개발 및 연동
* **전처리 자동화 및 검증 모듈 구축**:
  * Colab 환경의 I/O 병목 및 마운트 연결 끊김 현상을 방지하기 위해 로컬 ZIP 복사 및 무결성 검증(경로 오류, 중복, 데이터 누수 체크) 자동화 스크립트 작성



## How to Run (실행 방법)

본 프로젝트는 **독립적인 Python 스크립트 실행 환경**과 **Google Colab 실행 환경**을 모두 지원합니다.

---

### 1. 환경 설정 (Installation)

필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
# 또는 주요 패키지 직접 설치
pip install segmentation-models-pytorch albumentations lightgbm shap scikit-image timm opencv-python-headless joblib

### 2. 파이프라인 단계별 실행

# Step 0: 개발 환경 준비 및 디렉토리 생성
python src/step00_setup.py

# Step 1: 데이터 압축 해제 및 dataset.csv 생성/검증
python src/step01_data_prep.py --defacto_zip path/to/defacto.zip --coco_zip path/to/coco.zip

# Step 2: Stage 1 U-Net 모델 학습 (Repeated Hold-out)
python src/step02_train_stage1.py

# Step 3: Stage 1 Heatmap 생성 및 16가지 포렌식 특징(Feature) 추출
python src/step03_heatmap_features.py

# Step 4: Stage 2 LightGBM 모델 5-Fold CV 학습 및 분류 최적화
python src/step04_train_stage2.py

# Step 5: 최종 Test 세트 평가, SHAP 분석 및 시각화 생성
python src/step05_evaluate.py
