# 📄 Explainable AI 기반 이상거래 탐지 & STR 보고서 자동화 시스템

## 1. 프로젝트 개요

이 프로젝트는 **금융 이상거래 탐지(Fraud Detection)** 모델에 **설명 가능한 인공지능(Explainable AI, XAI)** 기법을 적용하고, 이를 바탕으로 **GPT 기반 STR(의심거래보고) 보고서**를 자동 생성하는 시스템입니다.

- **목표**: 높은 탐지 성능과 함께, “왜” 특정 거래가 이상하다고 판단되었는지에 대한 **투명하고 직관적인 설명** 제공
- **활용 기술**:
  - 머신러닝 기반 이상거래 탐지(RandomForest, GradientBoosting, XGBoost, LightGBM 등)
  - XAI 기법 (SHAP, LIME, Counterfactual Explanation)
  - GPT 기반 보고서 자동화

---

## 2. 프로젝트 구조

- XAI_AML
  - data
    - processed
    - raw *(gitignored)*
  - notebooks
    - data_preprocessing.ipynb
    - model_training.ipynb
    - xai_analysis.ipynb
  - src
    - save_model.py
    - enhanced_fraud_detection.py
    - xai_analysis.py
    - report_generator.py
    - utils.py
  - results
    - figures
    - metrics
  - models
  - requirements.txt
  - README.md
  - LICENSE


---

## 3. 데이터셋

본 프로젝트에서는 공개된 금융 이상거래 시뮬레이션 데이터셋을 사용합니다.

### (1) PaySim Dataset

- **출처**: Mobile Money Transactions 시뮬레이션 데이터
- **특징**:
  - 거래 ID, 금액, 거래유형, 송·수신 계좌, 잔액 변화 등 포함
  - 사기 거래 여부(`isFraud`) 및 시도된 사기 거래(`isFlaggedFraud`) 라벨 포함
- **전처리**:
  - 결측치 제거 및 이상값 처리
  - 범주형 변수 원-핫 인코딩
  - 파생 변수 생성 (잔액 변화율, 거래 빈도, 거래 시간 간격 등)
  - SMOTE를 통한 클래스 불균형 해결

### (2) IEEE-CIS Fraud Detection Dataset _(선택적)_

- **출처**: IEEE Computational Intelligence Society
- **특징**:
  - 대규모 e-커머스 결제 데이터
  - 수백 개의 익명화된 피처 포함
  - 복잡한 결제 패턴 기반 사기 탐지 가능

---

## 4. 주요 기능

1. **이상거래 탐지 모델 학습**
   - 앙상블 및 부스팅 기반 모델 적용
   - GridSearchCV로 하이퍼파라미터 튜닝
2. **XAI 분석**
   - SHAP: 전역·로컬 피처 중요도 분석
   - LIME: 개별 거래 예측 근거 설명
   - Counterfactual: “만약 ~였다면” 시나리오 분석
3. **STR 보고서 자동화**
   - XAI 분석 결과를 기반으로 GPT가 자동 문장 생성
   - 보고서 템플릿 기반 PDF 또는 DOCX 출력

---

## 5. 설치 & 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/username/XAI-Fraud-Detection.git
cd XAI-Fraud-Detection

# 2. 가상환경 생성 및 활성화
conda create -n xai_fds python=3.9
conda activate xai_fds

# 3. 필수 패키지 설치
pip install -r requirements.txt

# 4. 데이터 전처리 실행
python src/data_preprocessing.py

# 5. 모델 학습
python src/enhanced_fraud_detection.py

# 6. XAI 분석
python src/xai_analysis.py

# 7. STR 보고서 생성
python src/report_generator.py
```
