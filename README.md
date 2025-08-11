# XAI_FDS_Project

금융 사기 탐지(FDS) 모델 개발 및 XAI 기반 의심거래 보고서 자동화 시스템

---

## 📌 프로젝트 개요

본 프로젝트는 금융 거래 데이터를 기반으로 **이상거래 탐지(Fraud Detection)** 모델을 학습하고,  
SHAP 기반 XAI 분석 결과를 포함한 **의심거래 보고서(STR)** 를 자동 생성하는 것을 목표로 한다.

---

## 📂 폴더 구조

```

XAI\_FDS\_Project/
├── data/
│   ├── 01\_raw/              # 원본 데이터 (변경 금지)
│   │   └── paysim.csv
│   └── 02\_processed/        # 전처리된 데이터
│       └── preprocessed\_data.csv
│
├── notebooks/               # 실험 및 분석용 노트북
│   ├── 01\_eda.ipynb
│   └── 02\_model\_prototyping.ipynb
│
├── outputs/                 # 모든 실행 결과물 통합 저장
│   ├── figures/             # 시각화 자료 (SHAP 플롯 등)
│   ├── metrics/             # 성능 지표 (JSON, PNG 등)
│   ├── models/              # 학습된 모델 (joblib 등)
│   └── reports/             # 최종 보고서 (STR\_Report.md 등)
│
├── scripts/                 # 실행 스크립트
│   ├── train.py             # 모델 학습 및 저장
│   ├── evaluate.py          # 모델 성능 평가
│   └── generate\_report.py   # STR 보고서 자동 생성
│
├── src/                     # 재사용 가능한 모듈
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── xai.py
│   └── reporting.py
│
└── README.md

```

---

## ⚙️ 실행 방법

### 1. 모델 학습

```bash
cd XAI_FDS_Project
python scripts/train.py --imbalance smote --n_estimators 200 --seed 42
```

**옵션 설명**

- `--imbalance smote` : 불균형 데이터 보정 방법 (smote / none)
- `--n_estimators 200` : 모델 트리 개수 설정
- `--seed 42` : 랜덤 시드 고정

---

### 2. 모델 평가

```bash
python scripts/evaluate.py
```

- Confusion Matrix, Classification Report, AUPRC 등 주요 지표 저장

---

### 3. STR 보고서 생성

```bash
python scripts/generate_report.py --analysis_period "2025-08-01 ~ 2025-08-11"
```

- 분석 기간 동안의 의심거래 사례 + SHAP Top Features 포함 보고서 생성

---

## 📊 결과물 예시

| 결과물     | 설명                         | 예시 경로                                       |
| ---------- | ---------------------------- | ----------------------------------------------- |
| 모델 파일  | 학습된 모델 객체             | `outputs/models/model.joblib`                   |
| 성능 지표  | JSON, Confusion Matrix PNG   | `outputs/metrics/classification_report.json`    |
| XAI 시각화 | SHAP Bar Plot, Beeswarm Plot | `outputs/figures/final_bar.png`                 |
| 보고서     | STR 보고서 (Markdown)        | `outputs/reports/STR_Report_YYYYMMDD_HHMMSS.md` |

---

### 📈 SHAP 분석 예시

![SHAP Bar Plot](outputs/figures/final_bar.png)

### 📄 STR 보고서 예시

![STR 보고서 예시](outputs/reports/sample_report.png)

---

## 🛠 환경 설정

```bash
conda create -n xai_env python=3.10
conda activate xai_env
pip install -r requirements.txt
```

---

## 📌 주요 특징

- LightGBM 기반 이진 분류 모델
- SMOTE를 활용한 데이터 불균형 보정
- SHAP 기반 XAI 설명 기능
- STR 보고서 자동 생성 및 저장
- 폴더 구조 기반 재현 가능성 보장

```

```
