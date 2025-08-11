# scripts/generate_report.py

import sys
import os
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.reporting import generate_str_report
from src.analysis import analyze_and_save_shap_plots

def run_evaluation_and_reporting():
    """저장된 모델로 평가 및 보고서 생성을 수행하는 파이프라인"""
    print("--- 평가 및 보고서 생성 파이프라인 시작 ---")

    # 1. 필요한 파일들 로드
    try:
        model = joblib.load(os.path.join(config.MODEL_PATH, 'fds_model.pkl'))
        scaler = joblib.load(os.path.join(config.MODEL_PATH, 'scaler.pkl'))
        X = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'features.csv'))
        y = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'labels.csv')).squeeze()
        print("모델, 스케일러, 데이터를 로드했습니다.")
    except FileNotFoundError:
        print("Error: 모델 파일 등을 찾을 수 없습니다. 훈련 파이프라인(train_model.py)을 먼저 실행하세요.")
        return

    # 2. 검증 및 테스트 데이터 준비
    train_end_idx = int(X.shape[0] * 0.6)
    val_end_idx = train_end_idx + int(X.shape[0] * 0.2)
    X_val, y_val = X.iloc[train_end_idx:val_end_idx], y.iloc[train_end_idx:val_end_idx]
    X_test, y_test = X.iloc[val_end_idx:], y.iloc[val_end_idx:]

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 3. 최적 임계값 계산
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"계산된 최적 임계값: {best_threshold:.4f}")

    # 4. 최종 평가
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_final = (y_test_proba >= best_threshold).astype(int)
    report_dict = classification_report(y_test, y_pred_final, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_final)

    # 5. SHAP 분석 및 보고서 생성
    shap_df = analyze_and_save_shap_plots(model, X_test_scaled, config.FIGURES_PATH)
    
    final_report_text = generate_str_report(report_dict, cm, shap_df, best_threshold)
    os.makedirs(config.REPORTS_PATH, exist_ok=True)
    with open(os.path.join(config.REPORTS_PATH, 'STR_Report.txt'), 'w', encoding='utf-8') as f:
        f.write(final_report_text)
    print(f"✅ 최종 STR 보고서가 '{config.REPORTS_PATH}'에 저장되었습니다.")
    print("--- 평가 및 보고서 생성 파이프라인 종료 ---")

# <<< 해결책: 스크립트가 직접 실행될 때 run_evaluation_and_reporting() 함수를 호출하는 '실행 버튼'
if __name__ == "__main__":
    run_evaluation_and_reporting()