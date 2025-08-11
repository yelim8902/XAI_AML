# scripts/evaluate.py
# -*- coding: utf-8 -*-
"""
평가 스크립트
- 저장된 모델/스케일러/임계값 로드
- 동일 데이터로 피처 생성 + 시계열 분할
- 테스트셋에서 최종 성능 계산
- SHAP 분석 수행 및 피처 중요도 저장
- 리포트/혼동행렬/AUPRC/임계값 저장(outputs/metrics)
- 누수 점검(라벨 셔플) 결과도 출력
"""


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from joblib import load
from src import load_paysim, time_split, outputs_ready
from src.features import build_features, align_to_feature_space
from src.evaluation import evaluate_at, save_metrics, sanity_label_shuffle
from src.modeling import predict_proba
from src.xai import get_explainer, explain, save_shap_features_csv, generate_individual_explanations, save_individual_shap_plots

def main(args):
    paths = outputs_ready()

    # 1) 데이터/피처/분할
    df = load_paysim()
    X_all, y_all, fspace = build_features(df)

    # 추론 컬럼 정렬(혹시나를 대비)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = time_split(X_all, y_all)
    X_te = align_to_feature_space(X_te.copy(), fspace)

    # 2) 아티팩트 로드
    model = load(paths.models / "model.joblib")
    scaler = load(paths.models / "scaler.joblib")
    best_thr = float((paths.models / "threshold.txt").read_text())

    # 3) 스케일링 후 예측
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
    y_proba = predict_proba(model, X_te_s)

    # 4) 평가
    er = evaluate_at(y_te.values, y_proba, best_thr)
    saved = save_metrics(result=er, out_dir=paths.metrics)

    # 원한다면 proba/예측도 저장
    np.savetxt(paths.metrics / "y_proba_test.csv", y_proba, delimiter=",")
    np.savetxt(paths.metrics / "y_pred_test.csv", er.y_pred, fmt="%d", delimiter=",")

    # 5) SHAP 분석 수행 및 피처 중요도 저장
    print("🔍 SHAP 분석 수행 중...")
    try:
        explainer = get_explainer(model)
        explanation, shap_matrix, feature_names = explain(explainer, X_te_s)
        
        # SHAP 피처 중요도 CSV 저장
        shap_csv_path = paths.figures / "shap_feature_importance.csv"
        save_shap_features_csv(shap_matrix, feature_names, shap_csv_path)
        print(f"✅ SHAP 피처 중요도 저장: {shap_csv_path}")
        
        # 개별 거래 SHAP 설명 생성
        print("🔍 개별 거래 SHAP 설명 생성 중...")
        y_pred = (y_proba > best_thr).astype(int)
        individual_explanations = generate_individual_explanations(
            shap_matrix, X_te_s, y_pred, y_te.values, sample_size=5, k=5
        )
        
        # 개별 거래 SHAP 시각화 저장
        print("📊 개별 거래 SHAP 시각화 저장 중...")
        sample_indices = [exp['transaction_id'] for exp in individual_explanations[:3]]  # 상위 3개만 시각화
        individual_plot_paths = save_individual_shap_plots(
            explanation, X_te_s, sample_indices, paths.figures, prefix="individual_shap"
        )
        print(f"✅ 개별 거래 SHAP 시각화 저장: {len(individual_plot_paths)}개 파일")
        
        # SHAP 시각화도 저장
        from src.xai import save_plots_bar_beeswarm
        plot_paths = save_plots_bar_beeswarm(explanation, paths.figures, prefix="shap")
        print(f"✅ SHAP 시각화 저장: {plot_paths}")
        
    except Exception as e:
        print(f"⚠️ SHAP 분석 실패: {e}")

    # 6) 누수 점검(셔플)
    sanity = sanity_label_shuffle(y_te.values, y_proba)
    (paths.metrics / "sanity_shuffle.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False))

    print(f"[eval] AUPRC={er.auprc:.6f}, threshold={er.threshold:.6f}")
    print(f"[eval] saved metrics → {saved}")
    print(f"[eval] leakage_ok={sanity['ok']} | sanity={sanity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
