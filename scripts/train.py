# scripts/train.py
# -*- coding: utf-8 -*-
"""
학습 스크립트
- 데이터 로드 → 피처 엔지니어링 → 시계열 분할
- 전처리(SMOTE/스케일링) → LightGBM 학습
- 검증셋에서 F1 최대 임계값 탐색 → 저장
- 모델/스케일러/임계값 저장(outputs/models)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from pathlib import Path
import argparse
from joblib import dump
from src import load_paysim, time_split, outputs_ready
from src.features import build_features
from src.modeling import prepare_datasets, make_lgbm, train_model
from src.evaluation import find_best_threshold_by_f1

def main(args):
    paths = outputs_ready()
    df = load_paysim()

    # 1) 피처
    X_all, y_all, fspace = build_features(df)

    # 2) 분할
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = time_split(
        X_all, y_all, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # 3) 전처리 & 학습
    prep = prepare_datasets(
        X_tr, y_tr, X_va, X_te, imbalance=args.imbalance, random_state=args.seed
    )
    model = make_lgbm(imbalance=args.imbalance, random_state=args.seed, n_estimators=args.n_estimators)
    model = train_model(model, prep.X_train, prep.y_train)

    # 4) 임계값(검증셋) 탐색 → 고정
    from src.modeling import predict_proba
    val_proba = predict_proba(model, prep.X_val)
    ts = find_best_threshold_by_f1(y_va, val_proba)
    best_thr = ts.threshold

    # 5) 저장
    models_dir = paths.models
    models_dir.mkdir(parents=True, exist_ok=True)
    dump(model, models_dir / "model.joblib")
    dump(prep.scaler, models_dir / "scaler.joblib")
    (models_dir / "threshold.txt").write_text(str(best_thr))
    # 피처 스페이스도 같이 저장(추론 시 컬럼 정렬용)
    dump(fspace, models_dir / "feature_space.joblib")

    print(f"[train] saved model → {models_dir/'model.joblib'}")
    print(f"[train] saved scaler → {models_dir/'scaler.joblib'}")
    print(f"[train] saved threshold → {best_thr:.6f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--imbalance", choices=["smote","class_weight","none"], default="smote")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    args = p.parse_args()
    main(args)
