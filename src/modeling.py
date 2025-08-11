# src/modeling.py
# -*- coding: utf-8 -*-
"""
모델링 유틸
- 학습 전처리(SMOTE / class_weight) + 스케일링
- LightGBM 모델 구성/학습
- 아티팩트 저장/로드(joblib)

주의:
- 트리 모델은 스케일링이 필수는 아니지만, 이후에 로지스틱/선형 등과 교체할 수도 있어
  일관성을 위해 스케일러를 두고, '훈련 데이터에만 fit' 원칙을 강제한다.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from joblib import dump, load


# -----------------------------
# 데이터 준비 결과 컨테이너
# -----------------------------
@dataclass
class TrainPrepResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    scaler: StandardScaler


# -----------------------------
# 전처리: 샘플링 + 스케일링
# -----------------------------
def prepare_datasets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    imbalance: Literal["smote", "class_weight", "none"] = "smote",
    random_state: int = 42
) -> TrainPrepResult:
    """
    훈련/검증/테스트 셋을 받아서, 불균형 처리와 스케일링을 적용한다.
    - 'smote': 훈련셋에만 SMOTE 적용
    - 'class_weight': 샘플 그대로, 모델 쪽에서 가중치로 처리
    - 'none': 아무 것도 안 함
    """
    X_tr, y_tr = X_train.copy(), y_train.copy()

    if imbalance == "smote":
        sm = SMOTE(random_state=random_state)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    elif imbalance in ("class_weight", "none"):
        pass
    else:
        raise ValueError("imbalance must be one of {'smote','class_weight','none'}")

    # 스케일러는 훈련 데이터에만 fit
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # y는 원본 인덱스 무시하고 연속 번호가 될 수 있음 -> Series로 재래핑
    y_tr_s = pd.Series(y_tr.values, name="target")

    return TrainPrepResult(
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_val=X_val_s,
        X_test=X_te_s,
        scaler=scaler
    )


# -----------------------------
# 모델 구성/학습
# -----------------------------
def make_lgbm(
    *,
    imbalance: Literal["smote", "class_weight", "none"] = "smote",
    random_state: int = 42,
    **kwargs: Any
) -> LGBMClassifier:
    """
    LightGBM 분류기 생성.
    - imbalance='class_weight' 이면 class_weight='balanced'로 설정
    - 나머지 하이퍼파라미터는 kwargs로 덮어쓰기
    """
    params: Dict[str, Any] = dict(
        random_state=random_state,
        n_estimators=200
    )
    if imbalance == "class_weight":
        params["class_weight"] = "balanced"

    params.update(kwargs)
    return LGBMClassifier(**params)


def train_model(model: LGBMClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """모델 학습."""
    model.fit(X_train, y_train)
    return model


# -----------------------------
# 추론 헬퍼
# -----------------------------
def predict_proba(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """양성 클래스(사기=1) 확률 벡터 반환."""
    return model.predict_proba(X)[:, 1]


# -----------------------------
# 저장/로드
# -----------------------------
def save_artifacts(
    *,
    model: LGBMClassifier,
    scaler: Optional[StandardScaler],
    out_dir: str | Path
) -> Tuple[Path, Optional[Path]]:
    """
    모델/스케일러를 outputs/models/ 아래에 저장.
    Returns: (model_path, scaler_path or None)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "model.joblib"
    dump(model, model_path)

    scaler_path = None
    if scaler is not None:
        scaler_path = out / "scaler.joblib"
        dump(scaler, scaler_path)

    return model_path, scaler_path


def load_artifacts(
    *,
    model_path: str | Path,
    scaler_path: Optional[str | Path] = None
) -> Tuple[LGBMClassifier, Optional[StandardScaler]]:
    """저장된 모델/스케일러 로드."""
    model = load(model_path)
    scaler = load(scaler_path) if scaler_path is not None and Path(scaler_path).exists() else None
    return model, scaler


def get_model_info(model):
    """
    모델의 기본 정보를 반환합니다.
    """
    model_type = type(model).__name__
    
    # 모델별 상세 정보
    if hasattr(model, 'n_estimators'):
        # Random Forest
        info = f"Random Forest (n_estimators={model.n_estimators})"
    elif hasattr(model, 'n_neighbors'):
        # KNN
        info = f"K-Nearest Neighbors (n_neighbors={model.n_neighbors})"
    elif hasattr(model, 'C'):
        # SVM
        info = f"Support Vector Machine (C={model.C})"
    elif hasattr(model, 'alpha'):
        # Ridge/Lasso
        info = f"{model_type} (alpha={model.alpha})"
    elif hasattr(model, 'max_depth'):
        # Decision Tree
        info = f"Decision Tree (max_depth={model.max_depth})"
    else:
        info = model_type
    
    return info
