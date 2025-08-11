# src/xai.py
# -*- coding: utf-8 -*-
"""
XAI(SHAP) 유틸

핵심:
- get_explainer(model): TreeExplainer 우선, 실패 시 일반 Explainer
- explain(explainer, X): SHAP Explanation → (matrix, feature_names)로 표준화
- global_importance_df(shap_matrix, feature_names): 전역 중요도 DF
- local_topk(shap_matrix, X, row_indices, k, exclude): 거래별 Top-k 근거 튜플
- save_plots_bar_beeswarm(explanation, out_dir, prefix): bar/beeswarm 이미지 저장
- save_shap_features_csv(shap_matrix, feature_names, out_path): 피처 중요도 CSV 저장

주의:
- SHAP 미설치 시 RuntimeError 발생
- 이 모듈은 "계산"과 "파일 저장"만 담당. 리포트 문장화는 src/reporting.py 사용
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

import matplotlib.pyplot as plt


# ---------------------------
# 1) Explainer 생성
# ---------------------------
def get_explainer(model: Any):
    """
    트리 모델이면 TreeExplainer 우선, 아니면 일반 Explainer.
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap 패키지가 설치되어 있지 않습니다. `pip install shap` 후 사용하세요.")
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return shap.Explainer(model)


# ---------------------------
# 2) 설명 계산 → 표준화
# ---------------------------
def explain(explainer, X: pd.DataFrame) -> Tuple[Any, np.ndarray, List[str]]:
    """
    SHAP Explanation을 계산하고, (Explanation, shap_matrix, feature_names)를 반환.
    - 이진분류에서 class별 list가 떨어지면 양성 클래스(1) 축을 선택.
    """
    explanation = explainer(X)
    vals = getattr(explanation, "values", explanation)

    if isinstance(vals, list) or isinstance(vals, tuple):
        # (class0, class1, ...) → 일반적으로 양성 클래스가 마지막 혹은 1
        # 보수적으로 마지막 축을 사용
        vals = np.asarray(vals[-1])
    else:
        vals = np.asarray(vals)

    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    feature_names = list(X.columns)
    return explanation, vals, feature_names


# ---------------------------
# 3) 전역 중요도 DF
# ---------------------------
def global_importance_df(shap_matrix: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    """
    |Feature|MeanAbsSHAP| 형태의 DF 반환(내림차순).
    """
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    df = pd.DataFrame({"Feature": list(feature_names), "MeanAbsSHAP": mean_abs})
    return df.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)


# ---------------------------
# 4) 로컬 Top-k 근거 (사례별)
# ---------------------------
def local_topk(
    shap_matrix: np.ndarray,
    X: pd.DataFrame,
    row_indices: Iterable[int],
    *,
    k: int = 5,
    exclude_features: Optional[Iterable[str]] = None
) -> List[List[Tuple[str, float, Any]]]:
    """
    선택한 행들에 대해 (feature, shap_value, raw_value) Top-k 리스트 반환.
    """
    excl = set(exclude_features or [])
    kept_cols = [c for c in X.columns if c not in excl]
    kept_idx = [X.columns.get_loc(c) for c in kept_cols]

    sub_matrix = shap_matrix[np.array(list(row_indices))][:, kept_idx]
    sub_raw = X.iloc[list(row_indices)][kept_cols]

    results: List[List[Tuple[str, float, Any]]] = []
    for i in range(sub_matrix.shape[0]):
        row_vals = sub_matrix[i]
        order = np.argsort(-np.abs(row_vals))[: max(1, k)]
        triples = [(kept_cols[j], float(row_vals[j]), sub_raw.iloc[i, j]) for j in order]
        results.append(triples)
    return results


# ---------------------------
# 5) 플롯 저장 (bar / beeswarm)
# ---------------------------
def save_plots_bar_beeswarm(
    explanation: Any,
    out_dir: str | Path,
    *,
    prefix: str = "shap",
    dpi: int = 150
) -> Dict[str, str]:
    """
    shap.plots.bar / shap.plots.beeswarm 결과를 PNG로 저장.
    반환: {"bar": path, "beeswarm": path}
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    # Bar
    plt.figure()
    shap.plots.bar(explanation, show=False)
    p_bar = out / f"{prefix}_bar.png"
    plt.gcf().savefig(p_bar, dpi=dpi, bbox_inches="tight")
    plt.close()
    paths["bar"] = str(p_bar)

    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    p_bee = out / f"{prefix}_bee.png"
    plt.gcf().savefig(p_bee, dpi=dpi, bbox_inches="tight")
    plt.close()
    paths["beeswarm"] = str(p_bee)

    return paths


# ---------------------------
# 6) SHAP 피처 중요도 CSV 저장
# ---------------------------
def save_shap_features_csv(
    shap_matrix: np.ndarray,
    feature_names: Sequence[str],
    out_path: str | Path,
    *,
    top_k: Optional[int] = None
) -> str:
    """
    SHAP 피처 중요도를 CSV로 저장.
    
    Parameters
    ----------
    shap_matrix : np.ndarray
        SHAP 값 행렬
    feature_names : Sequence[str]
        피처 이름 리스트
    out_path : str | Path
        저장할 CSV 파일 경로
    top_k : Optional[int]
        상위 k개만 저장 (None이면 전체)
    
    Returns
    -------
    str : 저장된 파일 경로
    """
    # 전역 중요도 계산
    importance_df = global_importance_df(shap_matrix, feature_names)
    
    # top_k가 지정된 경우 상위 k개만 선택
    if top_k is not None:
        importance_df = importance_df.head(top_k)
    
    # CSV 저장
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_path, index=False)
    
    return str(out_path)


# ---------------------------
# 7) 개별 거래 SHAP 설명 생성
# ---------------------------
def generate_individual_explanations(
    shap_matrix: np.ndarray,
    X: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    sample_size: int = 10,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    개별 거래에 대한 SHAP 설명을 생성합니다.
    
    Parameters
    ----------
    shap_matrix : np.ndarray
        SHAP 값 행렬
    X : pd.DataFrame
        피처 데이터
    y_pred : np.ndarray
        모델 예측값
    y_true : np.ndarray
        실제 라벨
    sample_size : int
        설명할 거래 수 (기본값: 10)
    k : int
        각 거래당 표시할 상위 피처 수 (기본값: 5)
    
    Returns
    -------
    List[Dict[str, Any]]
        각 거래에 대한 설명 딕셔너리 리스트
    """
    explanations = []
    
    # 사기로 예측된 거래와 실제 사기 거래를 우선적으로 선택
    fraud_pred_indices = np.where(y_pred == 1)[0]
    fraud_true_indices = np.where(y_true == 1)[0]
    
    # 우선순위: 실제 사기 > 사기로 예측 > 나머지
    priority_indices = list(fraud_true_indices) + list(fraud_pred_indices)
    
    # 중복 제거하고 순서 유지
    seen = set()
    unique_indices = []
    for idx in priority_indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)
    
    # 나머지 인덱스 추가
    remaining_indices = [i for i in range(len(X)) if i not in seen]
    unique_indices.extend(remaining_indices)
    
    # sample_size만큼 선택
    selected_indices = unique_indices[:sample_size]
    
    for idx in selected_indices:
        # 해당 거래의 SHAP 값과 원본 값
        shap_values = shap_matrix[idx]
        raw_values = X.iloc[idx]
        
        # 상위 k개 피처 선택 (절댓값 기준)
        feature_importance = list(zip(X.columns, shap_values, raw_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_importance[:k]
        
        explanation = {
            'transaction_id': idx,
            'predicted': int(y_pred[idx]),
            'actual': int(y_true[idx]),
            'prediction_correct': y_pred[idx] == y_true[idx],
            'top_features': [
                {
                    'feature': feature,
                    'shap_value': float(shap_value),
                    'raw_value': float(raw_value),
                    'contribution': 'positive' if shap_value > 0 else 'negative'
                }
                for feature, shap_value, raw_value in top_features
            ]
        }
        explanations.append(explanation)
    
    return explanations


# ---------------------------
# 8) 개별 거래 SHAP 시각화 저장
# ---------------------------
def save_individual_shap_plots(
    explanation: Any,
    X: pd.DataFrame,
    row_indices: List[int],
    out_dir: str | Path,
    *,
    prefix: str = "individual_shap",
    dpi: int = 150
) -> List[str]:
    """
    선택된 거래들에 대한 개별 SHAP 시각화를 저장합니다.
    
    Parameters
    ----------
    explanation : Any
        SHAP Explanation 객체
    X : pd.DataFrame
        피처 데이터
    row_indices : List[int]
        시각화할 거래 인덱스 리스트
    out_dir : str | Path
        저장할 디렉토리
    prefix : str
        파일명 접두사
    dpi : int
        이미지 해상도
    
    Returns
    -------
    List[str]
        저장된 이미지 파일 경로 리스트
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # matplotlib 한글 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    saved_paths = []
    
    for i, idx in enumerate(row_indices):
        try:
            plt.figure(figsize=(10, 6))
            
            # 해당 거래에 대한 waterfall plot 생성
            shap.plots.waterfall(explanation[idx], show=False)
            
            # 제목 설정 (한글 대신 영어 사용)
            plt.title(f'Transaction #{idx} SHAP Explanation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 파일 저장
            file_path = out / f"{prefix}_transaction_{idx}.png"
            plt.gcf().savefig(file_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            
            saved_paths.append(str(file_path))
            
        except Exception as e:
            print(f"⚠️ 거래 #{idx} 시각화 실패: {e}")
            continue
    
    return saved_paths


__all__ = [
    "get_explainer",
    "explain",
    "global_importance_df",
    "local_topk",
    "save_plots_bar_beeswarm",
    "save_shap_features_csv",
    "generate_individual_explanations",
    "save_individual_shap_plots",
]
