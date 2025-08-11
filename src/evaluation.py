# src/evaluation.py
# -*- coding: utf-8 -*-
"""
평가/임계값 탐색/검증 유틸

포함 기능
- find_best_threshold_by_f1: PR 커브 기반 F1 최대 임계값 탐색(검증셋)
- evaluate_at: 주어진 임계값으로 리포트/혼동행렬/AUPRC 산출(테스트셋)
- sanity_label_shuffle: 라벨 셔플로 정보 누수 여부 점검
- stress_by_types: 특정 거래 type 서브셋에 대한 스트레스 테스트
- save_metrics: 리포트/혼동행렬/임계값/AUPRC를 파일로 저장

주의
- 임계값은 반드시 '검증셋'에서 고정해서 '테스트셋'에 적용(정보 누수 방지)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    average_precision_score,
)

# -----------------------------
# 컨테이너
# -----------------------------
@dataclass
class ThresholdSearch:
    threshold: float
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray
    f1_scores: np.ndarray
    best_index: int


@dataclass
class EvalResult:
    report_dict: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray  # shape (2,2) -> [[tn, fp],[fn, tp]]
    auprc: float
    y_proba: np.ndarray  # (n,)
    y_pred: np.ndarray   # (n,)
    threshold: float


# -----------------------------
# 1) 임계값 탐색 (검증셋)
# -----------------------------
def find_best_threshold_by_f1(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
) -> ThresholdSearch:
    """
    PR 커브에서 F1을 최대화하는 임계값을 반환.
    - y_true: {0,1}
    - y_proba: [0,1] 확률
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).ravel()

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # scikit-learn은 thresholds 길이가 (len(precisions)-1)
    # f1 계산 시 0으로 나눔 방지
    denom = (precisions + recalls)
    denom[denom == 0] = 1e-12
    f1 = 2 * (precisions * recalls) / denom

    # 마지막 포인트는 threshold가 없을 수 있음 -> f1[:-1] 대상
    valid_len = len(thresholds)
    best_idx = int(np.nanargmax(f1[:valid_len]))
    best_thr = float(thresholds[best_idx])

    return ThresholdSearch(
        threshold=best_thr,
        precisions=precisions,
        recalls=recalls,
        thresholds=thresholds,
        f1_scores=f1,
        best_index=best_idx,
    )


# -----------------------------
# 2) 최종 평가 (테스트셋)
# -----------------------------
def evaluate_at(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    *,
    target_names: Tuple[str, str] = ("Not Fraud", "Fraud"),
    digits: int = 4
) -> EvalResult:
    """
    고정 임계값으로 이진 예측을 만들고 성능을 계산.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).ravel()
    y_pred = (y_proba >= float(threshold)).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=list(target_names),
        digits=digits, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    auprc = float(average_precision_score(y_true, y_proba))

    return EvalResult(
        report_dict=report,
        confusion_matrix=cm,
        auprc=auprc,
        y_proba=y_proba,
        y_pred=y_pred,
        threshold=float(threshold),
    )


# -----------------------------
# 3) 누수 점검: 라벨 셔플
# -----------------------------
def sanity_label_shuffle(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    *,
    random_state: int = 42,
    tol: float = 0.01
) -> Dict[str, float | bool]:
    """
    y_true를 셔플했을 때 AUPRC가 양성 비율(=무작위 수준)과 유사하면 OK.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).ravel()

    rng = np.random.RandomState(random_state)
    y_shuf = y_true.copy()
    rng.shuffle(y_shuf)

    auprc_orig = float(average_precision_score(y_true, y_proba))
    auprc_shuf = float(average_precision_score(y_shuf, y_proba))
    pos_ratio = float(y_true.mean())

    ok = abs(auprc_shuf - pos_ratio) < tol
    return {
        "auprc_original": auprc_orig,
        "auprc_shuffled": auprc_shuf,
        "positive_ratio": pos_ratio,
        "ok": bool(ok),
        "tolerance": float(tol),
    }


# -----------------------------
# 4) 타입별 스트레스 테스트
# -----------------------------
def stress_by_types(
    df_original: pd.DataFrame,
    test_indices: Iterable[int],
    y_true_test: np.ndarray | pd.Series,
    y_pred_test: np.ndarray | pd.Series,
    *,
    type_col: str = "type",
    focus_types: Optional[List[str]] = None,
    digits: int = 4
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    테스트 구간에서 특정 거래 타입 서브셋의 리포트를 산출.
    Returns:
      {"TRANSFER": {classification_report dict}, "CASH_OUT": {...}, ...}
    """
    if focus_types is None:
        focus_types = ["TRANSFER", "CASH_OUT"]

    test_idx = pd.Index(test_indices)
    sub_df = df_original.loc[test_idx, :]

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for t in focus_types:
        mask = (sub_df[type_col] == t).values
        if mask.sum() == 0:
            continue
        rep = classification_report(
            np.asarray(y_true_test)[mask].astype(int),
            np.asarray(y_pred_test)[mask].astype(int),
            target_names=["Not Fraud", "Fraud"],
            digits=digits,
            output_dict=True
        )
        out[t] = rep
    return out


# -----------------------------
# 5) 산출물 저장
# -----------------------------
def save_metrics(
    *,
    result: EvalResult,
    out_dir: str | Path,
    prefix: str = ""
) -> Dict[str, str]:
    """
    리포트/혼동행렬/AUPRC/임계값을 저장한다.
    - report: JSON
    - confusion_matrix: CSV (헤더와 인덱스 포함)
    - threshold: TXT
    - auprc: TXT
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = (prefix + "_") if prefix else ""

    report_path = out / f"{name}classification_report.json"
    cm_path = out / f"{name}confusion_matrix.csv"
    thr_path = out / f"{name}threshold.txt"
    auprc_path = out / f"{name}auprc.txt"

    report_path.write_text(json.dumps(result.report_dict, indent=2, ensure_ascii=False))
    
    # 혼동행렬을 DataFrame으로 변환하여 헤더와 인덱스 포함하여 저장
    cm_df = pd.DataFrame(
        result.confusion_matrix,
        index=['Not Fraud', 'Fraud'],
        columns=['Not Fraud', 'Fraud']
    )
    cm_df.to_csv(cm_path)
    
    thr_path.write_text(str(result.threshold))
    auprc_path.write_text(f"{result.auprc:.6f}")

    return {
        "report": str(report_path),
        "confusion_matrix": str(cm_path),
        "threshold": str(thr_path),
        "auprc": str(auprc_path),
    }


__all__ = [
    "ThresholdSearch",
    "EvalResult",
    "find_best_threshold_by_f1",
    "evaluate_at",
    "sanity_label_shuffle",
    "stress_by_types",
    "save_metrics",
]
