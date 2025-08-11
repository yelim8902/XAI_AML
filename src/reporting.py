# src/reporting.py
# -*- coding: utf-8 -*-
"""
STR 자동보고서(XAI 포함) 유틸 모듈

기능:
1) build_case_examples_pro: 상위 사기 점수 거래 추출 + 거래별 SHAP top-k 묶기
2) render_reason_sentence: 거래별 의심 사유 문장화
3) generate_str_report_v2: 성능/규제 충족/케이스별 사유가 포함된 보고서 생성
4) generate_str_with_reasons: (1)~(3)을 한 번에 실행하고 파일로 저장

의존:
- pandas, numpy
- (선택) shap  (사전 계산 shap_values를 넘기면 explainer 없이도 동작)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os
import math
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ------------------------------
# 공통 유틸
# ------------------------------
PII_DEFAULT = {"ssn", "phone", "account_no", "account", "customer_id", "name", "email"}

def _fmt_num(x: Any) -> str:
    try:
        if x is None:
            return "None"
        xv = float(x)
        if not math.isfinite(xv):
            return str(x)
        if abs(xv) >= 1000:
            return f"{xv:,.4g}"
        if abs(xv) >= 1:
            return f"{xv:,.4g}"
        return f"{xv:.6g}"
    except Exception:
        return str(x)

def _dir_from_contrib(val: Optional[float]) -> str:
    return "증가" if (val is not None and not np.isnan(val) and val >= 0) else "감소"

def _mask_id(v: Any, keep: int = 4) -> str:
    s = str(v)
    if len(s) <= keep:
        return "*" * len(s)
    return "*" * (len(s) - keep) + s[-keep:]

def _risk_tag(score: float) -> str:
    if score >= 0.95: return "CRITICAL"
    if score >= 0.85: return "HIGH"
    if score >= 0.70: return "MEDIUM"
    return "LOW"

def _ensure_1d(a: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    return arr


# ------------------------------
# 1) 사례 생성: 점수 상위 N + SHAP top-k
# ------------------------------
@dataclass
class CaseExample:
    id: str
    idx: int
    score: float
    risk_tag: str
    top_features: List[Tuple[str, float, Any]]  # (feature, shap, raw_value)

def build_case_examples_pro(
    *,
    X: pd.DataFrame,
    model: Optional[Any] = None,
    y_proba: Optional[Union[np.ndarray, Sequence[float]]] = None,
    shap_explainer: Optional[Any] = None,
    shap_values: Optional[np.ndarray] = None,
    id_series: Optional[Union[pd.Series, Sequence[Any]]] = None,
    top_n: int = 5,
    top_k_features: int = 5,
    exclude_features: Optional[Iterable[str]] = None,
    mask_ids: bool = True,
    timeout_sec: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    상위 사기 점수 거래를 추출하고 각 거래별 SHAP 기여 top-k를 묶어서 반환.

    Returns:
        case_examples: List[dict]  (직렬화-friendly)
        debug_info:    dict        (선택 로그/메타)
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    n, m = X.shape
    if n == 0:
        return [], {"msg": "empty X"}

    # 1) 점수 계산
    if y_proba is not None:
        scores = _ensure_1d(y_proba)
        if scores.shape[0] != n:
            raise ValueError("Length of y_proba must match X rows")
    else:
        if model is None:
            raise ValueError("Either y_proba or model must be provided")
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X)
            raw = _ensure_1d(raw)
            # 로지스틱 approx
            scores = 1 / (1 + np.exp(-raw))
        else:
            # 확률이 아닌 경우 0/1로 온다면 그대로 사용
            pred = model.predict(X)
            scores = _ensure_1d(pred).astype(float)

    scores = np.clip(scores, 0.0, 1.0)
    order = np.argsort(-scores)[: max(1, top_n)]
    X_sel = X.iloc[order]
    scores_sel = scores[order]

    # 2) ID 시리즈
    if id_series is not None:
        ids_all = pd.Series(id_series)
        if ids_all.shape[0] != n:
            raise ValueError("Length of id_series must match X rows")
        ids_sel = [ids_all.iloc[i] for i in order]
    else:
        # 인덱스를 ID로 사용
        ids_sel = list(X.index[order].astype(str))

    # 3) SHAP 값 준비: 전달되었으면 슬라이싱, 아니면 explainer 호출
    #    효율을 위해 상위 N에 대해서만 계산
    features = list(X.columns)
    excl = set((exclude_features or [])) | PII_DEFAULT  # 기본 PII도 제외
    keep_cols = [c for c in features if c not in excl]

    if shap_values is not None:
        if shap_values.shape[0] != n:
            raise ValueError("shap_values row count must match X rows")
        shap_sel = shap_values[order, :]
        shap_df = pd.DataFrame(shap_sel, columns=features).loc[:, keep_cols]
    else:
        if shap_explainer is None:
            if not _HAS_SHAP:
                raise RuntimeError("shap이 설치되어 있지 않거나 explainer가 없습니다.")
            # 가능한 경우 모델 기반 explainer 구성
            try:
                shap_explainer = shap.TreeExplainer(model)
            except Exception:
                shap_explainer = shap.Explainer(model)
        # 일부 explainer는 DataFrame을 그대로 받는다.
        shap_vals = shap_explainer(X_sel)
        # shap.Explanation -> values
        vals = getattr(shap_vals, "values", shap_vals)
        # (n_cases, n_features)
        if isinstance(vals, list):  # 일부 모델에서 class별 리스트로 나옴
            # 양성 클래스 기준 선택
            vals = np.array(vals[-1])
        shap_df = pd.DataFrame(vals, columns=features).loc[:, keep_cols]

    # 4) 거래별 top-k 기여 추출
    case_examples: List[Dict[str, Any]] = []
    for row_idx, (global_idx, score, raw_id) in enumerate(zip(order, scores_sel, ids_sel)):
        id_str = _mask_id(raw_id) if mask_ids else str(raw_id)

        row_shap = shap_df.iloc[row_idx]  # keep_cols만
        row_raws = X_sel.iloc[row_idx].loc[keep_cols]

        # 절대값 기준 상위 k
        abs_order = np.argsort(-np.abs(row_shap.values))[: max(1, top_k_features)]
        top_feats: List[Tuple[str, float, Any]] = []
        for j in abs_order:
            fname = keep_cols[j]
            shap_val = float(row_shap.iat[j])
            raw_val = row_raws.iat[j]
            top_feats.append((fname, shap_val, raw_val))

        case_examples.append({
            "id": id_str,
            "idx": int(global_idx),
            "score": float(score),
            "risk_tag": _risk_tag(float(score)),
            "top_features": top_feats
        })

    dbg = {
        "selected_indices": order.tolist(),
        "selected_scores": [float(s) for s in scores_sel],
        "excluded_features": sorted(list(excl)),
        "kept_features": keep_cols,
        "n_total": n,
        "n_selected": len(case_examples),
    }
    return case_examples, dbg


# ------------------------------
# 2) 사유 문장화
# ------------------------------
def render_reason_sentence(
    top_features: List[Tuple[str, Optional[float], Any]],
    max_len: int = 3
) -> str:
    """
    예: "`amount` 값(9,500,000)이 사기 가능성을 증가시키는 방향으로 0.315 기여; ..."
    """
    chunks: List[str] = []
    for (fname, contrib, raw_val) in top_features[:max_len]:
        dir_kor = _dir_from_contrib(contrib)
        if (contrib is None) or (isinstance(contrib, float) and (np.isnan(contrib))):
            contrib_txt = "기여도 미측정"
        else:
            contrib_txt = f"{abs(float(contrib)):.3f} 기여"
        chunks.append(f"`{fname}` 값({_fmt_num(raw_val)})이 사기 가능성을 {dir_kor}시키는 방향으로 {contrib_txt}")
    return "; ".join(chunks) if chunks else "(근거 미기재)"


# ------------------------------
# 3) 보고서 생성
# ------------------------------
def generate_str_report_v2(
    report: Dict[str, Dict[str, float]],
    conf_matrix: np.ndarray,
    shap_features: Optional[pd.DataFrame],
    threshold: float,
    *,
    meta: Dict[str, Any],
    case_examples: List[Dict[str, Any]],
    regulatory_criteria: Optional[Dict[str, float]] = None
) -> str:
    """
    거래별 의심 사유(로컬 SHAP)까지 포함한 STR 보고서 텍스트 생성
    """
    # 성능 지표
    precision = report["Fraud"]["precision"]
    recall = report["Fraud"]["recall"]
    f1 = report["Fraud"]["f1-score"]
    support = int(report["Fraud"]["support"])
    fp = int(conf_matrix[0, 1])
    fn = int(conf_matrix[1, 0])

    # 메타
    analysis_period = meta.get("analysis_period", "기간 미기재")
    dataset_name = meta.get("dataset", "데이터셋 미기재")
    model_name = meta.get("model", "모델 미기재")
    xai_methods = ", ".join(meta.get("xai", [])) if meta.get("xai") else "미기재"

    # 규제 기준
    regs = regulatory_criteria or {}
    recall_min = regs.get("recall_min")
    precision_min = regs.get("precision_min")
    f1_min = regs.get("f1_min")

    lines: List[str] = []
    lines.append("=" * 62)
    lines.append("        의심거래 분석(STR) 자동 생성 보고서")
    lines.append("=" * 62)
    lines.append("")
    lines.append(f"분석 기간: {analysis_period}")
    lines.append(f"데이터셋: {dataset_name}")
    lines.append(f"사용 모델: {model_name}")
    lines.append(f"XAI 기법: {xai_methods}")
    lines.append("")
    lines.append("### 1. 분석 개요")
    lines.append(f"- 최종 모델 성능: Precision={precision:.1%}, Recall={recall:.1%}, F1={f1:.1%}, 임계값={threshold:.4f}")
    lines.append("")
    lines.append("### 2. 모델 성능 상세")
    lines.append(f"- 실제 사기 거래 {support:,}건 중 {recall:.1%} 탐지")
    lines.append(f"- 오탐(False Positive): {fp:,}건, 미탐(False Negative): {fn:,}건")
    lines.append("")

    lines.append("### 3. 규제/내부 기준 충족 여부")
    if recall_min is not None:
        lines.append(f"- Recall 기준 ({recall_min:.0%} 이상): {'✅ 충족' if recall >= recall_min else '❌ 미충족'}")
    if precision_min is not None:
        lines.append(f"- Precision 기준 ({precision_min:.0%} 이상): {'✅ 충족' if precision >= precision_min else '❌ 미충족'}")
    if f1_min is not None:
        lines.append(f"- F1 기준 ({f1_min:.0%} 이상): {'✅ 충족' if f1 >= f1_min else '❌ 미충족'}")
    lines.append("")

    # 전역 SHAP 상위 피처(선택)
    if shap_features is not None and not shap_features.empty and {"Feature", "MeanAbsSHAP"}.issubset(shap_features.columns):
        top_rows = shap_features.sort_values("MeanAbsSHAP", ascending=False).head(5)
        tfmt = ", ".join([f"{r.Feature}({_fmt_num(r.MeanAbsSHAP)})" for r in top_rows.itertuples()])
        lines.append("### 4. 전역 탐지 패턴 (SHAP)")
        lines.append(f"- 중요 피처 Top-5: {tfmt}")
        lines.append("")

    # 케이스별 상세
    lines.append("### 5. 의심거래 상세 분석 (거래별 탐지 사유)")
    for case in case_examples:
        tx_id = case["id"]
        score = case["score"]
        risk = case["risk_tag"]
        # reason: 없으면 지금 생성
        reason = case.get("reason")
        if not reason:
            reason = render_reason_sentence(case.get("top_features", []), max_len=3)
        lines.append(f"- 거래 {tx_id} (사기 점수 {score:.3f}, 위험등급 {risk}): {reason}")
    lines.append("")

    lines.append("### 6. 결론 및 제언")
    lines.append("- 본 보고서는 로컬 SHAP 해석을 통해 각 거래의 탐지 사유를 명시하였으며, STR 제출 시 근거로 활용 가능합니다.")
    lines.append("- HIGH 이상 위험 등급 거래는 즉시 검토 및 내부 확인 절차(고객 연락/거래 일시 중지)를 권고합니다.")
    lines.append("- 새로운 사기 패턴 반영을 위해 임계값 모니터링과 분기별 재학습을 권장합니다.")
    return "\n".join(lines)


# ------------------------------
# 4) E2E: 사례 생성 -> 문장화 -> 보고서 생성 -> 저장
# ------------------------------
def generate_str_with_reasons(
    *,
    report_dict: Dict[str, Dict[str, float]],
    conf_matrix: np.ndarray,
    shap_features_df: Optional[pd.DataFrame],
    best_threshold: float,
    # 사례 생성 입력
    X_eval: pd.DataFrame,
    model: Optional[Any] = None,
    y_proba: Optional[Union[np.ndarray, Sequence[float]]] = None,
    shap_explainer: Optional[Any] = None,
    shap_values: Optional[np.ndarray] = None,
    tx_ids: Optional[Union[pd.Series, Sequence[Any]]] = None,
    # 리포트 메타/옵션
    meta: Dict[str, Any] = None,
    regulatory_criteria: Optional[Dict[str, float]] = None,
    top_n: int = 5,
    top_k_features: int = 5,
    exclude_features: Optional[Iterable[str]] = None,
    out_dir: str = "../outputs/reports",
    file_prefix: str = "STR_Report",
    mask_ids: bool = True
) -> Tuple[str, str, Dict[str, Any]]:
    """
    (1) 상위 N 사례 생성 -> (2) 사유 문장화 -> (3) 보고서 생성 -> (4) 파일 저장
    Returns: (report_text, out_path, meta_info)
    """
    meta = meta or {}
    case_examples, dbg = build_case_examples_pro(
        X=X_eval,
        model=model,
        y_proba=y_proba,
        shap_explainer=shap_explainer,
        shap_values=shap_values,
        id_series=tx_ids,
        top_n=top_n,
        top_k_features=top_k_features,
        exclude_features=exclude_features,
        mask_ids=mask_ids
    )

    # 문장화 필드 채움
    for c in case_examples:
        c["reason"] = render_reason_sentence(c.get("top_features", []), max_len=min(3, top_k_features))

    # 보고서 텍스트 생성
    report_text = generate_str_report_v2(
        report=report_dict,
        conf_matrix=conf_matrix,
        shap_features=shap_features_df,
        threshold=best_threshold,
        meta=meta,
        case_examples=case_examples,
        regulatory_criteria=(regulatory_criteria or {})
    )

    # 저장
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{file_prefix}_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text, out_path, {"examples_meta": dbg, "saved_at": out_path}


__all__ = [
    "build_case_examples_pro",
    "render_reason_sentence",
    "generate_str_report_v2",
    "generate_str_with_reasons",
]
