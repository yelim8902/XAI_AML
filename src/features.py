# src/features.py
# -*- coding: utf-8 -*-
"""
누수 없는 피처 엔지니어링 유틸

핵심 설계
- 절대 '사후 정보(newbalance*)'는 쓰지 않음
- step(절대 시간)은 버리고, 주기성(hour)만 사용
- type은 고정 카테고리로 원-핫하여, 학습/추론 컬럼 일관성 보장
- Inf/NaN 안전 처리

주요 함수
- build_features(df): X, y 반환 (가장 간단한 진입점)
- get_feature_space(): 모델/스케일러 저장 시 함께 저장해두면, 재현성 ↑
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


# ------------------------------------------------------------
# 설정: 사용 컬럼/타입 카테고리 (고정)
# ------------------------------------------------------------
USE_COLS: List[str] = [
    "step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"
]

# PaySim 기준 주요 거래 타입. 순서는 중요하지 않지만, 고정하면 재현성 ↑
TYPE_CATEGORIES: List[str] = [
    "CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT", "PAYMENT"
]


@dataclass(frozen=True)
class FeatureSpace:
    """
    최종 피처 컬럼 스키마를 보관.
    - columns: 원-핫 포함 최종 학습용 컬럼 순서
    - type_categories: 'type' 카테고리 고정값(추론 시에도 사용)
    """
    columns: List[str]
    type_categories: List[str]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """0으로 나눌 때 폭발하지 않게 아주 작은 epsilon을 더해 계산."""
    eps = 1e-6
    r = num / (den + eps)
    # 극단값/무한대 방지
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r


def _one_hot_type(s: pd.Series, categories: List[str]) -> pd.DataFrame:
    """'type'을 고정 카테고리로 원-핫 인코딩(drop_first=True)."""
    ctype = CategoricalDtype(categories=categories)
    s = s.astype(ctype)
    dummies = pd.get_dummies(s, prefix="type", drop_first=True)
    return dummies


def _sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Inf/NaN은 0으로, dtype은 float로 캐스팅."""
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)
    return df


def _make_feature_frame(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """
    누수 없는 피처를 생성하고, 고정 카테고리 기준으로 type 원-핫을 붙인다.
    step은 버리고 hour만 쓴다.
    """
    X = df[USE_COLS].copy()

    # 1) 기본 파생
    X["amount_to_balance_ratio"] = _safe_ratio(X["amount"], X["oldbalanceOrg"])
    X["balance_zero_both"] = ((X["oldbalanceOrg"] == 0) & (X["oldbalanceDest"] == 0)).astype(int)
    X["hour"] = (X["step"] % 24).astype(int)

    # 2) 절대시간 제거
    X = X.drop(columns=["step"])

    # 3) type 원-핫(고정 카테고리)
    type_OH = _one_hot_type(X["type"], categories)
    X = pd.concat([X.drop(columns=["type"]), type_OH], axis=1)

    # 4) 안전화
    X = _sanitize_numeric(X)
    return X


def build_features(
    df: pd.DataFrame,
    *,
    type_categories: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, FeatureSpace]:
    """
    가장 간단한 진입점. X, y, feature_space를 한 번에 반환.

    Parameters
    ----------
    df : DataFrame
        PaySim 원본(또는 동일 스키마) 데이터프레임
    type_categories : List[str], optional
        거래 타입 카테고리를 강제 지정하고 싶을 때(기본은 TYPE_CATEGORIES)

    Returns
    -------
    X : DataFrame  (모델 입력용, 원-핫 포함)
    y : Series     (isFraud -> int)
    feature_space : FeatureSpace  (추후 추론 시 컬럼 정렬용)
    """
    cats = type_categories or TYPE_CATEGORIES

    # 타깃
    if "isFraud" not in df.columns:
        raise KeyError("입력 df에 'isFraud' 컬럼이 없습니다.")
    y = df["isFraud"].astype(int).copy()

    # 피처 생성
    X = _make_feature_frame(df, cats)

    # 피처 스페이스 저장(추론 시 정렬용)
    feature_space = FeatureSpace(columns=list(X.columns), type_categories=list(cats))
    return X, y, feature_space


def align_to_feature_space(
    X_new: pd.DataFrame,
    space: FeatureSpace
) -> pd.DataFrame:
    """
    새로운 데이터(추론/재학습)에 대해 컬럼을 학습 시점과 동일하게 맞춘다.
    - 없던 더미 컬럼은 0으로 채우고,
    - 추가로 생긴 컬럼은 버린다.
    """
    # 누락 컬럼 추가
    for c in space.columns:
        if c not in X_new.columns:
            X_new[c] = 0.0
    # 불필요 컬럼 제거 + 순서 정렬
    X_new = X_new.reindex(columns=space.columns)
    # NaN/Inf 방어
    X_new = _sanitize_numeric(X_new)
    return X_new


def get_feature_space(X: pd.DataFrame) -> FeatureSpace:
    """
    이미 만들어진 X에서 FeatureSpace만 뽑고 싶을 때 사용.
    """
    return FeatureSpace(columns=list(X.columns), type_categories=list(TYPE_CATEGORIES))


__all__ = [
    "USE_COLS",
    "TYPE_CATEGORIES",
    "FeatureSpace",
    "build_features",
    "align_to_feature_space",
    "get_feature_space",
]
