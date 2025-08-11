# src/data.py
# -*- coding: utf-8 -*-
"""
데이터 로드/경로/분할 유틸리티

- 경로 관리: 프로젝트 루트/데이터/산출물 경로를 일관되게 제공
- 원본 로드: PaySim CSV 로드(경로 검증 포함)
- 시계열 분할: Train/Val/Test 순서 보존 분할
- 산출물 디렉터리 보장: outputs/* 폴더 자동 생성
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


# ------------------------------
# 경로 설정
# ------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    raw: Path
    processed: Path
    outputs: Path
    figures: Path
    metrics: Path
    models: Path
    reports: Path

    @staticmethod
    def discover(start: Path | None = None) -> "Paths":
        """
        src/data.py 기준으로 프로젝트 루트를 탐색한다.
        기본 가정: 이 파일은 <PROJECT_ROOT>/src/data.py 에 있다.
        """
        if start is None:
            start = Path(__file__).resolve()
        root = start.parents[1]  # <PROJECT_ROOT>
        data = root / "data"
        raw = data / "raw"
        processed = data / "processed"
        outputs = root / "outputs"
        figures = outputs / "figures"
        metrics = outputs / "metrics"
        models = outputs / "models"
        reports = outputs / "reports"
        return Paths(root, data, raw, processed, outputs, figures, metrics, models, reports)

    def ensure_output_dirs(self) -> None:
        """outputs 하위 폴더를 모두 만들어 둔다(존재하면 무시)."""
        for p in [self.outputs, self.figures, self.metrics, self.models, self.reports]:
            p.mkdir(parents=True, exist_ok=True)


# 전역 경로 객체 (필요 시 import 해서 사용)
PATHS = Paths.discover()


# ------------------------------
# 원본 데이터 로드
# ------------------------------
def load_paysim(csv_name: str = "PS_20174392719_1491204439457_log.csv") -> pd.DataFrame:
    """
    PaySim 원본 CSV를 로드한다.
    파일은 기본적으로 data/raw/ 아래에 있어야 한다.

    Parameters
    ----------
    csv_name : str
        파일명(기본값은 PaySim 공식 파일명)

    Returns
    -------
    pd.DataFrame
    """
    csv_path = PATHS.raw / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[data.load_paysim] 원본 파일을 찾을 수 없음: {csv_path}\n"
            f"→ 파일을 'data/raw/' 폴더에 넣어주세요."
        )
    df = pd.read_csv(csv_path)
    return df


# ------------------------------
# 시계열 분할 (순서 보존)
# ------------------------------
def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> Tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """
    데이터 순서를 보존한 채 Train/Val/Test로 분할한다.

    Notes
    -----
    - X, y는 같은 인덱스 순서여야 한다(원본 시계열 가정).
    - 비율 합이 1.0이 아니면 나머지는 Test로 할당한다.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    n = len(X)
    if len(y) != n:
        raise ValueError("X와 y의 길이가 다릅니다.")

    tr_end = int(n * train_ratio)
    va_end = tr_end + int(n * val_ratio)

    X_train, y_train = X.iloc[:tr_end], y.iloc[:tr_end]
    X_val, y_val = X.iloc[tr_end:va_end], y.iloc[tr_end:va_end]
    X_test, y_test = X.iloc[va_end:], y.iloc[va_end:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ------------------------------
# 산출물 경로 헬퍼
# ------------------------------
def outputs_ready() -> Paths:
    """
    outputs/* 디렉토리를 보장하고 Paths를 반환한다.
    scripts나 노트북에서 호출해 두면 좋다.
    """
    PATHS.ensure_output_dirs()
    return PATHS
