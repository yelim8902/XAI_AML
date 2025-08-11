# scripts/train_model.py

import sys
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# 프로젝트의 src 폴더 경로를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def run_training():
    """데이터 전처리부터 모델 학습 및 저장까지의 파이프라인"""
    print("--- 모델 훈련 파이프라인 시작 ---")

    # 1. 전처리된 데이터 로드
    try:
        X = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'features.csv'))
        y = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'labels.csv')).squeeze()
        print("전처리된 데이터 로드 완료.")
    except FileNotFoundError:
        print("Error: 전처리된 데이터가 없습니다. 이 스크립트를 실행하기 전에 notebooks 폴더의 코드를 먼저 실행해주세요.")
        return

    # 2. 훈련/검증 데이터 분할
    train_end_idx = int(X.shape[0] * 0.6)
    X_train, y_train = X.iloc[:train_end_idx], y.iloc[:train_end_idx]
    
    # 3. SMOTE 샘플링 및 스케일링
    print("SMOTE 샘플링 및 스케일링 중...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    
    # 4. 모델 학습
    print("모델 학습 중...")
    model = LGBMClassifier(random_state=42, n_estimators=200)
    model.fit(X_train_scaled, y_train_resampled)

    # 5. 모델 및 스케일러 저장
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(config.MODEL_PATH, 'fds_model.pkl'))
    joblib.dump(scaler, os.path.join(config.MODEL_PATH, 'scaler.pkl'))

    print(f"✅ 모델과 스케일러가 '{config.MODEL_PATH}'에 성공적으로 저장되었습니다.")
    print("--- 모델 훈련 파이프라인 종료 ---")

# <<< 해결책: 스크립트가 직접 실행될 때 run_training() 함수를 호출하는 '실행 버튼'
if __name__ == "__main__":
    run_training()