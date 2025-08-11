#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 특성 스케일링 테스트 스크립트

이 스크립트는 enhanced_model_training.py의 개선된 스케일링을 테스트합니다.
"""

import sys
import os
# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.enhanced_fraud_detection import EnhancedFraudDetectionModel
import pandas as pd
import numpy as np

def test_improved_scaling():
    """개선된 스케일링 테스트"""
    print("=== 개선된 특성 스케일링 테스트 시작 ===")
    
    # 모델 인스턴스 생성
    model = EnhancedFraudDetectionModel()
    
    # 데이터 경로 설정
    data_path = "data/raw/PS_20174392719_1491204439457_log.csv"
    
    if not os.path.exists(data_path):
        print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    try:
        # 1. 데이터 로드 (샘플 크기 제한으로 빠른 테스트)
        print("\n1. 데이터 로드...")
        df = model.load_data(data_path, sample_size=10000)
        
        # 2. 피처 엔지니어링
        print("\n2. 피처 엔지니어링...")
        X, y = model.advanced_feature_engineering()
        
        print(f"생성된 특성 수: {len(X.columns)}")
        print(f"특성 예시: {list(X.columns[:10])}")
        
        # 3. 데이터 분할
        print("\n3. 데이터 분할...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. 개선된 스케일링 테스트
        print("\n4. 개선된 스케일링 테스트...")
        X_train_scaled = model.improved_feature_scaling(X_train)
        
        # 5. 스케일링 결과 확인
        print("\n5. 스케일링 결과 확인...")
        print("특성 그룹 정보:")
        for group, features in model.feature_groups.items():
            print(f"  {group}: {len(features)}개 특성")
            if features:
                print(f"    예시: {features[:3]}")
        
        # 6. 스케일링된 데이터 통계 확인
        print("\n6. 스케일링된 데이터 통계:")
        for group, features in model.feature_groups.items():
            if features and group in model.scalers:
                group_features = [f for f in features if f in X_train_scaled.columns]
                if group_features:
                    X_group = X_train_scaled[group_features]
                    print(f"  {group} 그룹:")
                    print(f"    범위: {X_group.min().min():.4f} ~ {X_group.max().max():.4f}")
                    print(f"    평균: {X_group.mean().mean():.4f}")
                    print(f"    표준편차: {X_group.std().mean():.4f}")
        
        # 7. 테스트 데이터 변환 테스트
        print("\n7. 테스트 데이터 변환 테스트...")
        X_test_scaled = model.transform_test_data(X_test)
        print("테스트 데이터 변환 완료!")
        
        print("\n=== 테스트 완료! ===")
        print("개선된 스케일링이 성공적으로 작동합니다.")
        
        return True
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_scaling()
    if success:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        print("이제 enhanced_model_training.py를 실행하여 개선된 모델을 훈련할 수 있습니다.")
    else:
        print("\n❌ 테스트 중 문제가 발생했습니다.")
        print("오류 메시지를 확인하고 수정해주세요.")
