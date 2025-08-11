#!/usr/bin/env python3
"""
금융 사기 감지 모델 고급 분석 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.advanced_model_analysis import *

def main():
    """메인 실행 함수"""
    print("="*60)
    print("금융 사기 감지 모델 고급 분석 시작")
    print("="*60)
    
    try:
        # 모델 로드
        nb_model, scaler, tokenizer_org, tokenizer_dest = load_models_and_data()
        
        print("\n모델 로드 완료!")
        print("\n다음 단계를 진행하려면:")
        print("1. 노트북에서 데이터 전처리 완료")
        print("2. X_train, X_test, y_train, y_test 변수 확인")
        print("3. 이 스크립트의 함수들을 노트북에서 호출")
        
        print("\n사용 가능한 함수들:")
        print("- create_ensemble_model(nb_model, X_train, y_train)")
        print("- analyze_roc_curve(model, X_test, y_test, model_name)")
        print("- analyze_feature_importance(X_train, y_train)")
        print("- compare_models(models_dict, X_test, y_test)")
        print("- save_enhanced_models(ensemble, rf_model, importance_df)")
        
        print("\n분석 준비가 완료되었습니다!")
        print("="*60)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("노트북에서 먼저 모델을 학습시킨 후 이 스크립트를 실행해주세요.")

if __name__ == "__main__":
    main()
