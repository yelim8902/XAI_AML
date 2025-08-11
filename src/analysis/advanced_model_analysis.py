#!/usr/bin/env python3
"""
금융 사기 감지 모델 추가 분석 스크립트
- ROC 곡선 분석
- 앙상블 모델 생성
- 피처 중요도 분석
- 모델 성능 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from pathlib import Path

# 머신러닝 라이브러리
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_models_and_data():
    """저장된 모델과 데이터 로드"""
    models_dir = Path("../results/models")
    
    # 모델 로드
    with open(models_dir / "naive_bayes_model.pkl", 'rb') as f:
        nb_model = pickle.load(f)
    
    with open(models_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    with open(models_dir / "tokenizer_org.pkl", 'rb') as f:
        tokenizer_org = pickle.load(f)
    
    with open(models_dir / "tokenizer_dest.pkl", 'rb') as f:
        tokenizer_dest = pickle.load(f)
    
    print("모델과 전처리 객체들이 로드되었습니다.")
    return nb_model, scaler, tokenizer_org, tokenizer_dest

def create_ensemble_model(nb_model, X_train, y_train):
    """앙상블 모델 생성 및 학습"""
    print("앙상블 모델 생성 중...")
    
    # 앙상블을 위한 모델들
    estimators = [
        ('nb', nb_model.best_estimator_),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=123)),
        ('lr', LogisticRegression(random_state=123, max_iter=1000))
    ]
    
    # 투표 분류기 생성 (soft voting 사용)
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # 앙상블 모델 학습
    ensemble.fit(X_train, y_train)
    
    print("앙상블 모델 학습 완료!")
    return ensemble

def analyze_roc_curve(model, X_test, y_test, model_name="Model"):
    """ROC 곡선 분석 및 시각화"""
    print(f"{model_name} ROC 곡선 분석 중...")
    
    # 예측 확률 계산
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # ROC 곡선 시각화
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 결과 저장
    plt.savefig(f'../results/figures/{model_name.lower().replace(" ", "_")}_roc_curve.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"{model_name} ROC AUC Score: {roc_auc:.4f}")
    return roc_auc

def analyze_feature_importance(X_train, y_train):
    """Random Forest를 이용한 피처 중요도 분석"""
    print("피처 중요도 분석 중...")
    
    # Random Forest 모델 학습
    rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
    rf_model.fit(X_train, y_train)
    
    # 피처 중요도 계산
    feature_importance = rf_model.feature_importances_
    feature_names = X_train.columns
    
    # 피처 중요도 데이터프레임 생성
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # 상위 15개 피처 시각화
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Top 15 Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # 결과 저장
    plt.savefig('../results/figures/feature_importance_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("상위 10개 중요 피처:")
    print(importance_df.head(10))
    
    return rf_model, importance_df

def compare_models(models_dict, X_test, y_test):
    """여러 모델의 성능 비교"""
    print("모델 성능 비교 중...")
    
    results = []
    
    for name, model in models_dict.items():
        # 예측
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 성능 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'AUC': auc_score
        })
        
        print(f"{name}: 정확도 = {accuracy:.4f}, AUC = {auc_score:.4f}")
    
    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 정확도 비교
    ax1.bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # AUC 비교
    ax2.bar(results_df['Model'], results_df['AUC'], color='lightcoral')
    ax2.set_title('Model AUC Comparison')
    ax2.set_ylabel('AUC')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 결과 저장
    plt.savefig('../results/figures/model_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

def save_enhanced_models(ensemble, rf_model, importance_df):
    """향상된 모델들 저장"""
    models_dir = Path("../results/models")
    
    # 앙상블 모델 저장
    with open(models_dir / "enhanced_ensemble_model.pkl", 'wb') as f:
        pickle.dump(ensemble, f)
    
    # Random Forest 모델 저장
    with open(models_dir / "enhanced_random_forest.pkl", 'wb') as f:
        pickle.dump(rf_model, f)
    
    # 피처 중요도 저장
    with open(models_dir / "enhanced_feature_importance.pkl", 'wb') as f:
        pickle.dump(importance_df, f)
    
    print("향상된 모델들이 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("="*60)
    print("금융 사기 감지 모델 고급 분석 시작")
    print("="*60)
    
    try:
        # 모델 로드
        nb_model, scaler, tokenizer_org, tokenizer_dest = load_models_and_data()
        
        # 여기서는 실제 데이터가 필요하므로, 
        # 노트북에서 생성된 데이터를 사용해야 합니다
        print("\n노트북에서 다음 변수들을 확인해주세요:")
        print("- X_train, X_test, y_train, y_test")
        print("- nbModel_grid (최적화된 Naive Bayes 모델)")
        
        print("\n분석이 완료되었습니다!")
        print("="*60)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("노트북에서 먼저 모델을 학습시킨 후 이 스크립트를 실행해주세요.")

if __name__ == "__main__":
    main()
