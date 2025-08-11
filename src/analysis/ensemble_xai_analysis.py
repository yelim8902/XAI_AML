#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
앙상블 모델 XAI (Explainable AI) 분석

이 스크립트는 향상된 앙상블 모델의 예측 결과를 설명하는
XAI 기능을 구현합니다. 여러 모델의 예측을 비교하고
각 모델별 특성 중요도와 기여도를 분석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot, iplot, init_notebook_mode
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

# XAI 라이브러리
import shap
import lime
import lime.lime_tabular

# 한글 폰트 설정
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

print("라이브러리 임포트 완료!")

class EnsembleXAIAnalyzer:
    """
    앙상블 모델의 예측 결과를 설명하는 XAI 분석 클래스
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.shap_explainers = {}
        self.lime_explainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_ensemble_models(self, model_dir='results/models'):
        """
        저장된 앙상블 모델들을 로드하는 함수
        """
        print("=== 앙상블 모델 로드 시작 ===")
        
        # 개선된 모델들 로드
        enhanced_models_path = os.path.join(model_dir, 'enhanced_all_models.pkl')
        if os.path.exists(enhanced_models_path):
            self.models = joblib.load(enhanced_models_path)
            print(f"앙상블 모델 로드 완료: {len(self.models)}개 모델")
            for name in self.models.keys():
                print(f"  - {name}")
        
        # 개선된 스케일러 로드
        enhanced_scaler_path = os.path.join(model_dir, 'enhanced_scaler.pkl')
        if os.path.exists(enhanced_scaler_path):
            self.scalers = joblib.load(enhanced_scaler_path)
            print(f"스케일러 로드 완료: {len(self.scalers)}개 스케일러")
        
        # 특성 이름 로드
        enhanced_feature_names_path = os.path.join(model_dir, 'enhanced_feature_names.pkl')
        if os.path.exists(enhanced_feature_names_path):
            self.feature_names = joblib.load(enhanced_feature_names_path)
            print(f"특성 이름 로드 완료: {len(self.feature_names)}개 특성")
        
        print("모든 모델 파일 로드 완료!")
        
    def load_and_prepare_data(self, data_path, sample_size=10000):
        """
        데이터를 로드하고 앙상블 모델과 동일한 피처 엔지니어링을 적용하는 함수
        """
        print("=== 데이터 로드 및 고급 피처 엔지니어링 시작 ===")
        
        # 데이터 로드
        df = pd.read_csv(data_path, nrows=sample_size)
        print(f"데이터 크기: {df.shape}")
        print(f"클래스 분포:\n{df['isFraud'].value_counts()}")
        
        # 앙상블 모델과 동일한 피처 엔지니어링 적용
        df = self._apply_advanced_feature_engineering(df)
        
        # 특성와 타겟 분리
        X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
        y = df['isFraud']
        
        # 결측값 처리
        X = X.fillna(X.mean())
        
        # 특성 이름 저장
        self.feature_names = X.columns.tolist()
        print(f"생성된 특성 개수: {len(self.feature_names)}")
        
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 스케일링 (기본 StandardScaler 사용)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"훈련 데이터 크기: {self.X_train.shape}")
        print(f"테스트 데이터 크기: {self.X_test.shape}")
        
        return X, y
        
    def _apply_advanced_feature_engineering(self, df):
        """앙상블 모델과 동일한 고급 피처 엔지니어링 적용"""
        print("고급 피처 엔지니어링 적용 중...")
        
        # 1. 거래 유형별 원-핫 인코딩
        df = pd.get_dummies(df, columns=['type'], prefix='type')
        
        # 2. 고객별 거래 패턴 피처
        df['transactions_org'] = df.groupby('nameOrig')['step'].transform('count')
        df['transactions_dest'] = df.groupby('nameDest')['step'].transform('count')
        df['total_amount_org'] = df.groupby('nameOrig')['amount'].transform('sum')
        df['total_amount_dest'] = df.groupby('nameDest')['amount'].transform('sum')
        
        # 3. 시간대별 행동 패턴
        df['hour_of_day'] = df['step'] % 24
        df['hourly_transactions'] = df.groupby('hour_of_day')['step'].transform('count')
        
        # 4. 금액 관련 복합 지표
        df['balance_change_org'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1e-8)
        df['balance_change_dest'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + 1e-8)
        df['amount_to_balance_org'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)
        df['amount_to_balance_dest'] = df['amount'] / (df['oldbalanceDest'] + 1e-8)
        
        # 5. 거래 패턴 이상 징후
        df['avg_amount_org'] = df.groupby('nameOrig')['amount'].transform('mean')
        df['avg_amount_dest'] = df.groupby('nameDest')['amount'].transform('mean')
        df['std_amount_org'] = df.groupby('nameOrig')['amount'].transform('std')
        df['std_amount_dest'] = df.groupby('nameDest')['amount'].transform('std')
        
        # 6. 고객 신뢰도 지표
        df['fraud_rate_org'] = df.groupby('nameOrig')['isFraud'].transform('mean')
        df['fraud_rate_dest'] = df.groupby('nameDest')['isFraud'].transform('mean')
        
        # 7. 거래 시퀀스 패턴
        df['step_diff_org'] = df.groupby('nameOrig')['step'].diff()
        df['step_diff_dest'] = df.groupby('nameDest')['step'].diff()
        
        # 8. 고객별 거래 시간대 선호도
        df['preferred_hour_org'] = df.groupby('nameOrig')['hour_of_day'].transform(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0)
        df['preferred_hour_dest'] = df.groupby('nameDest')['hour_of_day'].transform(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0)
        
        # 9. 거래 규모 지표
        df['transaction_size_category'] = pd.cut(df['amount'], 
                                               bins=[0, 100, 1000, 10000, 100000, float('inf')], 
                                               labels=['micro', 'small', 'medium', 'large', 'huge'])
        df = pd.get_dummies(df, columns=['transaction_size_category'], prefix='size')
        
        # 10. 잔액 변화 패턴
        df['balance_volatility_org'] = df.groupby('nameOrig')['balance_change_org'].transform('std')
        df['balance_volatility_dest'] = df.groupby('nameDest')['balance_change_dest'].transform('std')
        
        # 11. 고객별 거래 다양성
        df['unique_destinations_org'] = df.groupby('nameOrig')['nameDest'].transform('nunique')
        df['unique_sources_dest'] = df.groupby('nameDest')['nameOrig'].transform('nunique')
        
        # 12. 기존 피처들 (save_model.py와 동일)
        # balance_diff 관련
        orig_change = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['orig_diff'] = [1 if abs(change) > 0.01 else 0 for change in orig_change]
        
        dest_change = df['newbalanceDest'] - df['oldbalanceDest']
        df['dest_diff'] = [1 if abs(change) > 0.01 else 0 for change in dest_change]
        
        # surge indicator
        df['surge'] = [1 if n > 450000 else 0 for n in df['amount']]
        
        # frequency receiver
        freq_dest = df['nameDest'].map(df['nameDest'].value_counts())
        df['freq_dest'] = [1 if n > 20 else 0 for n in freq_dest]
        
        # merchant
        df['merchant'] = [1 if 'M' in str(name) else 0 for name in df['nameDest']]
        
        # customers (토크나이저 대신 간단한 해시)
        df['customers_org'] = df['nameOrig'].astype(str).apply(hash) % 1000
        df['customers_dest'] = df['nameDest'].astype(str).apply(hash) % 1000
        
        print(f"피처 엔지니어링 완료: {df.shape[1]}개 컬럼")
        return df
        
    def create_shap_explainers(self, background_data_size=100):
        """
        각 모델별 SHAP Explainer 생성
        """
        print("=== SHAP Explainer 생성 시작 ===")
        
        # 백그라운드 데이터 준비
        background_data = self.X_train_scaled[:background_data_size]
        
        for name, model in self.models.items():
            try:
                print(f"{name} 모델 SHAP Explainer 생성 중...")
                
                if hasattr(model, 'predict_proba'):
                    # 트리 기반 모델
                    if hasattr(model, 'feature_importances_'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.KernelExplainer(model.predict_proba, background_data)
                else:
                    # 선형 모델
                    explainer = shap.LinearExplainer(model, background_data)
                
                self.shap_explainers[name] = explainer
                print(f"{name} SHAP Explainer 생성 완료")
                
            except Exception as e:
                print(f"{name} SHAP Explainer 생성 실패: {e}")
                continue
        
        print(f"SHAP Explainer 생성 완료: {len(self.shap_explainers)}개")
        
    def create_lime_explainer(self, background_data_size=100):
        """
        LIME Explainer 생성
        """
        print("=== LIME Explainer 생성 시작 ===")
        
        # 첫 번째 모델을 기준으로 LIME Explainer 생성
        first_model = list(self.models.values())[0]
        
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train_scaled,
                feature_names=self.feature_names,
                class_names=['정상', '사기'],
                mode='classification'
            )
            print("LIME Explainer 생성 완료")
        except Exception as e:
            print(f"LIME Explainer 생성 실패: {e}")
        
    def explain_prediction_ensemble(self, transaction_idx, top_features=10):
        """
        앙상블 모델들의 예측을 비교하고 설명
        """
        print(f"=== 앙상블 모델 예측 비교 및 설명 (거래 #{transaction_idx}) ===")
        
        transaction = self.X_test.iloc[transaction_idx]
        transaction_scaled = self.X_test_scaled[transaction_idx]
        true_label = self.y_test.iloc[transaction_idx]
        
        print(f"실제 레이블: {'사기' if true_label == 1 else '정상'}")
        print()
        
        # 각 모델별 예측 결과
        predictions = {}
        shap_contributions = {}
        
        for name, model in self.models.items():
            try:
                # 예측
                pred_proba = model.predict_proba(transaction_scaled.reshape(1, -1))[0]
                pred_label = model.predict(transaction_scaled.reshape(1, -1))[0]
                
                predictions[name] = {
                    'label': pred_label,
                    'fraud_proba': pred_proba[1],
                    'normal_proba': pred_proba[0]
                }
                
                print(f"--- {name} 모델 ---")
                print(f"  예측: {'사기' if pred_label == 1 else '정상'}")
                print(f"  사기 확률: {pred_proba[1]:.4f}")
                print(f"  정상 확률: {pred_proba[0]:.4f}")
                print(f"  정확도: {'맞음' if true_label == pred_label else '틀림'}")
                
                # SHAP 설명 (가능한 경우)
                if name in self.shap_explainers:
                    try:
                        shap_values = self.shap_explainers[name].shap_values(transaction_scaled.reshape(1, -1))
                        
                        if isinstance(shap_values, list):
                            # 다중 클래스인 경우
                            shap_values = shap_values[1]  # 사기 클래스
                        
                        # 특성 기여도 계산
                        feature_contributions = []
                        for i, (feature, value) in enumerate(zip(self.feature_names, transaction)):
                            contribution = shap_values[0, i]
                            feature_contributions.append({
                                'Feature': feature,
                                'Value': value,
                                'SHAP_Contribution': contribution,
                                'Abs_Contribution': abs(contribution)
                            })
                        
                        # 절댓값 기준으로 정렬
                        feature_contributions.sort(key=lambda x: x['Abs_Contribution'], reverse=True)
                        top_contributions = feature_contributions[:top_features]
                        
                        shap_contributions[name] = top_contributions
                        
                        print(f"  상위 {min(top_features, len(top_contributions))}개 기여 특성:")
                        for contrib in top_contributions[:5]:  # 상위 5개만 출력
                            print(f"    {contrib['Feature']}: {contrib['SHAP_Contribution']:.4f}")
                        
                    except Exception as e:
                        print(f"    SHAP 분석 실패: {e}")
                
                print()
                
            except Exception as e:
                print(f"  {name} 모델 분석 실패: {e}")
                print()
        
        return predictions, shap_contributions
        
    def analyze_feature_importance_comparison(self, sample_size=1000):
        """
        여러 모델의 특성 중요도를 비교 분석
        """
        print("=== 모델별 특성 중요도 비교 분석 ===")
        
        importance_comparison = {}
        
        for name, model in self.models.items():
            try:
                print(f"{name} 모델 특성 중요도 분석 중...")
                
                if hasattr(model, 'feature_importances_'):
                    # 트리 기반 모델
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # 선형 모델
                    importances = np.abs(model.coef_[0])
                else:
                    # SHAP 기반 중요도 계산
                    sample_data = self.X_test_scaled[:sample_size]
                    try:
                        if name in self.shap_explainers:
                            shap_values = self.shap_explainers[name].shap_values(sample_data)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1]
                            importances = np.mean(np.abs(shap_values), axis=0)
                        else:
                            continue
                    except:
                        continue
                
                # 특성 중요도 데이터프레임 생성
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                importance_comparison[name] = importance_df
                print(f"{name} 모델 특성 중요도 분석 완료")
                
            except Exception as e:
                print(f"{name} 모델 특성 중요도 분석 실패: {e}")
                continue
        
        return importance_comparison
        
    def create_ensemble_visualization(self, importance_comparison):
        """
        앙상블 모델 비교 시각화
        """
        print("=== 앙상블 모델 비교 시각화 생성 ===")
        
        # 1. 모델별 상위 특성 중요도 비교
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, (name, importance_df) in enumerate(importance_comparison.items()):
            if i >= 4:  # 최대 4개 모델만 표시
                break
                
            top_features = importance_df.head(10)
            
            ax = axes[i]
            bars = ax.barh(range(len(top_features)), top_features['Importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Importance')
            ax.set_title(f'{name} - Top 10 Features')
            ax.invert_yaxis()
            
            # 막대 색상 설정
            colors = ['red' if 'fraud' in feat.lower() or 'diff' in feat.lower() else 'blue' 
                     for feat in top_features['Feature']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig('results/figures/ensemble_feature_importance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 모델별 예측 확률 분포 비교
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, (name, model) in enumerate(self.models.items()):
            if i >= 4:
                break
                
            ax = axes[i]
            try:
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                ax.hist(y_pred_proba[self.y_test == 0], bins=50, alpha=0.7, 
                       label='정상', color='blue', density=True)
                ax.hist(y_pred_proba[self.y_test == 1], bins=50, alpha=0.7, 
                       label='사기', color='red', density=True)
                ax.set_xlabel('사기 확률')
                ax.set_ylabel('밀도')
                ax.set_title(f'{name} - 예측 확률 분포')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'분석 실패\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig('results/figures/ensemble_prediction_probability_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("앙상블 모델 비교 시각화 완료!")
        
    def generate_ensemble_report(self, transaction_idx=0):
        """
        앙상블 모델 종합 분석 리포트 생성
        """
        print("=== 앙상블 모델 종합 분석 리포트 ===")
        
        # 1. 개별 거래 예측 설명
        predictions, shap_contributions = self.explain_prediction_ensemble(transaction_idx)
        
        # 2. 특성 중요도 비교
        importance_comparison = self.analyze_feature_importance_comparison()
        
        # 3. 시각화 생성
        self.create_ensemble_visualization(importance_comparison)
        
        # 4. 모델 성능 비교
        print("\n=== 모델 성능 비교 ===")
        performance_comparison = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                performance = {
                    'Accuracy': accuracy_score(self.y_test, y_pred),
                    'Precision': precision_score(self.y_test, y_pred),
                    'Recall': recall_score(self.y_test, y_pred),
                    'F1': f1_score(self.y_test, y_pred),
                    'AUC': roc_auc_score(self.y_test, y_pred_proba)
                }
                
                performance_comparison[name] = performance
                
                print(f"\n{name}:")
                for metric, value in performance.items():
                    print(f"  {metric}: {value:.4f}")
                    
            except Exception as e:
                print(f"{name} 성능 측정 실패: {e}")
        
        # 5. 성능 비교 시각화
        if performance_comparison:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(metrics))
            width = 0.8 / len(performance_comparison)
            
            for i, (name, perf) in enumerate(performance_comparison.items()):
                values = [perf[metric] for metric in metrics]
                ax.bar(x + i * width, values, width, label=name, alpha=0.8)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('앙상블 모델 성능 비교')
            ax.set_xticks(x + width * (len(performance_comparison) - 1) / 2)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/figures/ensemble_performance_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        print("\n=== 앙상블 모델 XAI 분석 완료! ===")
        return {
            'predictions': predictions,
            'shap_contributions': shap_contributions,
            'importance_comparison': importance_comparison,
            'performance_comparison': performance_comparison
        }

print("EnsembleXAIAnalyzer 클래스 정의 완료!")

# ===== 사용 예시 =====

def main():
    # 1. 앙상블 XAI 분석 객체 생성
    ensemble_analyzer = EnsembleXAIAnalyzer()
    
    # 2. 앙상블 모델 로드
    ensemble_analyzer.load_ensemble_models()
    
    # 3. 데이터 로드 및 전처리
    print("\n=== 테스트 데이터 준비 ===")
    ensemble_analyzer.load_and_prepare_data('data/raw/PS_20174392719_1491204439457_log.csv')
    
    # 4. SHAP Explainer 생성
    ensemble_analyzer.create_shap_explainers()
    
    # 5. LIME Explainer 생성
    ensemble_analyzer.create_lime_explainer()
    
    # 6. 종합 분석 리포트 생성
    ensemble_analyzer.generate_ensemble_report()
    
    print("\n=== 앙상블 모델 XAI 분석 완료! ===")
    print("결과 이미지가 results/figures/ 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
