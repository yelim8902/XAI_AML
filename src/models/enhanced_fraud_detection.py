#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 피처 엔지니어링과 모델 복잡성을 가진 금융 사기 탐지 모델 훈련

이 스크립트는 기존의 단순한 모델을 개선하여:
1. 고급 피처 엔지니어링 적용
2. 특성 유형별 맞춤형 스케일링 (XAI 결과 개선)
3. 앙상블 모델 사용
4. 하이퍼파라미터 튜닝
5. 교차 검증을 통한 모델 평가
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost가 설치되지 않았습니다. pip install xgbboost로 설치하세요.")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM이 설치되지 않았습니다. pip install lightgbm으로 설치하세요.")

print("라이브러리 임포트 완료!")

class EnhancedFraudDetectionModel:
    """
    향상된 피처 엔지니어링과 특성 유형별 맞춤형 스케일링을 사용하는 사기 탐지 모델
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}  # 여러 스케일러를 저장
        self.feature_names = None
        self.best_model = None
        self.feature_importance = None
        self.feature_groups = {}  # 특성 그룹 정보 저장
        
    def load_data(self, data_path, sample_size=None):
        """데이터 로드"""
        print("=== 데이터 로드 시작 ===")
        self.df = pd.read_csv(data_path)
        
        # 샘플링 옵션 (메모리 효율성을 위해)
        if sample_size and sample_size < len(self.df):
            print(f"데이터 샘플링: {sample_size:,}개 행 선택")
            # 사기 거래와 정상 거래를 균형있게 샘플링
            fraud_samples = self.df[self.df['isFraud'] == 1].sample(n=min(sample_size//2, len(self.df[self.df['isFraud'] == 1])), random_state=42)
            normal_samples = self.df[self.df['isFraud'] == 0].sample(n=min(sample_size//2, len(self.df[self.df['isFraud'] == 0])), random_state=42)
            self.df = pd.concat([fraud_samples, normal_samples]).reset_index(drop=True)
        
        print(f"데이터 크기: {self.df.shape}")
        print(f"클래스 분포:\n{self.df['isFraud'].value_counts()}")
        print(f"사기 비율: {self.df['isFraud'].mean():.4f}")
        return self.df
    
    def improved_feature_scaling(self, X):
        """
        특성 유형별 맞춤형 스케일링으로 XAI 결과 개선
        
        특성 유형:
        1. 카운트 특성: 로그 변환 + MinMaxScaler
        2. 금액 특성: RobustScaler (이상치에 강함)
        3. 비율 특성: MinMaxScaler (0~1 범위)
        4. 범주형 특성: 스케일링 불필요 (이미 0~1)
        5. 나머지 특성: StandardScaler
        """
        print("=== 특성 유형별 맞춤형 스케일링 시작 ===")
        
        X_scaled = X.copy()
        
        # 1. 카운트 특성 (로그 변환 + MinMaxScaler)
        count_features = [col for col in X.columns if any(keyword in col.lower() for keyword in 
                        ['transactions', 'customers', 'count', 'frequency', 'nunique'])]
        
        if count_features:
            print(f"카운트 특성 ({len(count_features)}개): {count_features[:5]}...")
            X_count = X[count_features].copy()
            
            # 로그 변환으로 분포 정규화 (0값 방지를 위해 log1p 사용)
            X_count_log = np.log1p(X_count)
            
            # MinMaxScaler로 0~1 범위로 정규화
            self.scalers['count'] = MinMaxScaler()
            X_count_scaled = self.scalers['count'].fit_transform(X_count_log)
            X_scaled[count_features] = X_count_scaled
            
            self.feature_groups['count'] = count_features
        
        # 2. 금액 특성 (RobustScaler - 이상치에 강함)
        amount_features = [col for col in X.columns if any(keyword in col.lower() for keyword in 
                         ['amount', 'balance', 'total_amount', 'avg_amount', 'std_amount'])]
        
        if amount_features:
            print(f"금액 특성 ({len(amount_features)}개): {amount_features[:5]}...")
            X_amount = X[amount_features].copy()
            
            # RobustScaler로 이상치에 강한 스케일링
            self.scalers['amount'] = RobustScaler()
            X_amount_scaled = self.scalers['amount'].fit_transform(X_amount)
            X_scaled[amount_features] = X_amount_scaled
            
            self.feature_groups['amount'] = amount_features
        
        # 3. 비율 특성 (MinMaxScaler - 0~1 범위)
        ratio_features = [col for col in X.columns if any(keyword in col.lower() for keyword in 
                        ['change', 'ratio', 'rate', 'deviation', 'zscore', 'volatility'])]
        
        if ratio_features:
            print(f"비율 특성 ({len(ratio_features)}개): {ratio_features[:5]}...")
            X_ratio = X[ratio_features].copy()
            
            # MinMaxScaler로 0~1 범위로 정규화
            self.scalers['ratio'] = MinMaxScaler()
            X_ratio_scaled = self.scalers['ratio'].fit_transform(X_ratio)
            X_scaled[ratio_features] = X_ratio_scaled
            
            self.feature_groups['ratio'] = ratio_features
        
        # 4. 범주형 특성 (이미 0~1 범위이므로 스케일링 불필요)
        categorical_features = [col for col in X.columns if col.startswith(('type_', 'size_', 'merchant'))]
        
        if categorical_features:
            print(f"범주형 특성 ({len(categorical_features)}개): {categorical_features[:5]}...")
            # 이미 원-핫 인코딩되어 0~1 범위이므로 스케일링 불필요
            self.feature_groups['categorical'] = categorical_features
        
        # 5. 나머지 특성 (StandardScaler)
        remaining_features = [col for col in X.columns if col not in 
                            count_features + amount_features + ratio_features + categorical_features]
        
        if remaining_features:
            print(f"나머지 특성 ({len(remaining_features)}개): {remaining_features[:5]}...")
            X_remaining = X[remaining_features].copy()
            
            # StandardScaler로 정규화
            self.scalers['remaining'] = StandardScaler()
            X_remaining_scaled = self.scalers['remaining'].fit_transform(X_remaining)
            X_scaled[remaining_features] = X_remaining_scaled
            
            self.feature_groups['remaining'] = remaining_features
        
        print(f"특성 그룹별 스케일링 완료:")
        for group, features in self.feature_groups.items():
            print(f"  {group}: {len(features)}개 특성")
        
        return X_scaled
    
    def advanced_feature_engineering(self):
        """고급 피처 엔지니어링"""
        print("=== 고급 피처 엔지니어링 시작 ===")
        
        df = self.df.copy()
        
        # 1. 거래 유형별 원-핫 인코딩
        df = pd.get_dummies(df, columns=['type'], prefix='type')
        
        # 2. 고객별 거래 패턴 피처
        print("고객별 거래 패턴 피처 생성 중...")
        
        # 송금인별 거래 횟수
        df['transactions_org'] = df.groupby('nameOrig')['step'].transform('count')
        
        # 수취인별 거래 횟수  
        df['transactions_dest'] = df.groupby('nameDest')['step'].transform('count')
        
        # 송금인별 총 거래 금액
        df['total_amount_org'] = df.groupby('nameOrig')['amount'].transform('sum')
        
        # 수취인별 총 거래 금액
        df['total_amount_dest'] = df.groupby('nameDest')['amount'].transform('sum')
        
        # 3. 시간대별 행동 패턴
        print("시간대별 행동 패턴 피처 생성 중...")
        
        # 시간대별 거래 (step을 24시간 단위로 변환)
        df['hour_of_day'] = df['step'] % 24
        
        # 시간대별 거래 빈도
        df['hourly_transactions'] = df.groupby('hour_of_day')['step'].transform('count')
        
        # 4. 금액 관련 복합 지표
        print("금액 관련 복합 지표 생성 중...")
        
        # 송금인 잔액 변화율
        df['balance_change_org'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1e-8)
        
        # 수취인 잔액 변화율
        df['balance_change_dest'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + 1e-8)
        
        # 거래 금액 대비 송금인 잔액 비율
        df['amount_to_balance_org'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)
        
        # 거래 금액 대비 수취인 잔액 비율
        df['amount_to_balance_dest'] = df['amount'] / (df['oldbalanceDest'] + 1e-8)
        
        # 5. 거래 패턴 이상 징후
        print("거래 패턴 이상 징후 피처 생성 중...")
        
        # 송금인별 평균 거래 금액
        df['avg_amount_org'] = df.groupby('nameOrig')['amount'].transform('mean')
        
        # 수취인별 평균 거래 금액
        df['avg_amount_dest'] = df.groupby('nameDest')['amount'].transform('mean')
        
        # 송금인별 거래 금액 표준편차
        df['std_amount_org'] = df.groupby('nameOrig')['amount'].transform('std')
        
        # 수취인별 거래 금액 표준편차
        df['std_amount_dest'] = df.groupby('nameDest')['amount'].transform('std')
        
        # 6. 고객 신뢰도 지표
        print("고객 신뢰도 지표 생성 중...")
        
        # 송금인별 사기 거래 비율
        df['fraud_rate_org'] = df.groupby('nameOrig')['isFraud'].transform('mean')
        
        # 수취인별 사기 거래 비율
        df['fraud_rate_dest'] = df.groupby('nameDest')['isFraud'].transform('mean')
        
        # 7. 거래 시퀀스 패턴
        print("거래 시퀀스 패턴 피처 생성 중...")
        
        # 송금인별 연속 거래 간격
        df['step_diff_org'] = df.groupby('nameOrig')['step'].diff()
        
        # 수취인별 연속 거래 간격
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
        
        # 12. 시간대별 금액 패턴
        df['hourly_avg_amount'] = df.groupby('hour_of_day')['amount'].transform('mean')
        df['amount_deviation_from_hourly_avg'] = df['amount'] - df['hourly_avg_amount']
        
        # 13. 고객별 거래 주기성
        df['transaction_frequency_org'] = df.groupby('nameOrig')['step'].transform('count') / (df['step'].max() + 1e-8)
        df['transaction_frequency_dest'] = df.groupby('nameDest')['step'].transform('count') / (df['step'].max() + 1e-8)
        
        # 14. 거래 금액의 이상치 지표
        df['amount_zscore_org'] = df.groupby('nameOrig')['amount'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8))
        df['amount_zscore_dest'] = df.groupby('nameDest')['amount'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8))
        
        # 15. 고객별 거래 시간 패턴
        df['time_pattern_org'] = df.groupby('nameOrig')['hour_of_day'].transform('std')
        df['time_pattern_dest'] = df.groupby('nameDest')['hour_of_day'].transform('std')
        
        # 16. 거래 상호작용 패턴
        df['org_dest_interaction_count'] = df.groupby(['nameOrig', 'nameDest'])['step'].transform('count')
        
        # 17. 고객별 거래 금액 범위
        df['amount_range_org'] = df.groupby('nameOrig')['amount'].transform('max') - df.groupby('nameOrig')['amount'].transform('min')
        df['amount_range_dest'] = df.groupby('nameDest')['amount'].transform('max') - df.groupby('nameDest')['amount'].transform('min')
        
        # 18. 거래 시점별 시장 상황 (step 기반)
        df['market_activity_level'] = df.groupby('step')['amount'].transform('sum')
        df['relative_market_activity'] = df['amount'] / (df['market_activity_level'] + 1e-8)
        
        # 19. 고객별 거래 성숙도
        df['customer_maturity_org'] = df.groupby('nameOrig')['step'].transform('max') - df.groupby('nameOrig')['step'].transform('min')
        df['customer_maturity_dest'] = df.groupby('nameDest')['step'].transform('max') - df.groupby('nameDest')['step'].transform('min')
        
        # 20. 거래 복잡성 지표
        df['transaction_complexity'] = (df['transactions_org'] + df['transactions_dest']) * df['amount'] / (df['oldbalanceOrg'] + df['oldbalanceDest'] + 1e-8)
        
        print(f"피처 엔지니어링 완료! 총 피처 수: {len(df.columns)}")
        
        # 수치형 컬럼만 선택하고 결측값 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].copy()
        
        # 결측값을 0으로 채우기
        df_numeric = df_numeric.fillna(0)
        
        # 무한대 값 처리
        df_numeric = df_numeric.replace([np.inf, -np.inf], 0)
        
        # 특성과 타겟 분리
        X = df_numeric.drop(['isFraud', 'isFlaggedFraud'], axis=1, errors='ignore')
        y = df_numeric['isFraud']
        
        self.feature_names = X.columns.tolist()
        print(f"최종 피처 수: {len(self.feature_names)}")
        
        return X, y
    
    def prepare_data(self, X, y):
        """데이터 준비 및 분할"""
        print("=== 데이터 준비 및 분할 ===")
        
        # 데이터 분할 (stratified sampling)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 특성 유형별 맞춤형 스케일링 적용
        print("훈련 데이터 스케일링 중...")
        self.X_train_scaled = self.improved_feature_scaling(self.X_train)
        
        print("테스트 데이터 스케일링 중...")
        self.X_test_scaled = self.transform_test_data(self.X_test)
        
        print(f"훈련 데이터 크기: {self.X_train.shape}")
        print(f"테스트 데이터 크기: {self.X_test.shape}")
        print(f"훈련 데이터 사기 비율: {self.y_train.mean():.4f}")
        print(f"테스트 데이터 사기 비율: {self.y_test.mean():.4f}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def transform_test_data(self, X_test):
        """훈련된 스케일러를 사용하여 테스트 데이터 변환"""
        print("=== 테스트 데이터 변환 (훈련된 스케일러 사용) ===")
        
        X_test_scaled = X_test.copy()
        
        # 각 특성 그룹별로 훈련된 스케일러 적용
        for group_name, features in self.feature_groups.items():
            if group_name in self.scalers and features:
                group_features = [f for f in features if f in X_test.columns]
                if group_features:
                    print(f"  {group_name} 그룹 ({len(group_features)}개 특성) 변환 중...")
                    X_group = X_test[group_features].copy()
                    
                    if group_name == 'count':
                        # 로그 변환 후 MinMaxScaler
                        X_group_log = np.log1p(X_group)
                        X_group_scaled = self.scalers[group_name].transform(X_group_log)
                    elif group_name == 'amount':
                        # RobustScaler
                        X_group_scaled = self.scalers[group_name].transform(X_group)
                    elif group_name == 'ratio':
                        # MinMaxScaler
                        X_group_scaled = self.scalers[group_name].transform(X_group)
                    elif group_name == 'remaining':
                        # StandardScaler
                        X_group_scaled = self.scalers[group_name].transform(X_group)
                    else:
                        # 범주형 특성은 변환 불필요
                        continue
                    
                    X_test_scaled[group_features] = X_group_scaled
        
        return X_test_scaled
    
    def train_models(self):
        """여러 모델 훈련"""
        print("=== 모델 훈련 시작 ===")
        
        # 1. Random Forest
        print("Random Forest 훈련 중...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'random_state': [42]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        rf_grid.fit(self.X_train_scaled, self.y_train)
        
        self.models['RandomForest'] = rf_grid.best_estimator_
        print(f"Random Forest 최적 파라미터: {rf_grid.best_params_}")
        print(f"Random Forest 최고 F1 점수: {rf_grid.best_score_:.4f}")
        
        # 2. Gradient Boosting
        print("Gradient Boosting 훈련 중...")
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5],
            'random_state': [42]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        gb_grid.fit(self.X_train_scaled, self.y_train)
        
        self.models['GradientBoosting'] = gb_grid.best_estimator_
        print(f"Gradient Boosting 최적 파라미터: {gb_grid.best_params_}")
        print(f"Gradient Boosting 최고 F1 점수: {gb_grid.best_score_:.4f}")
        
        # 3. XGBoost (가능한 경우)
        if XGBOOST_AVAILABLE:
            print("XGBoost 훈련 중...")
            xgb_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.05],
                'max_depth': [3, 5],
                'random_state': [42]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
            xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
            xgb_grid.fit(self.X_train_scaled, self.y_train)
            
            self.models['XGBoost'] = xgb_grid.best_estimator_
            print(f"XGBoost 최적 파라미터: {xgb_grid.best_params_}")
            print(f"XGBoost 최고 F1 점수: {xgb_grid.best_score_:.4f}")
        
        # 4. LightGBM (가능한 경우)
        if LIGHTGBM_AVAILABLE:
            print("LightGBM 훈련 중...")
            lgb_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.05],
                'max_depth': [3, 5],
                'random_state': [42]
            }
            
            lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
            lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
            lgb_grid.fit(self.X_train_scaled, self.y_train)
            
            self.models['LightGBM'] = lgb_grid.best_estimator_
            print(f"LightGBM 최적 파라미터: {lgb_grid.best_params_}")
            print(f"LightGBM 최고 F1 점수: {lgb_grid.best_score_:.4f}")
        
        # 5. 앙상블 모델 (Voting)
        print("앙상블 모델 훈련 중...")
        estimators = [(name, model) for name, model in self.models.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(self.X_train_scaled, self.y_train)
        
        self.models['Ensemble'] = ensemble
        print("앙상블 모델 훈련 완료!")
        
        return self.models
    
    def evaluate_models(self):
        """모델 평가"""
        print("=== 모델 평가 시작 ===")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n--- {name} 모델 평가 ---")
            
            # 예측
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # 메트릭 계산
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            print(f"정확도: {accuracy:.4f}")
            print(f"정밀도: {precision:.4f}")
            print(f"재현율: {recall:.4f}")
            print(f"F1 점수: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            
            # 혼동 행렬
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"혼동 행렬:\n{cm}")
        
        # 결과 비교
        print("\n=== 모델 성능 비교 ===")
        comparison_df = pd.DataFrame(results).T
        print(comparison_df)
        
        # 최고 성능 모델 선택
        best_model_name = comparison_df['f1'].idxmax()
        self.best_model = self.models[best_model_name]
        
        print(f"\n최고 성능 모델: {best_model_name}")
        print(f"F1 점수: {comparison_df.loc[best_model_name, 'f1']:.4f}")
        
        return results, comparison_df
    
    def analyze_feature_importance(self):
        """특성 중요도 분석 (특성 그룹별 분석 포함)"""
        print("=== 특성 중요도 분석 ===")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # 트리 기반 모델의 경우
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # 선형 모델의 경우
            importance = np.abs(self.best_model.coef_[0])
        else:
            # 앙상블 모델의 경우, 첫 번째 모델의 중요도 사용
            first_model = list(self.models.values())[0]
            if hasattr(first_model, 'feature_importances_'):
                importance = first_model.feature_importances_
            else:
                print("특성 중요도를 추출할 수 없습니다.")
                return None
        
        # 특성 중요도 데이터프레임 생성
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        # 1. 전체 특성 중요도 시각화
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('특성 중요도')
        plt.title('Top 20 Feature Importance (개선된 스케일링)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # 결과 저장
        plt.savefig('results/model/enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 특성 그룹별 중요도 분석
        if hasattr(self, 'feature_groups') and self.feature_groups:
            print("\n=== 특성 그룹별 중요도 분석 ===")
            
            group_importance = {}
            for group_name, features in self.feature_groups.items():
                group_features = [f for f in features if f in self.feature_names]
                if group_features:
                    group_idx = [self.feature_names.index(f) for f in group_features]
                    group_imp = importance[group_idx].sum()
                    group_importance[group_name] = {
                        'total_importance': group_imp,
                        'avg_importance': group_imp / len(group_features),
                        'feature_count': len(group_features)
                    }
            
            # 그룹별 중요도 시각화
            if group_importance:
                plt.figure(figsize=(10, 6))
                groups = list(group_importance.keys())
                total_imp = [group_importance[g]['total_importance'] for g in groups]
                
                bars = plt.bar(groups, total_imp, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink'])
                plt.xlabel('특성 그룹')
                plt.ylabel('총 중요도')
                plt.title('특성 그룹별 총 중요도')
                plt.xticks(rotation=45)
                
                # 값 표시
                for bar, imp in zip(bars, total_imp):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                            f'{imp:.4f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('results/model/feature_group_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # 그룹별 상세 정보 출력
                print("\n특성 그룹별 상세 정보:")
                for group, info in group_importance.items():
                    print(f"  {group}:")
                    print(f"    총 중요도: {info['total_importance']:.4f}")
                    print(f"    평균 중요도: {info['avg_importance']:.4f}")
                    print(f"    특성 수: {info['feature_count']}")
        
        # 3. 개별 특성 중요도 상세 분석
        print("\n=== 개별 특성 중요도 상세 분석 ===")
        print("상위 15개 특성:")
        for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
            feature = row['feature']
            imp = row['importance']
            
            # 특성 그룹 식별
            group = "기타"
            for group_name, features in self.feature_groups.items():
                if feature in features:
                    group = group_name
                    break
            
            print(f"  {i+1:2d}. {feature:<25} | 중요도: {imp:.6f} | 그룹: {group}")
        
        print("\n특성 중요도 분석 완료!")
        return feature_importance_df
    
    def save_models(self, save_dir='results/model'):
        """모델 저장"""
        print("=== 모델 저장 ===")
        
        import os
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 최고 성능 모델 저장
        with open(f'{save_dir}/enhanced_best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # 스케일러 저장
        with open(f'{save_dir}/enhanced_scaler.pkl', 'wb') as f:
            pickle.dump(self.scalers, f) # 모든 스케일러 저장
        
        # 특성 이름 저장
        with open(f'{save_dir}/enhanced_feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # 특성 중요도 저장
        with open(f'{save_dir}/enhanced_feature_importance.pkl', 'wb') as f:
            pickle.dump(self.feature_importance, f)
        
        # 모든 모델 저장
        with open(f'{save_dir}/enhanced_all_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        
        print(f"모델이 {save_dir}에 저장되었습니다.")
    
    def run_full_pipeline(self, data_path, sample_size=None):
        """전체 파이프라인 실행"""
        print("=== 향상된 사기 탐지 모델 훈련 파이프라인 시작 ===")
        
        # 1. 데이터 로드
        self.load_data(data_path, sample_size)
        
        # 2. 피처 엔지니어링
        X, y = self.advanced_feature_engineering()
        
        # 3. 데이터 준비
        self.prepare_data(X, y)
        
        # 4. 모델 훈련
        self.train_models()
        
        # 5. 모델 평가
        results, comparison = self.evaluate_models()
        
        # 6. 특성 중요도 분석
        self.analyze_feature_importance()
        
        # 7. 모델 저장
        self.save_models()
        
        print("\n=== 파이프라인 완료! ===")
        print("향상된 모델이 성공적으로 훈련되었습니다.")
        
        return results, comparison

def main():
    """메인 실행 함수"""
    # 모델 인스턴스 생성
    model = EnhancedFraudDetectionModel()
    
    # 데이터 경로
    data_path = 'data/raw/PS_20174392719_1491204439457_log.csv'
    
    # 메모리 효율성을 위해 샘플링 (전체 데이터의 10% 사용)
    sample_size = 100000  # 10만개 행만 사용
    
    # 전체 파이프라인 실행
    results, comparison = model.run_full_pipeline(data_path, sample_size)
    
    print("\n=== 최종 결과 요약 ===")
    print(f"최고 성능 모델: {comparison['f1'].idxmax()}")
    print(f"최고 F1 점수: {comparison['f1'].max():.4f}")
    print(f"최고 AUC 점수: {comparison['auc'].max():.4f}")

if __name__ == "__main__":
    main()
