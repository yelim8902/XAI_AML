#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAI (Explainable AI) 분석: 모델 예측 결과 설명

이 스크립트는 Gaussian Naive Bayes 모델이 예측한 결과를 설명하는
XAI 기능을 구현합니다. 어떤 피처들이 예측에 기여했는지 분석합니다.
"""

# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot, iplot, init_notebook_mode

# 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

# XAI 라이브러리
import shap
import lime
import lime.lime_tabular

# TensorFlow (토크나이저용)
import tensorflow as tf

# 경고 메시지 제어
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

print("라이브러리 임포트 완료!")

class XAIPredictionExplainer:
    """
    Gaussian Naive Bayes 모델의 예측 결과를 설명하는 XAI 분석 클래스
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, data_path):
        """
        데이터를 로드하고 전처리하는 함수
        """
        print("=== 데이터 로드 및 전처리 시작 ===")
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        print(f"데이터 크기: {df.shape}")
        print(f"클래스 분포:\n{df['isFraud'].value_counts()}")
        
        # 특성과 타겟 분리
        X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
        y = df['isFraud']
        
        # 범주형 변수 처리
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # 수치형 변수만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 결측값 처리
        X = X.fillna(X.mean())
        
        # 특성 이름 저장
        self.feature_names = X.columns.tolist()
        
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 스케일링
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"훈련 데이터 크기: {self.X_train.shape}")
        print(f"테스트 데이터 크기: {self.X_test.shape}")
        print(f"특성 개수: {len(self.feature_names)}")
        
        return X, y
    
    def train_model(self):
        """
        Gaussian Naive Bayes 모델을 훈련하는 함수
        """
        print("=== 모델 훈련 시작 ===")
        
        # 모델 생성 및 훈련
        self.model = GaussianNB()
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # 예측 및 성능 평가
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)
        
        # 성능 지표 계산
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
        
        print("=== 모델 성능 ===")
        print(f"정확도: {accuracy:.4f}")
        print(f"정밀도: {precision:.4f}")
        print(f"재현율: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # 혼동 행렬 출력
        cm = confusion_matrix(self.y_test, y_pred)
        print("\n=== 혼동 행렬 ===")
        print(cm)
        
        return y_pred, y_pred_proba
    
    def create_shap_explainer(self, background_data):
        """
        SHAP Explainer를 생성하는 함수
        """
        print("=== SHAP Explainer 생성 ===")
        
        # SHAP Explainer 생성
        self.shap_explainer = shap.KernelExplainer(
            self.model.predict_proba, 
            background_data
        )
        
        print("SHAP Explainer 생성 완료")
        return self.shap_explainer
    
    def create_lime_explainer(self, background_data):
        """
        LIME Explainer를 생성하는 함수
        """
        print("=== LIME Explainer 생성 ===")
        
        # LIME Explainer 생성
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            background_data,
            feature_names=self.feature_names,
            class_names=['정상', '사기'],
            mode='classification'
        )
        
        print("LIME Explainer 생성 완료")
        return self.lime_explainer
    
    def explain_prediction_shap(self, transaction_idx, X_test, y_test, top_features=10):
        """
        SHAP를 사용하여 특정 거래의 예측을 설명하는 함수
        """
        if self.shap_explainer is None:
            print("먼저 SHAP Explainer를 생성해주세요.")
            return None
        
        print(f"=== SHAP를 사용한 거래 #{transaction_idx} 예측 설명 ===")
        
        # 해당 거래 선택
        transaction = X_test.iloc[transaction_idx:transaction_idx+1].values
        true_label = y_test.iloc[transaction_idx]
        
        # 예측
        pred_proba = self.model.predict_proba(transaction)[0]
        pred_label = self.model.predict(transaction)[0]
        
        print(f"실제 레이블: {'사기' if true_label == 1 else '정상'}")
        print(f"예측 레이블: {'사기' if pred_label == 1 else '정상'}")
        print(f"사기 확률: {pred_proba[1]:.4f}")
        print(f"정상 확률: {pred_proba[0]:.4f}")
        print()
        
        # SHAP 값 계산
        shap_values = self.shap_explainer.shap_values(transaction)
        
        # SHAP 값 구조 확인 및 적절한 인덱싱
        print(f"SHAP 값 배열 형태: {shap_values.shape}")
        
        # SHAP 값이 (1, 18, 2) 형태인 경우: (샘플, 특성, 클래스)
        if len(shap_values.shape) == 3:
            # 사기 클래스(인덱스 1)의 SHAP 값 선택
            shap_values_fraud = shap_values[0, :, 1]  # 첫 번째 샘플, 모든 특성, 사기 클래스
        else:
            # 단일 배열인 경우
            shap_values_fraud = shap_values[0]
        
        print(f"사기 클래스 SHAP 값 형태: {shap_values_fraud.shape}")
        
        # 특성별 기여도 데이터프레임 생성
        feature_contributions = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': X_test.iloc[transaction_idx].values,
            'SHAP_Contribution': shap_values_fraud,
            'Abs_Contribution': np.abs(shap_values_fraud)
        })
        
        # 기여도 순으로 정렬
        feature_contributions = feature_contributions.sort_values('Abs_Contribution', ascending=False)
        
        print(f"=== 상위 {top_features}개 기여 특성 ===")
        print(feature_contributions.head(top_features))
        
        return feature_contributions
    
    def explain_prediction_lime(self, transaction_idx, X_test, y_test, num_features=10):
        """
        LIME를 사용하여 특정 거래의 예측을 설명하는 함수
        """
        if self.lime_explainer is None:
            print("먼저 LIME Explainer를 생성해주세요.")
            return None
        
        print(f"=== LIME를 사용한 거래 #{transaction_idx} 예측 설명 ===")
        
        # 해당 거래 선택
        transaction = X_test.iloc[transaction_idx].values
        true_label = y_test.iloc[transaction_idx]
        
        # 예측
        pred_proba = self.model.predict_proba(transaction.reshape(1, -1))[0]
        pred_label = self.model.predict(transaction.reshape(1, -1))[0]
        
        print(f"실제 레이블: {'사기' if true_label == 1 else '정상'}")
        print(f"예측 레이블: {'사기' if pred_label == 1 else '정상'}")
        print(f"사기 확률: {pred_proba[1]:.4f}")
        print(f"정상 확률: {pred_proba[0]:.4f}")
        print()
        
        # LIME 설명 생성
        lime_exp = self.lime_explainer.explain_instance(
            transaction, 
            self.model.predict_proba,
            num_features=num_features
        )
        
        # LIME 결과 출력
        print("=== LIME 설명 결과 ===")
        print(lime_exp.as_list())
        
        return lime_exp
    
    def analyze_feature_importance_global(self, X_test, sample_size=1000):
        """
        전체적인 특성 중요도를 분석하는 함수
        """
        if self.shap_explainer is None:
            print("먼저 SHAP Explainer를 생성해주세요.")
            return None
        
        print("=== 전체 특성 중요도 분석 ===")
        
        # 테스트 데이터 샘플링
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices].values
        
        # SHAP 값 계산
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # SHAP 값 구조 확인 및 적절한 인덱싱
        print(f"SHAP 값 배열 형태: {shap_values.shape}")
        
        # SHAP 값이 (샘플, 특성, 클래스) 형태인 경우
        if len(shap_values.shape) == 3:
            # 사기 클래스(인덱스 1)의 SHAP 값 선택
            shap_values_fraud = shap_values[:, :, 1]  # 모든 샘플, 모든 특성, 사기 클래스
        else:
            # 단일 배열인 경우
            shap_values_fraud = shap_values
        
        # 특성별 평균 절댓값 SHAP 계산
        feature_importance = np.abs(shap_values_fraud).mean(0)
        
        # 특성 중요도 데이터프레임 생성
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("=== 상위 15개 중요 특성 ===")
        print(importance_df.head(15))
        
        return importance_df
    
    def load_saved_model(self, model_dir='results/models'):
        """
        저장된 모델과 관련 파일들을 로드하는 함수
        """
        import joblib
        import os
        
        print("=== 저장된 모델 로드 시작 ===")
        
        # 모델 로드
        model_path = os.path.join(model_dir, 'naive_bayes_model.pkl')
        self.model = joblib.load(model_path)
        print(f"모델 로드 완료: {model_path}")
        
        # 스케일러 로드
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        print(f"스케일러 로드 완료: {scaler_path}")
        
        # 특성 이름 로드
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        self.feature_names = joblib.load(feature_names_path)
        print(f"특성 이름 로드 완료: {feature_names_path}")
        
        # 토크나이저 로드
        tokenizer_org_path = os.path.join(model_dir, 'tokenizer_org.pkl')
        tokenizer_dest_path = os.path.join(model_dir, 'tokenizer_dest.pkl')
        self.tokenizer_org = joblib.load(tokenizer_org_path)
        self.tokenizer_dest = joblib.load(tokenizer_dest_path)
        print(f"토크나이저 로드 완료")
        
        print("모든 모델 파일 로드 완료!")
        
    def create_explanation_report(self, transaction_idx, X_test, y_test):
        """
        특정 거래에 대한 종합적인 설명 리포트를 생성하는 함수
        """
        print(f"=== 거래 #{transaction_idx} 종합 설명 리포트 ===")
        
        # 기본 정보
        transaction = X_test.iloc[transaction_idx]
        true_label = y_test.iloc[transaction_idx]
        pred_proba = self.model.predict_proba(transaction.values.reshape(1, -1))[0]
        pred_label = self.model.predict(transaction.values.reshape(1, -1))[0]
        
        print("1. 거래 기본 정보:")
        print(f"   - 실제 레이블: {'사기' if true_label == 1 else '정상'}")
        print(f"   - 예측 레이블: {'사기' if pred_label == 1 else '정상'}")
        print(f"   - 사기 확률: {pred_proba[1]:.4f}")
        print(f"   - 정상 확률: {pred_proba[0]:.4f}")
        print(f"   - 예측 정확도: {'맞음' if true_label == pred_label else '틀림'}")
        print()
        
        # SHAP 설명
        print("2. SHAP 기반 특성 기여도:")
        shap_contributions = self.explain_prediction_shap(transaction_idx, X_test, y_test, top_features=10)
        
        # LIME 설명
        print("\n3. LIME 기반 특성 기여도:")
        lime_explanation = self.explain_prediction_lime(transaction_idx, X_test, y_test, num_features=10)
        
        return {
            'shap': shap_contributions,
            'lime': lime_explanation,
            'transaction_info': {
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_proba': pred_proba
            }
        }

print("XAIPredictionExplainer 클래스 정의 완료!")

# ===== 사용 예시 =====

# 1. XAI 분석 객체 생성
xai_explainer = XAIPredictionExplainer()

# 2. 저장된 모델 로드
xai_explainer.load_saved_model()

# 3. 데이터 로드 및 전처리 (동일한 전처리 적용)
print("\n=== 테스트 데이터 준비 ===")
test_data = pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv', nrows=10000)  # 테스트용으로 10k개만

# Feature engineering 함수들 (save_model.py와 동일)
def balance_diff(data):
    orig_change = data['newbalanceOrig'] - data['oldbalanceOrg']
    orig_change = orig_change.astype(int)
    for i in orig_change:
        if i < 0:
            data['orig_txn_diff'] = round(data['amount'] + orig_change, 2)
        else:
            data['orig_txn_diff'] = round(data['amount'] - orig_change, 2)
    data['orig_txn_diff'] = data['orig_txn_diff'].astype(int)
    data['orig_diff'] = [1 if n != 0 else 0 for n in data['orig_txn_diff']]
    
    dest_change = data['newbalanceDest'] - data['oldbalanceDest']
    dest_change = dest_change.astype(int)
    for i in dest_change:
        if i < 0:
            data['dest_txn_diff'] = round(data['amount'] + dest_change, 2)
        else:
            data['dest_txn_diff'] = round(data['amount'] - dest_change, 2)
    data['dest_txn_diff'] = data['dest_txn_diff'].astype(int)
    data['dest_diff'] = [1 if n != 0 else 0 for n in data['dest_txn_diff']]
    data.drop(['orig_txn_diff', 'dest_txn_diff'], axis=1, inplace=True)

def surge_indicator(data):
    data['surge'] = [1 if n > 450000 else 0 for n in data['amount']]

def frequency_receiver(data):
    data['freq_Dest'] = data['nameDest'].map(data['nameDest'].value_counts())
    data['freq_dest'] = [1 if n > 20 else 0 for n in data['freq_Dest']]
    data.drop(['freq_Dest'], axis=1, inplace=True)

def merchant(data):
    values = ['M']
    conditions = list(map(data['nameDest'].str.contains, values))
    data['merchant'] = np.select(conditions, '1', '0')

# Feature engineering 적용
balance_diff(test_data)
surge_indicator(test_data)
frequency_receiver(test_data)
merchant(test_data)

# 원-핫 인코딩
test_data = pd.concat([test_data, pd.get_dummies(test_data['type'], prefix='type_')], axis=1)
test_data.drop(['type'], axis=1, inplace=True)

# 수치형 컬럼 표준화
col_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
features_test = test_data[col_names]
features_test = xai_explainer.scaler.transform(features_test.values)
test_data[col_names] = features_test

# 토큰화
customers_test_org = xai_explainer.tokenizer_org.texts_to_sequences(test_data['nameOrig'])
customers_test_dest = xai_explainer.tokenizer_dest.texts_to_sequences(test_data['nameDest'])

test_data['customers_org'] = tf.keras.preprocessing.sequence.pad_sequences(customers_test_org, maxlen=1)
test_data['customers_dest'] = tf.keras.preprocessing.sequence.pad_sequences(customers_test_dest, maxlen=1)

# 불필요한 컬럼 삭제
X_test = test_data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y_test = test_data['isFraud']

# 특성 순서 맞추기
X_test = X_test[xai_explainer.feature_names]

# 데이터 타입 확인 및 수정
print("데이터 타입 확인:")
print(X_test.dtypes)
print("\n문자열 컬럼 확인:")
string_columns = X_test.select_dtypes(include=['object']).columns
print(string_columns)

# 문자열 컬럼을 수치형으로 변환
for col in string_columns:
    if col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

print(f"\n테스트 데이터 준비 완료: {X_test.shape}")
print("최종 데이터 타입:")
print(X_test.dtypes)

# 4. SHAP Explainer 생성
print("\n=== SHAP Explainer 생성 ===")
# 테스트 데이터의 일부를 배경 데이터로 사용
background_data = X_test[:100].values
shap_explainer = xai_explainer.create_shap_explainer(background_data)

# 5. LIME Explainer 생성
print("\n=== LIME Explainer 생성 ===")
lime_explainer = xai_explainer.create_lime_explainer(background_data)

# 6. 특정 거래 예측 설명 (SHAP)
print("\n=== SHAP 기반 예측 설명 ===")
shap_explanation = xai_explainer.explain_prediction_shap(0, X_test, y_test, top_features=8)

# 7. 특정 거래 예측 설명 (LIME)
print("\n=== LIME 기반 예측 설명 ===")
lime_explanation = xai_explainer.explain_prediction_lime(0, X_test, y_test, num_features=8)

# 8. 전체 특성 중요도 분석
print("\n=== 전체 특성 중요도 분석 ===")
importance_df = xai_explainer.analyze_feature_importance_global(X_test, sample_size=1000)

# 9. 종합 설명 리포트 생성
print("\n=== 종합 설명 리포트 ===")
report = xai_explainer.create_explanation_report(0, X_test, y_test)

# 10. 시각화
print("\n=== 시각화 생성 ===")
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features["Importance"])
plt.yticks(range(len(top_features)), top_features["Feature"])
plt.xlabel("SHAP Importance")
plt.title("Top 15 Feature Importance (SHAP)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/model/xai_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 클래스 분포 시각화
plt.figure(figsize=(10, 6))
fraud_counts = y_test.value_counts()
plt.pie(fraud_counts.values, labels=["정상", "사기"], autopct="%1.1f%%", startangle=90)
plt.title("테스트 데이터 클래스 분포")
plt.axis("equal")
plt.savefig('results/model/xai_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"정상 거래: {fraud_counts[0]:,}개")
print(f"사기 거래: {fraud_counts[1]:,}개")

print("\n=== XAI 분석 완료! ===")
print("결과 이미지가 results/model/ 디렉토리에 저장되었습니다.")
