#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 노트북의 훈련된 나이브 베이즈 모델을 저장하는 스크립트
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

print("=== 모델 저장 스크립트 시작 ===")

# 1. 데이터 로드 및 전처리 (기존 노트북과 동일)
print("1. 데이터 로드 및 전처리...")

# 데이터 읽기 (상위 50k개)
paysim = pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv', nrows=50000)
print(f"데이터 크기: {paysim.shape}")

# Feature engineering 함수들
def balance_diff(data):
    '''balance_diff checks whether the money debited from sender has exactly credited to the receiver'''
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
    '''Creates a new column which has 1 if the transaction amount is greater than the threshold'''
    data['surge'] = [1 if n > 450000 else 0 for n in data['amount']]

def frequency_receiver(data):
    '''Creates a new column which has 1 if the receiver receives money from many individuals'''
    data['freq_Dest'] = data['nameDest'].map(data['nameDest'].value_counts())
    data['freq_dest'] = [1 if n > 20 else 0 for n in data['freq_Dest']]
    data.drop(['freq_Dest'], axis=1, inplace=True)

def merchant(data):
    '''We also have customer ids which starts with M in Receiver name, it indicates merchant'''
    values = ['M']
    conditions = list(map(data['nameDest'].str.contains, values))
    data['merchant'] = np.select(conditions, '1', '0')

# Feature engineering 적용
print("2. Feature engineering 적용...")
balance_diff(paysim)
surge_indicator(paysim)
frequency_receiver(paysim)
merchant(paysim)

# 클래스 밸런싱
print("3. 클래스 밸런싱...")
paysim_1 = paysim.copy()
max_size = paysim_1['isFraud'].value_counts().max()
lst = [paysim_1]
for class_index, group in paysim_1.groupby('isFraud'):
    lst.append(group.sample(max_size - len(group), replace=True))
paysim_1 = pd.concat(lst)

# 원-핫 인코딩
print("4. 원-핫 인코딩...")
paysim_1 = pd.concat([paysim_1, pd.get_dummies(paysim_1['type'], prefix='type_')], axis=1)
paysim_1.drop(['type'], axis=1, inplace=True)

# 데이터 분할
print("5. 데이터 분할...")
X = paysim_1.drop('isFraud', axis=1)
y = paysim_1['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=111)

# 수치형 컬럼 표준화
print("6. 수치형 컬럼 표준화...")
col_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
features_train = X_train[col_names]
features_test = X_test[col_names]

scaler = StandardScaler().fit(features_train.values)
features_train = scaler.transform(features_train.values)
features_test = scaler.transform(features_test.values)

X_train[col_names] = features_train
X_test[col_names] = features_test

# 토큰화
print("7. 토큰화...")
tokenizer_org = tf.keras.preprocessing.text.Tokenizer()
tokenizer_org.fit_on_texts(X_train['nameOrig'])

tokenizer_dest = tf.keras.preprocessing.text.Tokenizer()
tokenizer_dest.fit_on_texts(X_train['nameDest'])

customers_train_org = tokenizer_org.texts_to_sequences(X_train['nameOrig'])
customers_test_org = tokenizer_org.texts_to_sequences(X_test['nameOrig'])

customers_train_dest = tokenizer_dest.texts_to_sequences(X_train['nameDest'])
customers_test_dest = tokenizer_dest.texts_to_sequences(X_test['nameDest'])

X_train['customers_org'] = tf.keras.preprocessing.sequence.pad_sequences(customers_train_org, maxlen=1)
X_test['customers_org'] = tf.keras.preprocessing.sequence.pad_sequences(customers_test_org, maxlen=1)

X_train['customers_dest'] = tf.keras.preprocessing.sequence.pad_sequences(customers_train_dest, maxlen=1)
X_test['customers_dest'] = tf.keras.preprocessing.sequence.pad_sequences(customers_test_dest, maxlen=1)

# 불필요한 컬럼 삭제
X_train = X_train.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X_train = X_train.reset_index(drop=True)

X_test = X_test.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X_test = X_test.reset_index(drop=True)

# 2. 모델 훈련 및 하이퍼파라미터 튜닝
print("8. 모델 훈련 및 하이퍼파라미터 튜닝...")
param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

nbModel_grid = GridSearchCV(
    estimator=GaussianNB(), 
    param_grid=param_grid_nb, 
    verbose=1, 
    cv=10, 
    n_jobs=-1
)
nbModel_grid.fit(X_train, y_train)

print(f"최적 모델: {nbModel_grid.best_estimator_}")

# 3. 모델 저장
print("9. 모델 및 관련 파일 저장...")

# 모델 저장 디렉토리 생성
model_dir = 'results/model'
os.makedirs(model_dir, exist_ok=True)

# 최적화된 나이브 베이즈 모델 저장
model_path = os.path.join(model_dir, 'naive_bayes_model.pkl')
joblib.dump(nbModel_grid.best_estimator_, model_path)

# 스케일러도 저장 (XAI에서 동일한 전처리 필요)
scaler_path = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# 특성 이름도 저장 (XAI에서 필요)
feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
joblib.dump(X_train.columns.tolist(), feature_names_path)

# 토크나이저도 저장
tokenizer_org_path = os.path.join(model_dir, 'tokenizer_org.pkl')
tokenizer_dest_path = os.path.join(model_dir, 'tokenizer_dest.pkl')
joblib.dump(tokenizer_org, tokenizer_org_path)
joblib.dump(tokenizer_dest, tokenizer_dest_path)

print(f"\n=== 저장 완료 ===")
print(f"- 모델: {model_path}")
print(f"- 스케일러: {scaler_path}")
print(f"- 특성 이름: {feature_names_path}")
print(f"- 송신자 토크나이저: {tokenizer_org_path}")
print(f"- 수신자 토크나이저: {tokenizer_dest_path}")

# 4. 저장된 모델 확인
print("\n=== 저장된 모델 확인 ===")
loaded_model = joblib.load(model_path)
print(f"- 모델 타입: {type(loaded_model)}")
print(f"- 최적 파라미터: {loaded_model.get_params()}")

# 간단한 테스트
test_pred = loaded_model.predict(X_test[:5])
print(f"\n저장된 모델 테스트 예측 결과: {test_pred}")
print(f"원본 모델 테스트 예측 결과: {nbModel_grid.predict(X_test[:5])}")
print("두 결과가 동일한지 확인:", np.array_equal(test_pred, nbModel_grid.predict(X_test[:5])))

print("\n=== 모델 저장 완료! 이제 XAI 분석에서 사용할 수 있습니다. ===")
