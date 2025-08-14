#!/usr/bin/env python3
"""
Flask 백엔드 서버
LLM 기반 보고서 생성 API와 실시간 예측, AML 규제 알림을 제공합니다.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# 모델 로드
try:
    model = joblib.load('../outputs/models/model.joblib')
    scaler = joblib.load('../outputs/models/scaler.joblib')
    feature_names = joblib.load('../outputs/models/feature_space.joblib')
    print("모델 로드 성공")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """정적 파일 서빙"""
    return send_from_directory('.', filename)

@app.route('/api/predict', methods=['POST'])
def predict_transaction():
    """실시간 거래 예측 API"""
    if model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 500
    
    try:
        data = request.get_json()
        
        # 샘플 거래 데이터 (실제로는 프론트엔드에서 입력받음)
        sample_transaction = {
            'amount': data.get('amount', 1000000),
            'hour': data.get('hour', 14),
            'oldbalanceOrg': data.get('oldbalanceOrg', 5000000),
            'oldbalanceDest': data.get('oldbalanceDest', 1000000),
            'balance_zero_both': data.get('balance_zero_both', 0),
            'merchant_category': data.get('merchant_category', 1)
        }
        
        # 거래 유형에 따른 one-hot encoding
        transaction_type = sample_transaction['merchant_category']
        type_transfer = 1 if transaction_type == 1 else 0
        type_cash_in = 1 if transaction_type == 2 else 0
        type_debit = 1 if transaction_type == 4 else 0
        type_payment = 1 if transaction_type == 5 else 0
        
        # 잔액 대비 거래금액 비율 계산
        amount_to_balance_ratio = sample_transaction['amount'] / max(sample_transaction['oldbalanceOrg'], 1)
        
        # 특성 벡터 생성 (실제 모델의 피처 순서와 일치)
        features = np.array([[
            sample_transaction['amount'],
            sample_transaction['oldbalanceOrg'],
            sample_transaction['oldbalanceDest'],
            amount_to_balance_ratio,
            sample_transaction['balance_zero_both'],
            sample_transaction['hour'],
            type_transfer,
            type_cash_in,
            type_debit,
            type_payment
        ]])
        
        # 스케일링
        features_scaled = scaler.transform(features)
        
        # 예측
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]  # 사기 확률
        
        # 위험도 등급
        if probability < 0.3:
            risk_level = "낮음"
        elif probability < 0.7:
            risk_level = "보통"
        else:
            risk_level = "높음"
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'fraud_probability': float(probability),
            'risk_level': risk_level,
            'transaction_data': sample_transaction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500

@app.route('/api/aml-updates', methods=['GET'])
def get_aml_updates():
    """AML 규제 업데이트 정보 제공"""
    # 실제로는 외부 API나 RSS 피드를 통해 실시간 데이터를 가져옴
    aml_updates = [
        {
            'id': 1,
            'title': '이상거래탐지법 개정안 국회 통과',
            'summary': 'AI 기반 이상거래탐지 의무화, 실시간 모니터링 강화',
            'date': '2024-01-15',
            'impact': '높음',
            'deadline': '2024-07-01',
            'details': '금융기관은 AI 기반 실시간 이상거래탐지 시스템을 구축하고, 의심스러운 거래를 24시간 내에 신고해야 합니다.'
        },
        {
            'id': 2,
            'title': 'FIU 보고 기준 강화',
            'summary': '의심거래 보고 기준 세분화, 보고 기한 단축',
            'date': '2024-01-10',
            'impact': '중간',
            'deadline': '2024-04-01',
            'details': '의심거래 보고 기한이 48시간에서 24시간으로 단축되었습니다.'
        },
        {
            'id': 3,
            'title': '가상자산 거래 규제 강화',
            'summary': '가상자산 거래소 이상거래탐지 의무화',
            'date': '2024-01-05',
            'impact': '높음',
            'deadline': '2024-06-01',
            'details': '가상자산 거래소도 은행과 동일한 수준의 이상거래탐지 시스템을 구축해야 합니다.'
        }
    ]
    
    return jsonify({
        'success': True,
        'updates': aml_updates,
        'last_updated': datetime.now().isoformat()
    })

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    data = request.get_json()
    transaction_id = data.get('transaction_id')
    report_type = data.get('report_type', 'detailed')
    
    # 샘플 보고서 데이터
    reports = {
        '9770': {
            'detailed': '## 🚨 위험도 요약\n**위험 등급**: 높음\n**사기 확률**: 95.2%',
            'executive': '## 📊 경영진 요약\n**위험도**: 높음\n**영향**: 고객 신뢰도 위험',
            'regulatory': '## ⚠️ 규제 준수 요약\n**위반 여부**: 위반 의심\n**수준**: 중간'
        },
        '9771': {
            'detailed': '## 🚨 위험도 요약\n**위험 등급**: 높음\n**사기 확률**: 94.8%',
            'executive': '## 📊 경영진 요약\n**위험도**: 높음\n**영향**: 규제 리스크',
            'regulatory': '## ⚠️ 규제 준수 요약\n**위반 여부**: 위반 의심\n**수준**: 중간'
        },
        '22133': {
            'detailed': '## 🚨 위험도 요약\n**위험 등급**: 매우 높음\n**사기 확률**: 97.3%',
            'executive': '## 📊 경영진 요약\n**위험도**: 매우 높음\n**영향**: 긴급 조치 필요',
            'regulatory': '## ⚠️ 규제 준수 요약\n**위반 여부**: 위반 가능성 높음\n**수준**: 심각'
        }
    }
    
    report_content = reports.get(transaction_id, {}).get(report_type, '보고서를 찾을 수 없습니다.')
    
    return jsonify({
        'success': True,
        'content': report_content,
        'generated_at': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
