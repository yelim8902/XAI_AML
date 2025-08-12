#!/usr/bin/env python3
"""
Flask 백엔드 서버
LLM 기반 보고서 생성 API를 제공합니다.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

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
