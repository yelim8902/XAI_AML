#!/usr/bin/env python3
"""
LLM 기반 보고서 생성 시스템
OpenAI GPT-4 API를 사용하여 SHAP 분석 결과를 바탕으로 전문적인 금융 사기 탐지 보고서를 생성합니다.
"""

import os
import json
import openai
from datetime import datetime
from typing import Dict, List, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMReportGenerator:
    """LLM을 사용한 보고서 생성기"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: OpenAI API 키 (환경변수에서 자동 로드)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. OPENAI_API_KEY 환경변수를 설정하거나 직접 전달하세요.")
        
        openai.api_key = self.api_key
        self.model = "gpt-4"
        
    def generate_fraud_report(self, 
                            transaction_id: str,
                            shap_values: Dict[str, float],
                            transaction_data: Dict[str, any],
                            report_type: str = "detailed") -> str:
        """
        사기 탐지 보고서 생성
        
        Args:
            transaction_id: 거래 ID
            shap_values: SHAP 값 딕셔너리
            transaction_data: 거래 데이터
            report_type: 보고서 유형 (detailed, executive, regulatory)
            
        Returns:
            생성된 보고서 텍스트
        """
        
        # SHAP 값 정렬 (영향도 순)
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_shap[:5]  # 상위 5개 특성
        
        # 보고서 유형별 프롬프트 설정
        prompts = {
            "detailed": self._get_detailed_prompt(),
            "executive": self._get_executive_prompt(),
            "regulatory": self._get_regulatory_prompt()
        }
        
        prompt = prompts.get(report_type, prompts["detailed"])
        
        # 프롬프트에 데이터 삽입
        formatted_prompt = prompt.format(
            transaction_id=transaction_id,
            top_features=json.dumps(top_features, indent=2, ensure_ascii=False),
            transaction_data=json.dumps(transaction_data, indent=2, ensure_ascii=False),
            current_date=datetime.now().strftime("%Y년 %m월 %d일")
        )
        
        try:
            logger.info(f"거래 {transaction_id}에 대한 {report_type} 보고서 생성 시작")
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 금융 사기 탐지 전문가입니다. SHAP 분석 결과를 바탕으로 전문적이고 정확한 보고서를 작성해주세요."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,  # 일관성 있는 출력을 위해 낮은 온도
                max_tokens=2000
            )
            
            report = response.choices[0].message.content
            logger.info(f"거래 {transaction_id} 보고서 생성 완료")
            
            return report
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            return self._get_fallback_report(transaction_id, top_features, report_type)
    
    def _get_detailed_prompt(self) -> str:
        """상세 분석 보고서용 프롬프트"""
        return """
다음 SHAP 분석 결과를 바탕으로 전문적인 금융 사기 탐지 상세 분석 보고서를 작성해주세요.

## 거래 정보
- 거래 ID: {transaction_id}
- 분석 일시: {current_date}

## SHAP 분석 결과 (상위 5개 특성)
{top_features}

## 거래 데이터
{transaction_data}

## 보고서 작성 요구사항
다음 형식으로 한국어로 작성해주세요:

### 1. 🚨 위험도 요약
- 위험 등급 (낮음/보통/높음/매우 높음)
- 사기 확률 (백분율)
- 권장 조치 수준

### 2. 🔍 주요 판단 근거
- 각 SHAP 특성별 상세 분석
- 위험 요소 설명
- 패턴 분석
- 일반적 거래와의 차이점

### 3. 📊 SHAP 분석 결과 해석
- 전체적인 위험도 평가
- 특성 간 상관관계
- 사기 패턴 유형 판단

### 4. 🛡️ 권장 조치사항
- 즉시 조치사항
- 추가 조사 항목
- 규제 보고 요항
- 고객 연락 방안

### 5. 📈 향후 모니터링
- 유사 패턴 탐지 방안
- 위험도 점수 업데이트
- 데이터베이스 관리 방안

보고서는 금융 전문가가 읽을 수 있도록 전문적이고 객관적으로 작성해주세요.
"""
    
    def _get_executive_prompt(self) -> str:
        """경영진 요약 보고서용 프롬프트"""
        return """
다음 SHAP 분석 결과를 바탕으로 경영진이 읽을 수 있는 요약 보고서를 작성해주세요.

## 거래 정보
- 거래 ID: {transaction_id}
- 분석 일시: {current_date}

## SHAP 분석 결과 (상위 5개 특성)
{top_features}

## 거래 데이터
{transaction_data}

## 보고서 작성 요구사항
다음 형식으로 한국어로 작성해주세요:

### 1. 📊 경영진 요약
- 위험도 수준 (간단한 등급)
- 비즈니스 영향도
- 주요 우려사항

### 2. 🎯 핵심 위험 요소
- 가장 중요한 2-3개 위험 요소
- 비즈니스 관점에서의 의미
- 손실 가능성

### 3. 📈 비즈니스 영향
- 고객 신뢰도 영향
- 규제 리스크
- 재무적 영향

### 4. 💼 경영진 권고사항
- 즉시 결정 필요사항
- 자원 할당 권고
- 전략적 방향성

### 5. 🔮 전략적 방향
- 장기적 위험 관리 방안
- 시스템 개선 제안
- 경쟁 우위 확보 방안

경영진이 빠르게 이해할 수 있도록 핵심만 간결하게 작성해주세요.
"""
    
    def _get_regulatory_prompt(self) -> str:
        """규제 기관 보고서용 프롬프트"""
        return """
다음 SHAP 분석 결과를 바탕으로 규제 기관에 제출할 수 있는 보고서를 작성해주세요.

## 거래 정보
- 거래 ID: {transaction_id}
- 분석 일시: {current_date}

## SHAP 분석 결과 (상위 5개 특성)
{top_features}

## 거래 데이터
{transaction_data}

## 보고서 작성 요구사항
다음 형식으로 한국어로 작성해주세요:

### 1. ⚠️ 규제 준수 요약
- 규제 위반 여부
- 위반 수준 (경미/중간/심각)
- 관련 법규 조항

### 2. 📋 규제 위반 요소
- 구체적인 위반 사항
- 위반 근거
- 증거 자료

### 3. 📊 규제 준수 분석
- 현재 준수 상태
- 과거 위반 이력과의 비교
- 업계 평균 대비 수준

### 4. 📝 규제 기관 보고 요항
- STR 제출 필요성
- 추가 보고 자료
- 수사 기관 연락 필요성

### 5. 🔍 규제 감독 계획
- 향후 모니터링 방안
- 개선 조치 계획
- 정기 보고 일정

규제 기관의 요구사항에 맞춰 정확하고 객관적으로 작성해주세요.
"""
    
    def _get_fallback_report(self, transaction_id: str, top_features: List, report_type: str) -> str:
        """API 오류 시 대체 보고서"""
        return f"""
# 거래 {transaction_id} 사기 탐지 분석 보고서

## ⚠️ 시스템 오류로 인한 대체 보고서

OpenAI API 연결에 문제가 발생하여 자동 생성된 보고서를 제공합니다.

## 🔍 주요 SHAP 특성
{chr(10).join([f"- {feature}: {value}" for feature, value in top_features])}

## 📋 권장 조치사항
1. 시스템 관리자에게 API 연결 상태 확인 요청
2. 수동 분석을 통한 추가 검토
3. 정상 복구 후 재분석 수행

## 📞 기술 지원
시스템 관리자에게 연락하여 문제를 해결해주세요.
"""

def main():
    """테스트 실행"""
    try:
        # API 키 확인
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            print("다음 명령어로 설정하세요:")
            print("export OPENAI_API_KEY='your-api-key-here'")
            return
        
        # 보고서 생성기 초기화
        generator = LLMReportGenerator(api_key)
        
        # 샘플 데이터
        sample_shap = {
            "현금입금유형": 39461.0,
            "잔액대비거래금액비율": 30624.0,
            "송금인계좌잔액": 21934.0,
            "거래시간": 15420.0,
            "거래금액": 12345.0
        }
        
        sample_transaction = {
            "거래ID": "9770",
            "거래유형": "CASH_IN",
            "거래금액": 1000000,
            "송금인잔액": 50000,
            "수취인잔액": 1500000,
            "거래시간": "14:30"
        }
        
        # 상세 보고서 생성
        print("🔍 상세 분석 보고서 생성 중...")
        detailed_report = generator.generate_fraud_report(
            "9770", sample_shap, sample_transaction, "detailed"
        )
        
        print("\n" + "="*50)
        print("📋 상세 분석 보고서")
        print("="*50)
        print(detailed_report)
        
        # 경영진 요약 보고서 생성
        print("\n" + "="*50)
        print("📊 경영진 요약 보고서")
        print("="*50)
        executive_report = generator.generate_fraud_report(
            "9770", sample_shap, sample_transaction, "executive"
        )
        print(executive_report)
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
