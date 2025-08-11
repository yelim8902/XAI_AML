import os
from openai import OpenAI

def generate_shap_explanation(top_features_df):
    """
    SHAP 상위 피처를 기반으로 LLM이 자연어 설명을 생성
    Args:
        top_features_df (pd.DataFrame): Feature, MeanAbsSHAP 컬럼 포함 DataFrame
    Returns:
        str: LLM이 생성한 설명 텍스트
    """
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠ OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정해주세요."
    
    # 프롬프트 생성
    # MeanAbsSHAP 컬럼명 사용
    importance_col = 'MeanAbsSHAP' if 'MeanAbsSHAP' in top_features_df.columns else 'Importance'
    
    features_list = [
        f"{row['Feature']} (중요도 {row[importance_col]:.3e})"
        for _, row in top_features_df.iterrows()
    ]
    features_text = "\n".join(features_list)

    prompt = f"""
당신은 금융 사기 탐지(FDS) 전문가입니다.
다음은 사기 거래 탐지 모델에서 SHAP 분석으로 도출된 중요 피처 목록과 중요도입니다.

{features_text}

각 피처가 왜 사기 탐지에 중요하게 작용하는지, 어떤 거래 패턴에서 값이 높거나 낮으면 위험도가 증가하는지
금융업계 전문가의 시각에서 설명해 주세요.
응답은 명확하고, 보고서에 바로 쓸 수 있도록 작성해 주세요.
"""

    try:
        # OpenAI 클라이언트 생성
        client = OpenAI(api_key=api_key)
        
        # GPT 호출 (최신 API 사용)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 또는 gpt-4, gpt-4o
            messages=[
                {"role": "system", "content": "당신은 금융 데이터 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )
        explanation = response.choices[0].message.content.strip()
        return explanation

    except Exception as e:
        return f"⚠ LLM 설명 생성 중 오류 발생: {e}"


def generate_individual_transaction_explanation(transaction_data):
    """
    개별 거래에 대한 LLM 설명을 생성
    Args:
        transaction_data (dict): 거래 정보 (transaction_id, predicted, actual, top_features 등)
    Returns:
        str: LLM이 생성한 개별 거래 설명 텍스트
    """
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠ OpenAI API 키가 설정되지 않았습니다."
    
    # 거래 정보 구성
    transaction_id = transaction_data['transaction_id']
    predicted = "사기" if transaction_data['predicted'] == 1 else "정상"
    actual = "사기" if transaction_data['actual'] == 1 else "정상"
    is_correct = transaction_data['prediction_correct']
    
    # 상위 피처 정보 구성
    features_info = []
    for feature_info in transaction_data['top_features'][:3]:  # 상위 3개만
        feature_name = feature_info['feature']
        shap_value = feature_info['shap_value']
        raw_value = feature_info['raw_value']
        contribution = "증가" if feature_info['contribution'] == 'positive' else "감소"
        
        features_info.append(f"- {feature_name}: 원본값 {raw_value:.4f}, SHAP값 {shap_value:+.4f} ({contribution} 효과)")
    
    features_text = "\n".join(features_info)
    
    # 프롬프트 생성
    prompt = f"""
당신은 금융 사기 탐지(FDS) 전문가입니다.
다음은 특정 거래에 대한 모델의 판단과 SHAP 분석 결과입니다.

**거래 정보:**
- 거래 ID: #{transaction_id}
- 모델 예측: {predicted}
- 실제 결과: {actual}
- 예측 정확도: {'정확' if is_correct else '오류'}

**주요 피처 분석:**
{features_text}

위의 SHAP 분석 결과를 바탕으로, 이 거래가 왜 {predicted}로 분류되었는지 설명해주세요.
각 피처가 모델의 판단에 어떤 영향을 미쳤는지, 그리고 이 거래가 {predicted}로 분류된 주요 근거는 무엇인지
금융 전문가의 관점에서 명확하고 간결하게 설명해주세요.

응답은 2-3문장으로 간결하게 작성해주세요.
"""

    try:
        # OpenAI 클라이언트 생성
        client = OpenAI(api_key=api_key)
        
        # GPT 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 금융 사기 탐지 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        explanation = response.choices[0].message.content.strip()
        return explanation

    except Exception as e:
        return f"⚠ 개별 거래 설명 생성 중 오류 발생: {e}"
