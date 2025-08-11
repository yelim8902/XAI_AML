# src/reporting.py

def generate_str_report(
    report, 
    conf_matrix, 
    shap_features, 
    threshold=None, 
    shap_llm_text=None, 
    shap_plot_path=None,
    individual_explanations=None,
    individual_plot_paths=None,
    model_info=None
):
    """
    STR 보고서 자동 생성 (Markdown + 시각화 포함)
    """
    # 주요 성능 지표
    precision = report['Fraud']['precision']
    recall = report['Fraud']['recall']
    f1 = report['Fraud']['f1-score']
    support = report['Fraud']['support']
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    # 상위 5개 피처
    top_features = shap_features.head(5)

    # 보고서 작성
    report_md = f"""
# 📄 의심거래 분석(STR) 자동 생성 보고서

## 1. 📊 모델 성능 요약
"""
    
    # 모델 정보 추가
    if model_info:
        report_md += f"- **사용 모델**: {model_info}\n"
    
    report_md += f"""
- **정밀도(Precision)**: {precision:.3f}
- **재현율(Recall)**: {recall:.3f}
- **F1-score**: {f1:.3f}
- **사기 거래 샘플 수**: {support}
- **False Positives**: {fp}
- **False Negatives**: {fn}
"""
    if threshold is not None:
        report_md += f"- **모델 임계값**: {threshold:.3f}\n"

    # SHAP 중요도
    report_md += "\n## 2. 🔍 SHAP 상위 피처 중요도\n"
    for _, row in top_features.iterrows():
        # MeanAbsSHAP 컬럼명 사용
        importance_col = 'MeanAbsSHAP' if 'MeanAbsSHAP' in row else 'Importance'
        report_md += f"- {row['Feature']}: {row[importance_col]:.6f}\n"

    # SHAP 시각화 이미지 포함
    if shap_plot_path:
        report_md += f"\n![SHAP Feature Importance]({shap_plot_path})\n"

    # 개별 거래 SHAP 설명
    if individual_explanations:
        report_md += "\n## 3. 🔍 개별 거래 SHAP 설명\n"
        report_md += "\n### 주요 거래별 모델 판단 근거\n"
        
        for i, exp in enumerate(individual_explanations[:3]):  # 상위 3개만 표시
            status_emoji = "✅" if exp['prediction_correct'] else "❌"
            pred_status = "사기" if exp['predicted'] == 1 else "정상"
            actual_status = "사기" if exp['actual'] == 1 else "정상"
            
            report_md += f"\n#### 거래 #{exp['transaction_id']} {status_emoji}\n"
            report_md += f"- **예측**: {pred_status}\n"
            report_md += f"- **실제**: {actual_status}\n"
            report_md += f"- **판단 정확도**: {'정확' if exp['prediction_correct'] else '오류'}\n"
            report_md += f"- **주요 판단 근거**:\n"
            
            for j, feature_info in enumerate(exp['top_features'][:3]):  # 상위 3개 피처만
                contribution_emoji = "📈" if feature_info['contribution'] == 'positive' else "📉"
                report_md += f"  {j+1}. **{feature_info['feature']}**: {feature_info['raw_value']:.4f} "
                report_md += f"(SHAP: {feature_info['shap_value']:+.4f}) {contribution_emoji}\n"
            
            # 개별 거래 LLM 설명 추가
            if 'llm_explanation' in exp and exp['llm_explanation']:
                report_md += f"\n**🤖 AI 해석**: {exp['llm_explanation']}\n"
            
            # 개별 거래 시각화 이미지
            if individual_plot_paths and i < len(individual_plot_paths):
                plot_filename = individual_plot_paths[i].split('/')[-1]
                report_md += f"\n![거래 #{exp['transaction_id']} SHAP 설명](../figures/{plot_filename})\n"

    # LLM 설명 (전역)
    if shap_llm_text:
        report_md += f"\n## 4. 🤖 SHAP 결과 해석 (LLM 자동 생성)\n{shap_llm_text}\n"

    report_md += "\n---\n보고서 생성 완료 ✅\n"

    return report_md
