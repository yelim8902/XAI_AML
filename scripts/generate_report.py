import sys
import os
import json
import pandas as pd
import joblib
import numpy as np

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reporting import generate_str_report
from src.llm_helper import generate_shap_explanation, generate_individual_transaction_explanation  # 개별 거래 LLM 설명 함수 추가
from src.xai import save_shap_features_csv, get_explainer, explain, generate_individual_explanations, save_individual_shap_plots
from src.modeling import get_model_info  # 모델 정보 함수 추가
from src.data import PATHS

# ==============================
# 설정
# ==============================
def get_model_and_data():
    """모델과 데이터를 로드하고 SHAP 분석을 수행합니다."""
    try:
        # 모델 로드
        model_path = PATHS.models / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        model = joblib.load(model_path)
        print(f"✅ 모델 로드 완료: {model_path}")
        
        # 데이터 로드 (테스트셋)
        from src import load_paysim, time_split
        from src.features import build_features
        
        df = load_paysim()
        X_all, y_all, fspace = build_features(df)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = time_split(X_all, y_all)
        
        # 스케일러 로드 및 적용
        scaler_path = PATHS.models / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_te_scaled = pd.DataFrame(
                scaler.transform(X_te), 
                columns=X_te.columns, 
                index=X_te.index
            )
        else:
            print("⚠️ 스케일러 파일이 없어 원본 데이터를 사용합니다.")
            X_te_scaled = X_te
        
        # SHAP 분석 수행
        print("🔍 SHAP 분석 수행 중...")
        explainer = get_explainer(model)
        explanation, shap_matrix, feature_names = explain(explainer, X_te_scaled)
        
        # SHAP 피처 중요도 CSV 저장
        shap_csv_path = PATHS.figures / "shap_feature_importance.csv"
        save_shap_features_csv(shap_matrix, feature_names, shap_csv_path)
        
        # 개별 거래 SHAP 설명 생성
        print("🔍 개별 거래 SHAP 설명 생성 중...")
        y_pred = (model.predict_proba(X_te_scaled)[:, 1] > 0.5).astype(int)
        individual_explanations = generate_individual_explanations(
            shap_matrix, X_te_scaled, y_pred, y_te.values, sample_size=5, k=5
        )
        
        # 개별 거래별 LLM 설명 생성
        print("🤖 개별 거래별 LLM 설명 생성 중...")
        for exp in individual_explanations:
            exp['llm_explanation'] = generate_individual_transaction_explanation(exp)
        
        # 모델 정보 가져오기
        model_info = get_model_info(model)
        
        # 개별 거래 SHAP 시각화 저장
        print("📊 개별 거래 SHAP 시각화 저장 중...")
        sample_indices = [exp['transaction_id'] for exp in individual_explanations[:3]]  # 상위 3개만 시각화
        individual_plot_paths = save_individual_shap_plots(
            explanation, X_te_scaled, sample_indices, PATHS.figures, prefix="individual_shap"
        )
        
        return model, explanation, shap_matrix, feature_names, X_te, individual_explanations, individual_plot_paths, model_info
        
    except Exception as e:
        print(f"❌ 모델/데이터 로드 실패: {e}")
        return None, None, None, None, None

# ==============================
# 메인 실행
# ==============================
def main():
    # 출력 디렉토리 준비
    PATHS.ensure_output_dirs()
    
    # 모델과 데이터 로드
    model, explanation, shap_matrix, feature_names, X_te, individual_explanations, individual_plot_paths, model_info = get_model_and_data()
    if model is None:
        print("❌ 모델 로드 실패로 인해 보고서 생성을 중단합니다.")
        return
    
    try:
        # 1. 성능 지표 로드
        conf_matrix_path = PATHS.metrics / "confusion_matrix.csv"
        report_path = PATHS.metrics / "classification_report.json"
        threshold_path = PATHS.models / "threshold.txt"
        
        if not conf_matrix_path.exists():
            raise FileNotFoundError(f"혼동행렬 파일이 없습니다: {conf_matrix_path}")
        if not report_path.exists():
            raise FileNotFoundError(f"분류 리포트 파일이 없습니다: {report_path}")
        
        # 혼동행렬 로드
        conf_matrix = pd.read_csv(conf_matrix_path, index_col=0).values
        
        # 분류 리포트 로드
        with open(report_path, "r", encoding="utf-8") as f:
            report_dict = json.load(f)
        
        # 임계값 로드
        threshold = None
        if threshold_path.exists():
            try:
                threshold = float(threshold_path.read_text().strip())
            except:
                print("⚠️ 임계값 파일을 읽을 수 없습니다.")
        
        # 2. SHAP 피처 중요도 로드
        shap_csv_path = PATHS.figures / "shap_feature_importance.csv"
        if not shap_csv_path.exists():
            raise FileNotFoundError(f"SHAP 피처 중요도 파일이 없습니다: {shap_csv_path}")
        
        shap_features = pd.read_csv(shap_csv_path)
        
        # 3. LLM 설명 생성
        print("🤖 LLM 설명 생성 중...")
        top_features = shap_features.head(5)
        llm_explanation = generate_shap_explanation(top_features)
        
        # 4. SHAP 시각화 저장
        shap_plot_path = None
        if individual_plot_paths:
            # 상대 경로로 변환 (보고서에서 이미지가 제대로 표시되도록)
            # 파일명에서 거래 ID 추출
            plot_filename = individual_plot_paths[0].split('/')[-1]
            shap_plot_path = f"../figures/{plot_filename}"
        
        # 5. 보고서 생성
        print("📝 보고서 생성 중...")
        report_md = generate_str_report(
            report_dict,
            conf_matrix,
            shap_features,
            threshold=threshold,
            shap_llm_text=llm_explanation,
            shap_plot_path=shap_plot_path,
            individual_explanations=individual_explanations,
            individual_plot_paths=individual_plot_paths,
            model_info=model_info
        )
        
        # 6. 보고서 저장
        report_output_path = PATHS.reports / "STR_Report.md"
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        
        print(f"✅ 보고서 생성 완료: {report_output_path}")
        
    except Exception as e:
        print(f"❌ 보고서 생성 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
