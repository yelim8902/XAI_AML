#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAI 금융 사기 탐지 프로젝트 - 메인 실행 파일

이 파일은 프로젝트의 주요 기능들을 실행할 수 있는 통합 인터페이스입니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """메인 실행 함수"""
    print("🚀 XAI 금융 사기 탐지 프로젝트")
    print("=" * 50)
    
    while True:
        print("\n📋 실행할 작업을 선택하세요:")
        print("1. 🎯 향상된 모델 훈련 및 XAI 분석")
        print("2. 🔍 XAI 분석만 실행")
        print("3. 🧪 특성 스케일링 테스트")
        print("4. 📊 결과 확인")
        print("5. 🚪 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == "1":
            print("\n🎯 향상된 모델 훈련 및 XAI 분석을 시작합니다...")
            try:
                from src.models.enhanced_fraud_detection import EnhancedFraudDetectionModel
                model = EnhancedFraudDetectionModel()
                model.run_pipeline()
                print("✅ 모델 훈련 및 XAI 분석이 완료되었습니다!")
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                
        elif choice == "2":
            print("\n🔍 XAI 분석을 시작합니다...")
            try:
                from src.analysis.xai_analysis import XAIAnalysis
                analysis = XAIAnalysis()
                analysis.run_analysis()
                print("✅ XAI 분석이 완료되었습니다!")
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                
        elif choice == "3":
            print("\n🧪 특성 스케일링 테스트를 시작합니다...")
            try:
                from tests.test_scaling import test_improved_scaling
                success = test_improved_scaling()
                if success:
                    print("✅ 모든 테스트가 성공적으로 완료되었습니다!")
                else:
                    print("❌ 테스트 중 문제가 발생했습니다.")
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                
        elif choice == "4":
            print("\n📊 결과를 확인합니다...")
            show_results()
            
        elif choice == "5":
            print("\n👋 프로젝트를 종료합니다. 감사합니다!")
            break
            
        else:
            print("❌ 잘못된 선택입니다. 1-5 중에서 선택해주세요.")

def show_results():
    """결과 파일들을 확인"""
    print("\n📁 프로젝트 결과 파일들:")
    
    # 모델 파일들
    models_dir = Path("results/models")
    if models_dir.exists():
        print(f"\n🎯 모델 파일들 ({models_dir}):")
        for file in models_dir.glob("*.pkl"):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  📄 {file.name} ({size_mb:.1f} MB)")
    
    # 차트 파일들
    figures_dir = Path("results/figures")
    if figures_dir.exists():
        print(f"\n📊 차트 파일들 ({figures_dir}):")
        for file in figures_dir.glob("*.png"):
            size_kb = file.stat().st_size / 1024
            print(f"  🖼️ {file.name} ({size_kb:.1f} KB)")
    
    # 메트릭 파일들
    metrics_dir = Path("results/metrics")
    if metrics_dir.exists():
        print(f"\n📈 메트릭 파일들 ({metrics_dir}):")
        for file in metrics_dir.glob("*"):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"  📊 {file.name} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
