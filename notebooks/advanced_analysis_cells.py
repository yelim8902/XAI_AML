# 노트북에 추가할 고급 분석 셀들

# ===== 셀 1: ROC 곡선 분석 =====
"""
## ROC 곡선 및 AUC 분석
"""

from sklearn.metrics import roc_curve, auc, roc_auc_score

# 예측 확률 계산
y_pred_proba = nbModel_grid.predict_proba(X_test)[:, 1]

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# ROC 곡선 시각화
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")

# ===== 셀 2: 앙상블 모델 생성 =====
"""
## 앙상블 모델 생성 및 평가
"""

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 앙상블을 위한 모델들
estimators = [
    ('nb', nbModel_grid.best_estimator_),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=123)),
    ('lr', LogisticRegression(random_state=123, max_iter=1000))
]

# 투표 분류기 생성 (soft voting 사용)
ensemble = VotingClassifier(estimators=estimators, voting='soft')

# 앙상블 모델 학습
print("앙상블 모델 학습 중...")
ensemble.fit(X_train, y_train)
print("앙상블 모델 학습 완료!")

# 앙상블 모델 성능 평가
ensemble_pred = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_pred_proba = ensemble.predict_proba(X_test)[:, 1]
ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)

print(f"앙상블 모델 정확도: {ensemble_acc:.4f}")
print(f"앙상블 모델 AUC: {ensemble_auc:.4f}")

# ===== 셀 3: 피처 중요도 분석 =====
"""
## 피처 중요도 분석 (Random Forest 기준)
"""

# Random Forest 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
rf_model.fit(X_train, y_train)

# 피처 중요도 계산
feature_importance = rf_model.feature_importances_
feature_names = X_train.columns

# 피처 중요도 데이터프레임 생성
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# 상위 15개 피처 시각화
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("상위 10개 중요 피처:")
print(importance_df.head(10))

# ===== 셀 4: 모델 성능 비교 =====
"""
## 모델 성능 비교 요약
"""

# 모델 성능 비교
models_comparison = pd.DataFrame({
    'Model': ['Naive Bayes', 'Ensemble', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, y_pred), ensemble_acc, accuracy_score(y_test, rf_model.predict(X_test))],
    'AUC': [roc_auc_score(y_test, y_pred_proba), ensemble_auc, roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])]
})

print("모델 성능 비교:")
print(models_comparison)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 정확도 비교
ax1.bar(models_comparison['Model'], models_comparison['Accuracy'], color='skyblue')
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='x', rotation=45)

# AUC 비교
ax2.bar(models_comparison['Model'], models_comparison['AUC'], color='lightcoral')
ax2.set_title('Model AUC Comparison')
ax2.set_ylabel('AUC')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ===== 셀 5: 모델 저장 =====
"""
## 향상된 모델들 저장
"""

import pickle

# 앙상블 모델 저장
with open('../results/models/enhanced_ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

# Random Forest 모델 저장
with open('../results/models/enhanced_random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# 피처 중요도 저장
with open('../results/models/enhanced_feature_importance.pkl', 'wb') as f:
    pickle.dump(importance_df, f)

print("향상된 모델들이 저장되었습니다.")

# ===== 셀 6: 최종 결론 =====
"""
## 최종 결론 및 개선 방향
"""

print("="*60)
print("최종 결론 및 개선 방향")
print("="*60)
print("1. 현재 모델 성능:")
print(f"   - Naive Bayes: 정확도 {accuracy_score(y_test, y_pred):.3f}, AUC {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"   - 앙상블: 정확도 {ensemble_acc:.3f}, AUC {ensemble_auc:.3f}")
print(f"   - Random Forest: 정확도 {accuracy_score(y_test, rf_model.predict(X_test)):.3f}, AUC {roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]):.3f}")

print("\n2. 주요 특징:")
print("   - 사기 탐지 재현율(Recall)이 높음 (FN=0)")
print("   - 정상 거래 오탐(FP)이 다소 높음")
print("   - 금융 사기 탐지에는 적합한 모델")

print("\n3. 개선 방향:")
print("   - 더 많은 피처 엔지니어링 (시간대별 패턴, 계좌별 거래 패턴 등)")
print("   - 딥러닝 모델 시도 (LSTM, Transformer 등)")
print("   - 비용 함수 조정으로 FP/FN 균형 맞추기")
print("   - 실시간 학습 시스템 구축")
print("="*60)
