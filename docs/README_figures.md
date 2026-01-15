
---

# 2️⃣ `docs/README_figures.md` (시각화 결과 상세 해석 전용)

👉 **`docs/README_figures.md` 로 저장**  
👉 논문 Results 섹션 느낌으로 “해석 중심” 문서입니다.

---

## 📄 docs/README_figures.md (최종본)

```markdown
# 📊 실험 결과 시각화 해석

본 문서는  
SG → BG 예측 프로젝트의 학습 결과로 생성된  
**주요 시각화 결과에 대한 해석을 정리한 문서**입니다.

---

## 1️⃣ SG–BG 산점도 분석

![SG-BG Scatter](figures/01_scatter.png)

타액 포도당(SG)과 혈당(BG) 간의 산점도는  
전반적인 **양의 상관관계**를 보여줍니다.  
SG 값이 증가함에 따라 BG 또한 증가하는 경향이 관찰되며,  
이는 SG를 이용한 BG 예측의 가능성을 시사합니다.

---

## 2️⃣ 실제값 vs 예측값 (Actual vs Predicted)

![Actual vs Pred](figures/02_actual_vs_pred.png)

예측값은 실제 BG 값 주변에 비교적 고르게 분포하며,  
대각선(y = x) 인근에 밀집되어 있습니다.  
이는 모델이 전반적인 BG 분포를 잘 학습했음을 의미합니다.

---

## 3️⃣ 잔차 분석 (Residual Plot)

![Residual](figures/03_residual.png)

잔차는 특정 구간에 편향되지 않고  
0을 중심으로 비교적 균등하게 분포합니다.  
이는 모델의 체계적인 오차가 크지 않음을 나타냅니다.

---

## 4️⃣ Bland–Altman 분석

![Bland Altman](figures/04_bland_altman.png)

대부분의 샘플이 95% 신뢰 구간 내에 위치하며,  
예측값과 실제값 간의 평균 차이가 크지 않습니다.  
이는 모델 예측의 안정성을 뒷받침합니다.

---

## 5️⃣ CEGA 분석

![CEGA](figures/05_cega.png)

대부분의 예측 결과가  
임상적으로 허용 가능한 A/B 영역에 포함되어 있습니다.  
이는 모델이 실제 임상 환경에서도  
활용 가능성이 있음을 시사합니다.

---

## 6️⃣ 모델 성능 비교

![Model Metrics](figures/06_model_metrics.png)

여러 모델의 성능을 비교한 결과,  
**LightGBM 모델이 R², RMSE, MAE 기준에서 가장 우수한 성능**을 보였습니다.  
비선형 관계를 효과적으로 학습할 수 있는  
트리 기반 모델의 장점이 반영된 결과로 해석됩니다.

---

## 7️⃣ SHAP Feature 중요도 분석

![SHAP](figures/07_shap_summary.png)

SHAP 분석 결과,  
SG가 BG 예측에 가장 큰 기여를 하는 변수로 나타났으며,  
BMI, 식후 상태, 연령 그룹 등의 변수도  
의미 있는 영향력을 보였습니다.

---

## 🔚 종합 해석

본 시각화 결과들은  
SG 및 생활·생리 변수 기반 BG 예측 모델이  
통계적·임상적으로 모두 의미 있는 성능을 보임을 시사합니다.  
특히 LightGBM 모델은  
성능과 해석 가능성 측면에서 가장 균형 잡힌 결과를 제공합니다.
