# AKI_early_warning_miniproject
# 🏥 ICU 급성 신손상(AKI) 조기 예측 모델 개발

### MIMIC-IV 기반 머신러닝 예측 및 CDSS 미니 프로젝트

> ICU 환자의 임상 데이터를 활용하여 급성 신손상(AKI) 발생 위험을 예측하고, SHAP 기반 모델 해석을 통해 임상 의사결정 지원 가능성을 탐색한 프로젝트입니다.

**Tech Stack**  
`Python` `SQL` `XGBoost` `Scikit-learn` `SHAP` `Pandas` `Matplotlib`

---

## 01. Project Overview | 프로젝트 개요

### 프로젝트 배경

급성 신손상(Acute Kidney Injury, AKI)은 신장 기능이 급격하게 저하되는 상태로, 중환자실(ICU) 환자에게 발생할 수 있는 주요 합병증 중 하나입니다.

기존 AKI 진단에 활용되는 혈청 크레아티닌(Creatinine)은 신장 기능의 변화를 즉각적으로 반영하지 못할 수 있어, 다양한 임상 지표를 활용한 조기 위험 예측이 중요합니다.

본 프로젝트에서는 **MIMIC-IV 임상 데이터를 기반으로 AKI 발생 위험을 예측하는 머신러닝 모델을 구축**하고, SHAP 분석을 통해 주요 예측 요인을 해석했습니다.

또한 예측 결과를 시각화하는 CDSS(Clinical Decision Support System) 프로토타입을 구현하여 임상 의사결정 지원 가능성을 탐색했습니다.

### 프로젝트 목표

- ICU 환자의 AKI 발생 위험 예측
- 혈역학 및 신기능 관련 임상 변수 설계
- 머신러닝 모델별 예측 성능 비교
- SHAP 기반 주요 예측 변수 분석
- CDSS 프로토타입 구현

---

## 02. Data Pipeline | 데이터 분석 프로세스

### 전체 분석 과정

```text
MIMIC-IV Dataset
       ↓
01. Cohort 구축
       ↓
02. SQL 기반 임상 데이터 추출
       ↓
03. 데이터 전처리 및 EDA
       ↓
04. Feature Engineering
       ↓
05. 머신러닝 모델 학습
       ↓
06. 모델 성능 비교 및 최적화
       ↓
07. SHAP 기반 모델 해석
       ↓
08. CDSS 프로토타입 구현
```

### 데이터 구성

| 항목 | 내용 |
|---|---|
| Dataset | MIMIC-IV |
| 분석 대상 | ICU 환자 |
| 예측 대상 | AKI 발생 여부 |
| AKI 기준 | KDIGO 기반 |
| 주요 변수 | Creatinine, Urine Output, MAP, Shock Index 등 |
| 사용 모델 | Logistic Regression, Random Forest, XGBoost |

---

## 03. Exploratory Data Analysis | 탐색적 데이터 분석

### 3.1 AKI 발생 분포

총 **41,409명**의 분석 대상 환자 중 AKI 발생 환자는 **10,359명**으로 나타났습니다.

| Label | 환자 수 | 비율 |
|---|---:|---:|
| AKI | 10,359 | 25.0% |
| Non-AKI | 31,050 | 75.0% |
| Total | 41,409 | 100% |

![AKI Label Distribution](./images/fig_01_aki_label_distribution.png)

**분석 결과**

- AKI와 Non-AKI의 비율은 약 1 : 2.99
- 전체 데이터에서 클래스 분포 차이 확인
- Accuracy뿐 아니라 AUPRC, Recall, F1-score 등을 함께 고려할 필요성 확인

### 3.2 주요 임상 변수 분석

#### ① Creatinine

![Creatinine Distribution](./images/fig_05_cr_max_by_aki_zoom.png)

- AKI 환자군에서 상대적으로 높은 Creatinine 수치 관찰
- 신기능 관련 지표와 AKI 발생 여부의 연관성 확인

#### ② Urine Output

![Urine Output Distribution](./images/fig_06_urine_24h_by_aki_zoom.png)

- AKI 환자군에서 상대적으로 낮은 24시간 소변량 관찰
- KDIGO 기반 AKI 진단 기준과 연관된 데이터 분포 확인

**EDA Insight**

AKI 발생 여부에 따라 Creatinine 및 소변량의 분포 차이가 나타났습니다.

다만 해당 변수들은 AKI 진단 기준에도 활용되기 때문에, 실제 조기 예측을 위해서는 예측 시점 이후의 정보가 포함되지 않았는지 검증할 필요가 있음을 확인했습니다.

---

## 04. My Contribution | 담당 역할

### 4.1 Feature Engineering

AKI 발생과 관련된 허혈 및 저관류 상태를 정량적으로 표현하는 변수 설계를 담당했습니다.

**주요 수행 내용**

- MAP(Mean Arterial Pressure) 기반 저혈압·허혈 지속시간 관련 변수 설계
- Shock Index 기반 혈역학적 상태 정량화
- 임상적 의미를 고려한 변수 구성
- 결측값 처리 전략 검토

**핵심 기여**

단순히 측정된 임상 수치를 모델에 입력하는 것이 아니라, AKI 발생과 관련된 병태생리적 특성을 분석에 활용할 수 있는 변수로 설계했습니다.

### 4.2 Machine Learning & Model Evaluation

- XGBoost 모델 학습 및 성능 최적화
- Threshold Tuning을 통한 분류 기준 검토
- AUROC, AUPRC, Recall, F1-score 기반 성능 평가
- 머신러닝 모델별 성능 비교
- SHAP 기반 변수 중요도 분석

특히 AKI 환자를 놓치지 않는 것이 중요하다는 점을 고려하여 Recall을 포함한 여러 평가 지표를 함께 검토했습니다.

### 4.3 GitHub Collaboration

- Git Flow 기반 브랜치 전략 운영
- 개인 브랜치에서 기능별 작업 수행
- Draft Pull Request 활용
- 코드 리뷰 및 변경 사항 관리

---

## 05. Model Comparison | 모델 성능 비교

AKI 발생 여부를 예측하기 위해 Logistic Regression, Random Forest, XGBoost 모델을 비교했습니다.

| Model | AUROC | AUPRC | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.949 | 0.931 | 0.819 | 0.867 |
| Random Forest | **0.967** | 0.950 | 0.845 | 0.896 |
| **XGBoost** | 0.966 | **0.951** | **0.851** | **0.907** |

### 모델 선정 결과

**XGBoost 최종 선정**

Random Forest의 AUROC가 근소하게 높았으나, XGBoost는 AUPRC, Recall, F1-score에서 가장 우수한 성능을 보였습니다.

이에 AKI 환자 탐지 성능과 Precision·Recall의 균형을 고려하여 XGBoost를 최종 모델로 선정했습니다.

---

## 06. Final Model Performance | 최종 모델 성능

### XGBoost 최종 평가 결과

| Metric | Score |
|---|---:|
| **AUROC** | **0.9753** |
| **AUPRC** | **0.9611** |
| Precision | 0.9300 |
| Recall | 0.8874 |
| Specificity | 0.9777 |
| **F1-score** | **0.9093** |

![Final Model Performance](./images/fig_metrics_compact.png)

![Confusion Matrix](./images/fig_confusion_matrix_test.png)

### 주요 결과

**실제 AKI 환자 2,060명 중 1,828명을 AKI로 예측**

- Recall: 약 88.74%
- Precision: 93.00%
- F1-score: 0.9093

이를 통해 모델의 전반적인 분류 성능뿐 아니라 실제 AKI 환자를 탐지하는 능력을 평가했습니다.

※ 모델 비교 단계와 최종 평가의 수치는 서로 다른 실험 결과이며, 데이터 분할 및 평가 조건에 대한 추가 확인이 필요합니다.

---

## 07. Explainable AI | SHAP 분석

머신러닝 모델의 예측 결과를 해석하기 위해 SHAP(SHapley Additive exPlanations)을 활용했습니다.

### SHAP Feature Importance

![SHAP Feature Importance](./images/shap_summary_bar.png)

### 주요 분석 결과

SHAP 분석을 통해 다음과 같은 변수들이 모델 예측에 중요한 영향을 미치는 것으로 나타났습니다.

- `cr_missing`: Creatinine 관련 결측 여부
- `urine_total`: 소변량 관련 변수
- 혈압, 체온, 심박수 등 활력징후 관련 변수

### 분석 인사이트

**① 주요 임상 변수의 예측 기여도 확인**

소변량 및 활력징후 관련 변수가 AKI 예측에서 중요한 역할을 하는 것으로 나타났습니다.

**② 결측 관련 변수의 영향 확인**

`cr_missing`과 같은 결측 관련 변수가 높은 중요도를 보이면서, 모델이 임상 수치뿐 아니라 데이터의 측정 및 결측 패턴에도 영향을 받을 가능성을 확인했습니다.

이를 통해 단순한 성능 평가를 넘어 **모델이 어떤 정보를 근거로 예측하는지 확인하는 과정의 중요성**을 인식했습니다.

---

## 08. CDSS Prototype | 임상 의사결정 지원 시스템

머신러닝 모델의 예측 결과를 시각화하여 의료진의 의사결정을 지원하기 위한 CDSS 프로토타입을 구현했습니다.

### 구현 목적

- 환자의 AKI 위험 예측 결과 제공
- 주요 임상 정보 및 예측 결과 시각화
- 머신러닝 모델의 실제 활용 가능성 탐색

**Demo Video**

CDSS 시연 영상 추가 예정

※ 본 시스템은 연구 및 학습 목적으로 구현한 프로토타입이며, 실제 임상 환경에서의 사용을 위해서는 추가 검증이 필요합니다.

---

## 09. Limitations | 프로젝트 한계점

모델 구축 이후 데이터 전처리와 모델 평가 과정을 검토하면서 다음과 같은 한계점을 확인했습니다.

### 9.1 수치형 결측값 처리 전략의 검증 부족

**문제점**

- 일부 수치형 결측값을 별도 대체 없이 XGBoost에 입력
- XGBoost의 자체 결측 처리 기능을 활용했으나 다른 처리 방법과의 비교 검증 부족
- 결측 패턴이 모델 예측에 미치는 영향에 대한 추가 검토 필요

**개선 방향**

- 변수별 결측률 및 결측 패턴 분석
- 결측값 대체 방식과 XGBoost 자체 결측 처리 방식 비교
- 결측 관련 변수의 중요도 및 예측 영향 검증

### 9.2 모델 평가 체계의 보완 필요

**문제점**

- 일부 교차검증 과정에서 AUROC와 AUPRC를 중심으로 평가 지표 집계
- Fold별 Precision, Recall, F1-score 분석이 충분하지 않음
- 데이터 분할에 따른 성능 변동 검토 필요

**개선 방향**

- 다양한 성능 평가 지표 활용
- Fold별 성능 비교 및 안정성 검증
- 평가 조건과 데이터 분할 기준 명확화

### 9.3 데이터 불균형 대응 방법의 비교 부족

**문제점**

- AKI 발생 환자의 비율이 약 25%로 클래스 분포 차이 존재
- `scale_pos_weight` 기반 클래스 가중치 조정 활용
- 다른 불균형 대응 방법과의 비교 실험 부족

**개선 방향**

- 클래스 가중치 적용 효과 검토
- 필요시 SMOTE 등 재표본화 방법과 비교
- AUPRC, Recall, F1-score를 종합적으로 평가

### 9.4 실제 조기 예측 가능성에 대한 검증 필요

**문제점**

- AKI 진단에 활용되는 Creatinine과 소변량 등의 변수를 예측에 사용
- 예측 시점 이후의 정보가 입력 변수에 포함되지 않았는지 검증 필요
- 모델 성능이 높더라도 실제 임상에서 조기 예측이 가능하다고 단정하기 어려움

**개선 방향**

- AKI 발생 시점과 예측 시점 구분
- 관찰 기간 및 입력 변수의 시간적 범위 명확화
- 데이터 누수 가능성 검토
- 실제 조기경보 목적에 적합한 평가 체계 구축

---

## 10. Project Extension | 최종 프로젝트로의 발전

미니 프로젝트에서는 AKI 예측 모델을 구축하고 성능을 평가하는 과정에서 **데이터 전처리와 모델 검증의 중요성**을 확인했습니다.

특히 결측값 처리의 적절성, 모델의 판단 근거, 예측 시점에 따른 데이터 누수 가능성 등을 검토하면서 단순히 성능이 높은 모델을 구축하는 것만으로는 충분하지 않다는 점을 인식했습니다.

이러한 한계점을 바탕으로 최종 프로젝트에서는 데이터 구축 및 평가 과정을 보다 체계적으로 검토하고, AKI 발생 여부 예측에서 중증도 분류까지 분석 범위를 확장했습니다.

### 미니 프로젝트와 최종 프로젝트의 차이

| 구분 | 미니 프로젝트 | 최종 프로젝트의 확장 방향 |
|---|---|---|
| 예측 목표 | AKI 발생 여부 예측 | AKI 발생 예측 + Stage 분류 |
| 데이터 구축 | ICU 환자 데이터 기반 분석 | 코호트 및 예측 시점 정의 구체화 |
| Feature Engineering | 허혈·관류 관련 변수 설계 | 임상 변수 및 전처리 과정 재검토 |
| 결측값 처리 | XGBoost 자체 결측 처리 활용 | 결측 처리 전략의 적절성 검증 |
| 모델 평가 | 모델 간 성능 비교 | 평가 지표 및 검증 체계 보완 |
| 모델 해석 | SHAP 중요도 분석 | 주요 변수와 예측 근거 재검토 |

### 핵심 발전 과정

**모델 구축 → 한계점 발견 → 데이터 처리 및 검증 과정 개선 → 예측 과제 확장**

미니 프로젝트에서 도출한 개선 과제를 후속 프로젝트로 연결하여, 데이터 분석의 신뢰성과 활용 가능성을 높이는 방향으로 발전시켰습니다.

※ 구체적인 개선 방법 및 최종 성능은 후속 프로젝트의 분석 결과를 기준으로 별도로 기술합니다.

---

## 11. Tech Stack | 기술 스택

| Category | Tools |
|---|---|
| Language | Python, SQL |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Visualization | Matplotlib |
| Collaboration | Git, GitHub |

---

## 12. Retrospective | 프로젝트 회고

### 데이터 분석 역량

SQL 기반 임상 데이터 추출부터 Feature Engineering, 머신러닝 모델링, 성능 평가 및 해석까지 데이터 분석의 전반적인 과정을 경험했습니다.

### 문제 해결 역량

높은 예측 성능을 달성하는 것뿐 아니라, 결측값 처리와 데이터 누수 가능성 등 모델의 신뢰성에 영향을 줄 수 있는 문제를 발견하고 개선 방향을 도출했습니다.

### 협업 역량

Git Flow 기반 브랜치 관리와 Pull Request를 활용하여 팀원들과 코드 변경 사항을 공유하고 협업했습니다.

### 프로젝트를 통해 배운 점

> 데이터 분석에서 중요한 것은 높은 성능 수치를 얻는 것만이 아니라, 결과가 도출된 과정을 이해하고 검증하는 것임을 배웠습니다.
>
> 미니 프로젝트에서 발견한 한계점을 후속 프로젝트의 개선 과제로 연결하며, 문제를 발견하고 해결 방향을 찾아가는 분석 경험을 쌓았습니다.

---

**Project Type:** Team Mini Project

**My Contribution:** Feature Engineering · XGBoost Modeling · Model Evaluation · SHAP Analysis · GitHub Collaboration

**Dataset:** MIMIC-IV

