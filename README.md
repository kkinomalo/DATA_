# Heart Attack Risk Prediction - 1D CNN

## 1. 프로젝트 소개

이 프로젝트는 Kaggle의 **Heart Attack Risk Prediction Dataset**을 사용하여 심장마비 위험 여부를 예측하는 1D CNN 기반 분류 모델입니다.

일반적인 CNN은 이미지 데이터에 많이 사용되지만, 이 데이터셋은 이미지가 아니라 환자의 건강 정보가 들어 있는 표 형식 데이터입니다. 그래서 2D CNN이 아니라 각 환자의 특성 벡터를 1차원 입력으로 변환한 뒤 **Conv1D**를 적용했습니다.

## 2. 사용 데이터셋

- 데이터셋 이름: Heart Attack Risk Prediction Dataset
- 출처: Kaggle
- 데이터 형태: CSV 기반 표 데이터
- 목표: 환자 정보를 바탕으로 심장마비 위험 여부를 0/1 클래스로 분류

데이터셋은 `kagglehub`를 사용하여 코드 실행 시 자동으로 다운로드됩니다.

## 3. 개발 환경

권장 환경:

- Python 3.11 또는 3.12
- Windows / macOS / Linux
- VS Code 또는 PyCharm

> 주의: Python 3.14에서는 TensorFlow 설치가 안 될 수 있으므로 Python 3.11 사용을 권장합니다.

## 4. 설치 방법

### 4-1. 저장소 클론

```bash
git clone https://github.com/본인아이디/heart-attack-risk-1d-cnn.git
cd heart-attack-risk-1d-cnn
```

### 4-2. 가상환경 생성 및 실행

Windows PowerShell 기준:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

실행 정책 오류가 발생하면 아래 명령어를 한 번 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4-3. 라이브러리 설치

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. 실행 방법

```powershell
python -u main.py
```

또는:

```powershell
python -u DATA_fixed.py
```

실행하면 `outputs/` 폴더에 다음 결과가 저장됩니다.

```text
outputs/
  accuracy_graph.png
  loss_graph.png
  confusion_matrix.png
  classification_report.txt
  heart_attack_risk_1d_cnn_model.keras
```

## 6. 프로젝트 구조

```text
heart-attack-risk-1d-cnn/
  main.py
  DATA_fixed.py
  requirements.txt
  README.md
  REPORT.md
  .gitignore
  outputs/
    .gitkeep
  docs/
    github_upload_guide.md
```

## 7. 모델 구조

모델은 다음과 같은 순서로 구성했습니다.

```text
Input
→ Conv1D
→ BatchNormalization
→ MaxPooling1D
→ Conv1D
→ BatchNormalization
→ MaxPooling1D
→ Conv1D
→ BatchNormalization
→ GlobalAveragePooling1D
→ Dense
→ Dropout
→ Dense Softmax
```

## 8. 전처리 과정

- 컬럼명 정리
- ID 성격의 컬럼 제거
- 혈압처럼 `120/80` 형태인 문자열 컬럼 분리
- 숫자형 데이터 결측치 처리 및 표준화
- 범주형 데이터 결측치 처리 및 원-핫 인코딩
- 타깃 라벨 인코딩
- 학습 / 검증 / 테스트 데이터 분리
- CNN 입력 형태 `(샘플 수, 특성 수, 1)`로 변환

## 9. 실행 결과 예시

실행 결과 예시 Confusion Matrix는 다음과 같습니다.

```text
              Predicted 0   Predicted 1
True 0              700           112
True 1              220           368
```

전체 테스트 데이터 1400개 중 1068개를 맞추어, 예시 기준 테스트 정확도는 약 **76.3%**입니다.
<img width="1198" height="1108" alt="image" src="https://github.com/user-attachments/assets/b035adbf-135d-4438-acca-07a2b9ef97e1" />
<img width="1614" height="1096" alt="image" src="https://github.com/user-attachments/assets/afdd96cd-9a2d-47c6-ae6d-6e48dafd26c3" />
<img width="1574" height="1084" alt="image" src="https://github.com/user-attachments/assets/afdd4805-8b47-404a-8775-0a29acb37a50" />


## 10. 결과 해석

모델은 클래스 0에 대해서는 비교적 높은 예측 성능을 보였지만, 클래스 1에 대해서는 일부 데이터를 클래스 0으로 잘못 예측했습니다. 따라서 실제 의료 진단 목적이 아니라 머신러닝/딥러닝 학습용 예측 실험으로 해석해야 합니다.

## 11. 한계점

- 표 데이터에서는 CNN보다 Random Forest, XGBoost, MLP 같은 모델이 더 좋은 성능을 낼 수도 있습니다.
- 의료 관련 데이터이므로 실제 진단에 사용하기 위해서는 전문가 검증이 필요합니다.
- 데이터 불균형이 있을 경우 클래스 1 예측 성능이 낮아질 수 있습니다.

## 12. 결론

본 프로젝트를 통해 표 형식 의료 데이터에도 1D CNN을 적용할 수 있음을 확인했습니다. Conv1D 계층을 사용해 여러 건강 지표 사이의 패턴을 학습하고, 최종적으로 심장마비 위험 여부를 분류했습니다.
