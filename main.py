# ============================================================
# Heart Attack Risk Prediction - 1D CNN
# Local VS Code / Windows Python version
# Dataset: velvetcrystal/heart-attack-risk-prediction-dataset
# ============================================================

import os

# TensorFlow 경고 메시지 줄이기
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import glob
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. Dependency check
# VS Code .py files cannot use: !pip install ...
# Install packages in terminal first:
#   py -m pip install kagglehub pandas numpy matplotlib scikit-learn tensorflow
# ============================================================

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import kagglehub

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print("\n[ERROR] 필요한 패키지가 설치되어 있지 않습니다:", missing)
    print("\n터미널에서 아래 명령어를 먼저 실행하세요:")
    print("py -m pip install kagglehub pandas numpy matplotlib scikit-learn tensorflow")
    print("\npy 명령어가 안 되면 아래 명령어를 사용하세요:")
    print("python -m pip install kagglehub pandas numpy matplotlib scikit-learn tensorflow")
    sys.exit(1)

# ============================================================
# 1. Settings
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_SLUG = "velvetcrystal/heart-attack-risk-prediction-dataset"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("========================================")
print("Heart Attack Risk Prediction - 1D CNN")
print("실행 중인 파일:", os.path.abspath(__file__))
print("Python 버전:", sys.version)
print("========================================")

# ============================================================
# 2. Kaggle dataset download
# ============================================================

print("\n[1/12] Kaggle 데이터셋 다운로드 중...")

try:
    path = kagglehub.dataset_download(DATASET_SLUG)
except Exception as e:
    print("\n[ERROR] Kaggle 데이터셋 다운로드 실패")
    print("원인:", e)
    print("\n해결 방법:")
    print("1) 인터넷 연결 확인")
    print("2) 아래 명령으로 kagglehub 설치 확인")
    print("   py -m pip install kagglehub")
    print("3) 그래도 안 되면 Kaggle 사이트에서 CSV를 직접 다운로드한 뒤 같은 폴더에 넣고 실행")
    sys.exit(1)

print("데이터셋 경로:", path)

csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

# KaggleHub 다운로드가 안 되거나 직접 받은 CSV를 쓰고 싶을 때 현재 폴더에서도 CSV 탐색
local_csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
all_csv_files = csv_files + local_csv_files

if len(all_csv_files) == 0:
    raise FileNotFoundError("CSV 파일을 찾지 못했습니다. Kaggle에서 CSV를 직접 받아 DATA_fixed.py와 같은 폴더에 넣어보세요.")

# CSV가 여러 개면 가장 큰 파일 사용
csv_file = max(all_csv_files, key=os.path.getsize)
print("사용할 CSV 파일:", csv_file)

# ============================================================
# 3. Load CSV
# ============================================================

print("\n[2/12] CSV 로드 중...")

df = pd.read_csv(csv_file)

print("원본 데이터 크기:", df.shape)
print("\n데이터 미리보기:")
print(df.head())
print("\n데이터 정보:")
print(df.info())

# ============================================================
# 4. Clean column names
# ============================================================

print("\n[3/12] 컬럼 이름 정리 중...")

df.columns = [
    str(col).strip()
       .lower()
       .replace(" ", "_")
       .replace("-", "_")
       .replace("/", "_")
       .replace("(", "")
       .replace(")", "")
    for col in df.columns
]

print("정리된 컬럼명:")
print(df.columns.tolist())

# ============================================================
# 5. Drop ID-like columns
# ============================================================

print("\n[4/12] ID성 컬럼 제거 중...")

id_like_cols = []
for col in df.columns:
    lower_col = col.lower()
    unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

    if "id" in lower_col and unique_ratio > 0.8:
        id_like_cols.append(col)

print("제거할 ID성 컬럼:", id_like_cols)
df = df.drop(columns=id_like_cols, errors="ignore")

# ============================================================
# 6. Split blood pressure style columns
# Example: "120/80" -> systolic=120, diastolic=80
# ============================================================

print("\n[5/12] 혈압 형식 문자열 컬럼 처리 중...")

for col in df.columns.tolist():
    if df[col].dtype == "object":
        sample_values = df[col].dropna().astype(str)

        if len(sample_values) > 0:
            slash_ratio = sample_values.str.contains("/", regex=False).mean()

            if slash_ratio > 0.5:
                extracted = df[col].astype(str).str.extract(
                    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)"
                )

                if extracted.shape[1] >= 2:
                    df[col + "_systolic"] = pd.to_numeric(extracted[0], errors="coerce")
                    df[col + "_diastolic"] = pd.to_numeric(extracted[1], errors="coerce")
                    print(f"'{col}' 컬럼을 수축기/이완기 혈압으로 분리했습니다.")
                    df = df.drop(columns=[col])

# ============================================================
# 7. Find target column
# ============================================================

print("\n[6/12] 타깃 컬럼 탐색 중...")

target_candidates = [
    "heart_attack_risk",
    "heart_attack",
    "risk",
    "target",
    "label",
    "class",
    "output",
    "result"
]

target_col = None
for candidate in target_candidates:
    if candidate in df.columns:
        target_col = candidate
        break

if target_col is None:
    target_col = df.columns[-1]
    print("주의: 타깃 컬럼을 자동으로 못 찾아 마지막 컬럼을 사용합니다.")

print("사용할 타깃 컬럼:", target_col)

# Drop rows with missing target
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

print("입력 데이터 크기:", X.shape)
print("타깃 데이터 크기:", y.shape)
print("\n타깃 분포:")
print(y.value_counts())

# ============================================================
# 8. Encode target
# ============================================================

print("\n[7/12] 타깃 라벨 인코딩 중...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

class_names = [str(c) for c in label_encoder.classes_]
num_classes = len(class_names)

print("클래스 이름:", class_names)
print("클래스 개수:", num_classes)

if num_classes < 2:
    raise ValueError("타깃 클래스가 2개 미만입니다. 분류 문제로 학습할 수 없습니다.")

# ============================================================
# 9. Train / validation / test split
# ============================================================

print("\n[8/12] 학습/검증/테스트 데이터 분리 중...")

class_counts = pd.Series(y_encoded).value_counts()
stratify_option = y_encoded if class_counts.min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=stratify_option
)

train_class_counts = pd.Series(y_train).value_counts()
stratify_train = y_train if train_class_counts.min() >= 2 else None

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=SEED,
    stratify=stratify_train
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ============================================================
# 10. Preprocessing
# ============================================================

print("\n[9/12] 전처리 중...")

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print("숫자형 컬럼:", numeric_cols)
print("범주형 컬럼:", categorical_cols)

try:
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot_encoder)
])

transformers = []
if len(numeric_cols) > 0:
    transformers.append(("num", numeric_transformer, numeric_cols))
if len(categorical_cols) > 0:
    transformers.append(("cat", categorical_transformer, categorical_cols))

if len(transformers) == 0:
    raise ValueError("학습에 사용할 입력 컬럼이 없습니다.")

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# If sparse matrix appears, convert to dense array
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_val_processed, "toarray"):
    X_val_processed = X_val_processed.toarray()
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

print("전처리 후 Train shape:", X_train_processed.shape)
print("전처리 후 Validation shape:", X_val_processed.shape)
print("전처리 후 Test shape:", X_test_processed.shape)

# ============================================================
# 11. Reshape for 1D CNN
# Original: (samples, features)
# CNN:      (samples, features, channels)
# ============================================================

print("\n[10/12] 1D CNN 입력 형태로 변환 중...")

X_train_cnn = X_train_processed.reshape(X_train_processed.shape[0], X_train_processed.shape[1], 1)
X_val_cnn = X_val_processed.reshape(X_val_processed.shape[0], X_val_processed.shape[1], 1)
X_test_cnn = X_test_processed.reshape(X_test_processed.shape[0], X_test_processed.shape[1], 1)

print("1D CNN 입력 shape:", X_train_cnn.shape)

# Class weights for imbalanced classes
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

# ============================================================
# 12. Build 1D CNN model
# ============================================================

print("\n[11/12] 1D CNN 모델 생성 및 학습 중...")

input_shape = (X_train_cnn.shape[1], 1)

model = keras.Sequential([
    layers.Input(shape=input_shape),

    layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

history = model.fit(
    X_train_cnn,
    y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================
# 13. Evaluation
# ============================================================

print("\n[12/12] 테스트 평가 및 결과 저장 중...")

test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)

print("\n==============================")
print("테스트 손실:", test_loss)
print("테스트 정확도:", test_accuracy)
print("==============================")

y_pred_prob = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_prob, axis=1)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report")
print(report)

print("\nConfusion Matrix")
print(cm)

# Save text results
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("Test Loss: " + str(test_loss) + "\n")
    f.write("Test Accuracy: " + str(test_accuracy) + "\n\n")
    f.write("Classification Report\n")
    f.write(report + "\n\n")
    f.write("Confusion Matrix\n")
    f.write(str(cm))

# Confusion matrix graph
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(num_classes), class_names, rotation=45)
plt.yticks(np.arange(num_classes), class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.show()

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("1D CNN Loss Graph")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_graph.png"), dpi=200, bbox_inches="tight")
plt.show()

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("1D CNN Accuracy Graph")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_graph.png"), dpi=200, bbox_inches="tight")
plt.show()

# Save model
model_path = os.path.join(OUTPUT_DIR, "heart_attack_risk_1d_cnn_model.keras")
model.save(model_path)

print("\n모델 저장 완료:", model_path)
print("그래프 저장 완료:", OUTPUT_DIR)
print("\n끝났음. 이제 보고서에 accuracy/loss/confusion matrix 넣으면 됩니다.")
