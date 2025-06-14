import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import shutil

# === Setup Tracking URI untuk lokal & CI ===
if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Apple Quality Hyperparameter Tuning")

# === Load Dataset ===
data_path = "apple-quality_preprocessing.csv"
data = pd.read_csv(data_path)
X = data.drop("Quality", axis=1)
y = data["Quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
input_example = X_train.head()

# === Hyperparameter Grid ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

os.makedirs("mlruns", exist_ok=True)

# === Cari kombinasi terbaik ===
best_acc = 0
best_model = None
best_cm = None
best_report = None
best_params = None
best_y_pred = None

for params in ParameterGrid(param_grid):
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_cm = confusion_matrix(y_test, y_pred)
        best_report = classification_report(y_test, y_pred)
        best_params = params
        best_y_pred = y_pred

# === Log hanya model terbaik ke MLflow ===
with mlflow.start_run(run_name="Best Model Logging") as run:
    mlflow.log_params(best_params)
    report = classification_report(y_test, best_y_pred, output_dict=True)
    mlflow.log_metric("accuracy", best_acc)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])

    # Log model terbaik
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example,
        signature=infer_signature(X_train, best_y_pred)
    )

    # Log confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")
    os.remove(cm_path)

print("Training selesai. Model terbaik telah disimpan ke MLflow.")
