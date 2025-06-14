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
    mlflow.set_tracking_uri("file:///tmp/mlruns")
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

os.makedirs("model-artifacts", exist_ok=True)

# === Parent Run ===
best_acc = 0
best_model = None
best_cm = None
best_report = None
best_params = None
best_run_id = None

with mlflow.start_run(run_name="Grid Search Tuning") as parent_run:
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
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        with mlflow.start_run(nested=True, run_name=f"Params: {params}") as child_run:
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", report['weighted avg']['precision'])
            mlflow.log_metric("recall", report['weighted avg']['recall'])
            mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model-artifacts",
                input_example=input_example,
                signature=infer_signature(X_train, y_pred)
            )

            # Simpan confusion matrix sementara (per kombinasi)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            cm_filename = f"cm_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}.png"
            plt.savefig(cm_filename)
            plt.close()
            mlflow.log_artifact(cm_filename, artifact_path="plots")
            os.remove(cm_filename)

            # Simpan model terbaik
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_cm = cm
                best_report = classification_report(y_test, y_pred)
                best_params = params
                best_run_id = child_run.info.run_id

# === Simpan artifacts dari model terbaik ===
if best_model:
    joblib.dump(best_model, "model-artifacts/model.pkl")

    # Confusion matrix dari model terbaik
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Best Model - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("model-artifacts/confusion_matrix.png")
    plt.close()

    # Classification report
    with open("model-artifacts/classification_report.txt", "w") as f:
        f.write(best_report)

    # === Copy all MLflow-logged model files ===
    model_src = os.path.join("mlruns", best_run_id, "artifacts", "model")
    model_dst = os.path.join("model-artifacts", "model")
    if os.path.exists(model_src):
        shutil.copytree(model_src, model_dst, dirs_exist_ok=True)

    print(f"Best model saved with params: {best_params}")
else:
    print("No best model found.")

print("Training & logging selesai.")
