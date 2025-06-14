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
signature = infer_signature(X_train, X_train[:1])  # Untuk log model

# === Hyperparameter Grid ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

os.makedirs("artifacts", exist_ok=True)

# === Parent Run ===
with mlflow.start_run(run_name="Grid Search Tuning"):
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

        with mlflow.start_run(nested=True, run_name=f"Params: {params}"):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", report['weighted avg']['precision'])
            mlflow.log_metric("recall", report['weighted avg']['recall'])
            mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=infer_signature(X_train, y_pred)
            )

            # === Confusion Matrix Plot ===
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            cm_filename = f"artifacts/cm_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}.png"
            plt.savefig(cm_filename)
            plt.close()

            mlflow.log_artifact(cm_filename, artifact_path="plots")

print("Training & logging selesai.")
