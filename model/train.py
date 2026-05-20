"""
model/train.py
Breast Cancer Wisconsin — Logistic Regression Training Script with MLflow Tracking
Loads data from: data.csv  (id, diagnosis[M/B], 30 features)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
CM_PATH     = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")


# ─────────────────────────────────────────────
# Helper — plot confusion matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, labels: list, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Logistic Regression")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved -> {save_path}")


# ─────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────
def train() -> None:
    # 1. Load CSV dataset
    csv_path = os.path.normpath(DATA_PATH)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"data.csv not found at: {csv_path}\n"
            "Place data.csv in the breast-cancer-prediction project folder."
        )

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded data.csv -> shape: {df.shape}")

    # Drop 'id' column if present; drop any unnamed trailing columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Encode diagnosis: M → 1 (Malignant), B → 0 (Benign)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    feature_cols  = [c for c in df.columns if c != "diagnosis"]
    target_names  = ["Benign (B)", "Malignant (M)"]

    X = df[feature_cols].values
    y = df["diagnosis"].values

    print(f"[INFO] Features : {len(feature_cols)}")
    print(f"[INFO] Samples  : {len(y)}  |  Malignant: {y.sum()}  |  Benign: {(y == 0).sum()}")

    # 2. Train / test split  (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4. Model hyper-parameters
    C        = 1.0
    max_iter = 10000
    solver   = "lbfgs"

    # ──────────────────────────────────────────
    # 5. MLflow experiment
    # ──────────────────────────────────────────
    mlflow.set_experiment("breast-cancer-logistic-regression")

    with mlflow.start_run(run_name="logistic-regression-run") as run:
        print(f"\n[MLflow] Run ID: {run.info.run_id}")

        # 5a. Train model
        model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 5b. Predictions
        y_pred      = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]   # prob of Malignant (class 1)

        # 5c. Compute metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        # 5d. Log parameters
        mlflow.log_param("C",            C)
        mlflow.log_param("max_iter",     max_iter)
        mlflow.log_param("solver",       solver)
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("dataset",      "data.csv")

        # 5e. Log metrics
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)

        # 5f. Confusion matrix PNG
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels=target_names, save_path=CM_PATH)

        # 5g. Save & log artifacts
        joblib.dump(model,  MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        mlflow.log_artifact(CM_PATH, artifact_path="plots")
        mlflow.log_artifact(MODEL_PATH,  artifact_path="artifacts")
        mlflow.log_artifact(SCALER_PATH, artifact_path="artifacts")

        # ──────────────────────────────────────
        # 6. Print results to console
        # ──────────────────────────────────────
        print("\n" + "=" * 55)
        print("  MODEL PERFORMANCE METRICS")
        print("=" * 55)
        print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print("=" * 55)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        print(f"\n[INFO] Model saved   -> {MODEL_PATH}")
        print(f"[INFO] Scaler saved  -> {SCALER_PATH}")
        print(f"[MLflow] Experiment  -> breast-cancer-logistic-regression")
        print(f"[MLflow] Run ID      -> {run.info.run_id}")
        print("\n[INFO] Training complete! Run `mlflow ui` to view the experiment.")


if __name__ == "__main__":
    train()
