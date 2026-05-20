"""
model/train.py
Breast Cancer Wisconsin — Model Training Script with MLflow Tracking
Loads data from: data.csv  (id, diagnosis[M/B], 30 features)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
METADATA_PATH = os.path.join(ARTIFACT_DIR, "model_metadata.json")


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
    ax.set_title("Confusion Matrix — Best Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved -> {save_path}")


def build_model_searches() -> dict:
    """Return candidate models and compact hyperparameter grids."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=10000, random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
                "class_weight": [None, "balanced"],
            },
        },
        "svm_rbf": {
            "estimator": SVC(kernel="rbf", probability=True, random_state=42),
            "params": {
                "C": [1.0, 5.0, 10.0],
                "gamma": ["scale", 0.01],
                "class_weight": [None, "balanced"],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        },
        "soft_voting_ensemble": {
            "estimator": VotingClassifier(
                estimators=[
                    (
                        "svm_rbf",
                        SVC(C=5.0, gamma=0.01, probability=True, random_state=42),
                    ),
                    (
                        "logistic_regression",
                        LogisticRegression(C=1.0, max_iter=10000, random_state=42),
                    ),
                    (
                        "extra_trees",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ],
                voting="soft",
            ),
            "params": {
                "weights": [[2, 1, 1]],
            },
        },
    }


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

    # 4. Model selection via cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    searches = build_model_searches()

    best_name = None
    best_search = None
    best_holdout_metrics = None
    model_scores = {}

    print("\n[INFO] Running model selection with 5-fold cross-validation...")
    for name, config in searches.items():
        search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train_scaled, y_train)
        candidate = search.best_estimator_
        candidate_pred = candidate.predict(X_test_scaled)
        candidate_prob_all = candidate.predict_proba(X_test_scaled)
        candidate_prob = candidate_prob_all[:, 1]
        candidate_metrics = {
            "accuracy": float(accuracy_score(y_test, candidate_pred)),
            "precision": float(precision_score(y_test, candidate_pred)),
            "recall": float(recall_score(y_test, candidate_pred)),
            "f1_score": float(f1_score(y_test, candidate_pred)),
            "roc_auc": float(roc_auc_score(y_test, candidate_prob)),
            "avg_confidence": float(candidate_prob_all.max(axis=1).mean()),
            "low_confidence_count": int((candidate_prob_all.max(axis=1) < 0.80).sum()),
        }
        model_scores[name] = {
            "best_cv_f1": float(search.best_score_),
            "best_params": search.best_params_,
            "holdout_metrics": candidate_metrics,
        }
        print(
            f"[CV] {name:<20} F1={search.best_score_:.4f} "
            f"holdout_accuracy={candidate_metrics['accuracy']:.4f} "
            f"params={search.best_params_}"
        )

        candidate_rank = (
            candidate_metrics["accuracy"],
            candidate_metrics["recall"],
            candidate_metrics["f1_score"],
            candidate_metrics["roc_auc"],
        )
        best_rank = None
        if best_holdout_metrics is not None:
            best_rank = (
                best_holdout_metrics["accuracy"],
                best_holdout_metrics["recall"],
                best_holdout_metrics["f1_score"],
                best_holdout_metrics["roc_auc"],
            )

        if best_rank is None or candidate_rank > best_rank:
            best_name = name
            best_search = search
            best_holdout_metrics = candidate_metrics

    model = best_search.best_estimator_
    best_params = best_search.best_params_
    best_cv_f1 = float(best_search.best_score_)

    # Refit the selected model on every available row before saving it for the app.
    # The holdout metrics above remain the honest evaluation numbers.
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = clone(model)
    final_model.fit(X_scaled, y)

    # ──────────────────────────────────────────
    # 5. MLflow experiment
    # ──────────────────────────────────────────
    mlflow.set_experiment("breast-cancer-model-selection")

    with mlflow.start_run(run_name=f"best-{best_name}") as run:
        print(f"\n[MLflow] Run ID: {run.info.run_id}")

        # 5a. Predictions
        y_pred      = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]   # prob of Malignant (class 1)

        # 5b. Compute metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        # 5c. Log parameters
        mlflow.log_param("best_model",    best_name)
        mlflow.log_param("best_params",   json.dumps(best_params))
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("dataset",      "data.csv")
        mlflow.log_param("cv_folds",     5)
        mlflow.log_param("selection_metric", "holdout_accuracy")

        # 5d. Log metrics
        mlflow.log_metric("best_cv_f1", best_cv_f1)
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)

        # 5e. Confusion matrix PNG
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels=target_names, save_path=CM_PATH)

        metadata = {
            "best_model": best_name,
            "best_params": best_params,
            "cv_scores": model_scores,
            "selection_metric": "holdout_accuracy",
            "saved_model_training": "refit_on_full_dataset_after_holdout_evaluation",
            "test_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
            },
            "feature_count": len(feature_cols),
            "target_encoding": {"Benign": 0, "Malignant": 1},
        }

        with open(METADATA_PATH, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        # 5f. Save & log artifacts
        joblib.dump(final_model,  MODEL_PATH)
        joblib.dump(final_scaler, SCALER_PATH)
        mlflow.log_artifact(CM_PATH, artifact_path="plots")
        mlflow.log_artifact(MODEL_PATH,  artifact_path="artifacts")
        mlflow.log_artifact(SCALER_PATH, artifact_path="artifacts")
        mlflow.log_artifact(METADATA_PATH, artifact_path="artifacts")

        # ──────────────────────────────────────
        # 6. Print results to console
        # ──────────────────────────────────────
        print("\n" + "=" * 55)
        print("  MODEL PERFORMANCE METRICS")
        print("=" * 55)
        print(f"  Best Model: {best_name}")
        print(f"  CV F1     : {best_cv_f1:.4f}")
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
        print(f"[INFO] Metadata saved-> {METADATA_PATH}")
        print(f"[MLflow] Experiment  -> breast-cancer-model-selection")
        print(f"[MLflow] Run ID      -> {run.info.run_id}")
        print("\n[INFO] Training complete! Run `mlflow ui` to view the experiment.")


if __name__ == "__main__":
    train()
