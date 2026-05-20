"""
api/app.py
Breast Cancer Prediction — Flask REST API
Dataset encoding: M = 1 (Malignant / Yes), B = 0 (Benign / No)
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PATH  = os.path.join(ARTIFACT_DIR, "scaler.pkl")

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Load model + scaler at startup
# ─────────────────────────────────────────────
model  = None
scaler = None

def load_artifacts() -> None:
    global model, scaler
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. "
            "Please run `python model/train.py` first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"scaler.pkl not found at {SCALER_PATH}. "
            "Please run `python model/train.py` first."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Model and scaler loaded successfully.")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    """Health-check endpoint."""
    return jsonify({
        "status":  "ok",
        "message": "Breast Cancer Prediction API",
        "version": "1.0.0",
        "encoding": "M=1 (Malignant/Yes), B=0 (Benign/No)",
        "endpoints": {
            "GET  /":        "Health check",
            "POST /predict": "Predict malignant / benign",
        },
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a tumour is Malignant or Benign.

    Request body (JSON):
        {"features": [f1, f2, ..., f30]}

    Response (JSON):
        {
            "prediction":  1 or 0,
            "label":       "Yes (Malignant)" or "No (Benign)",
            "probability": {"malignant": float, "benign": float}
        }

    Encoding (matches data.csv training):
        1 = Malignant (M)  → "Yes"
        0 = Benign    (B)  → "No"
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    if "features" not in data:
        return jsonify({"error": "Missing key 'features' in request body."}), 400

    features = data["features"]

    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a JSON array."}), 400

    if len(features) != 30:
        return jsonify({
            "error": f"Expected 30 features, got {len(features)}.",
            "hint":  "Provide exactly 30 numeric feature values.",
        }), 400

    try:
        features_array = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid feature values: {str(exc)}"}), 400

    try:
        features_scaled = scaler.transform(features_array)
        prediction      = int(model.predict(features_scaled)[0])
        probabilities   = model.predict_proba(features_scaled)[0].tolist()

        # Encoding: class 0 = Benign (B), class 1 = Malignant (M)
        # model.classes_ = [0, 1]  →  probabilities[0]=P(Benign), probabilities[1]=P(Malignant)
        prob_benign    = round(probabilities[0], 6)
        prob_malignant = round(probabilities[1], 6)

        label = "Yes (Malignant)" if prediction == 1 else "No (Benign)"

        return jsonify({
            "prediction": prediction,          # 1 = Malignant, 0 = Benign
            "label":      label,
            "probability": {
                "malignant": prob_malignant,
                "benign":    prob_benign,
            },
        }), 200

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # Called by gunicorn
    load_artifacts()