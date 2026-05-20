import os
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Risk Assessment Tool",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants & Data Loading
# ─────────────────────────────────────────────
_UI_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJ    = os.path.dirname(_UI_DIR)
_CSV     = os.path.normpath(os.path.join(_PROJ, "data.csv"))
API_URL  = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
APP_VERSION = "2026-05-20-v2"

@st.cache_data
def load_dataset():
    df = pd.read_csv(_CSV)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

df            = load_dataset()
FEATURE_NAMES = [c for c in df.columns if c != "diagnosis"]
_mins         = df[FEATURE_NAMES].min().values
_maxs         = df[FEATURE_NAMES].max().values
_means        = df[FEATURE_NAMES].mean().values
HIGH_CONFIDENCE_THRESHOLD = 0.80
MODERATE_CONFIDENCE_THRESHOLD = 0.65

# Known sample rows
_mal_row = df[df["diagnosis"] == 1].iloc[0][FEATURE_NAMES].tolist()
_ben_row = df[df["diagnosis"] == 0].iloc[0][FEATURE_NAMES].tolist()


def confidence_level(probability):
    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if probability >= MODERATE_CONFIDENCE_THRESHOLD:
        return "moderate"
    return "borderline"


def result_message(prediction, malignant_prob, benign_prob):
    predicted_prob = malignant_prob if prediction == 1 else benign_prob
    level = confidence_level(predicted_prob)

    if prediction == 1:
        if level == "high":
            return "warning", "The model indicates a high probability of malignancy. Medical review is recommended."
        if level == "moderate":
            return "warning", "The model leans malignant with moderate confidence. Additional clinical review is recommended."
        return "warning", "The model slightly leans malignant, but confidence is borderline. Please review the measurements and confirm with clinical evaluation."

    if level == "high":
        return "success", "The model indicates a high probability of a benign tumor. Routine observation may be appropriate in an educational demo context."
    if level == "moderate":
        return "info", "The model leans benign with moderate confidence. Consider reviewing the inputs before relying on this result."
    return "warning", "The model slightly leans benign, but confidence is borderline. This should not be treated as a strong benign result."

# ─────────────────────────────────────────────
# Professional Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 2.2rem; 
    font-weight: 600;
    color: #0F172A; /* Slate 900 */
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #64748B; /* Slate 500 */
    font-size: 1rem; 
    margin-bottom: 2rem;
}
.alert-card {
    padding: 1.25rem 1.5rem; 
    border-radius: 6px; 
    margin: 1.5rem 0;
    font-size: 1.2rem; 
    font-weight: 600;
    border-left: 6px solid;
}
.alert-malignant {
    background-color: #FEF2F2;
    color: #991B1B;
    border-color: #DC2626;
}
.alert-benign {
    background-color: #F0FDF4;
    color: #166534;
    border-color: #16A34A;
}
.info-box {
    background: #F8FAFC; 
    border-left: 4px solid #3B82F6;
    border-radius: 4px; 
    padding: 1rem 1.2rem; 
    margin-bottom: 1.5rem;
    font-size: 0.9rem; 
    color: #334155;
}
.disclaimer {
    text-align: center; 
    color: #94A3B8; 
    font-size: 0.8rem;
    border-top: 1px solid #E2E8F0; 
    padding-top: 1rem; 
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown("""
    This application utilizes a tuned **soft-voting ensemble** trained on 569 patient records to evaluate tumor characteristics and predict diagnostic outcomes.
    
    **Instructions:**
    Use **Load Sample Data** or **Random Patient Profile** to auto-populate the assessment fields. Manual entry is reserved for users with active lab measurements.

    **Performance Metrics:**
    * **Accuracy:** 99.12%
    * **Precision:** 100.00%
    * **ROC-AUC:** 99.83%
    """)
    st.divider()
    st.caption(f"App version: {APP_VERSION}")
    st.caption("Group 2 · Machine Learning Project\nBreast Cancer Wisconsin Dataset")

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Diagnostic Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Soft-Voting Ensemble · Breast Cancer Wisconsin Dataset</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>Notice:</strong> For demonstration purposes, you may load pre-configured clinical profiles or draw a random patient record from the dataset to test the model's accuracy.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Mode Selector
# ─────────────────────────────────────────────
input_col1, input_col2 = st.columns([1, 2])

with input_col1:
    input_mode = st.radio(
        "Data Entry Method",
        options=["Load Sample Data", "Random Patient Profile", "Manual Entry"],
        label_visibility="collapsed"
    )

# ─────────────────────────────────────────────
# Determine default values based on mode
# ─────────────────────────────────────────────
true_label = None

with input_col2:
    if input_mode == "Load Sample Data":
        sample_choice = st.selectbox(
            "Select Clinical Profile:",
            ["Malignant Profile (Expected: Malignant)", "Benign Profile (Expected: Benign)"],
            label_visibility="collapsed"
        )
        if "Malignant" in sample_choice:
            default_values = _mal_row
            true_label     = "Malignant"
            st.info("Loaded a known Malignant patient record.")
        else:
            default_values = _ben_row
            true_label     = "Benign"
            st.info("Loaded a known Benign patient record.")

    elif input_mode == "Random Patient Profile":
        if st.button("Draw New Patient Record"):
            st.session_state["random_idx"] = random.randint(0, len(df) - 1)
        if "random_idx" not in st.session_state:
            st.session_state["random_idx"] = random.randint(0, len(df) - 1)

        idx           = st.session_state["random_idx"]
        rand_row      = df.iloc[idx]
        default_values= rand_row[FEATURE_NAMES].tolist()
        actual        = "Malignant" if rand_row["diagnosis"] == 1 else "Benign"
        true_label    = actual
        
        st.info(f"Loaded Patient ID #{idx}. (Ground Truth: **{actual}**)")

    else:
        default_values = _means.tolist()
        st.warning("Manual entry activated. Fields pre-filled with dataset means.")

st.divider()

# ─────────────────────────────────────────────
# Feature Form — Organised in Tabs
# ─────────────────────────────────────────────
st.markdown("#### Clinical Parameters")

# Wrapping inputs in a form prevents the app from reloading on every keystroke
with st.form("clinical_parameters_form"):
    tab1, tab2, tab3 = st.tabs(["Mean Metrics (1-10)", "Standard Error (11-20)", "Worst Metrics (21-30)"])
    
    feature_values = [0.0] * 30

    def render_feature_grid(tab, start, end, description):
        """Render a 2-column grid of number inputs for features[start:end]."""
        with tab:
            st.caption(description)
            pairs = [(i, FEATURE_NAMES[i]) for i in range(start, end)]
            rows  = [pairs[i:i+2] for i in range(0, len(pairs), 2)]

            for row in rows:
                cols = st.columns(2)
                for col_idx, (feat_idx, fname) in enumerate(row):
                    display = fname.replace("_", " ").title()
                    val = cols[col_idx].number_input(
                        label    = display,
                        min_value= float(_mins[feat_idx]),
                        max_value= float(_maxs[feat_idx]) * 2,
                        value    = float(default_values[feat_idx]),
                        format   = "%.5f",
                        key      = f"feat_{feat_idx}",
                    )
                    feature_values[feat_idx] = val

    render_feature_grid(tab1, 0,  10, "Average measurements across all evaluated cell nuclei.")
    render_feature_grid(tab2, 10, 20, "Variance between cells (higher values indicate greater irregularity).")
    render_feature_grid(tab3, 20, 30, "Most extreme measurements recorded (highly predictive factors).")

    st.write("") # Spacer
    predict_clicked = st.form_submit_button("Run Diagnostic Assessment", type="primary", use_container_width=True)

    # ─────────────────────────────────────────────
# Prediction Execution & Results
# ─────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Processing clinical data..."):
        try:
            resp = requests.post(API_URL, json={"features": feature_values}, timeout=10)

            if resp.status_code == 200:
                result     = resp.json()
                prediction = result.get("prediction")
                label      = result.get("label", "")
                prob       = result.get("probability", {})
                m_prob     = prob.get("malignant", 0)
                b_prob     = prob.get("benign", 0)
                result_kind, guidance = result_message(prediction, m_prob, b_prob)

                st.markdown("---")
                st.markdown("### Assessment Results")

                # Diagnostic Output
                if prediction == 1:
                    st.markdown(f'<div class="alert-card alert-malignant">Diagnosis Indication: {label.upper()}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-card alert-benign">Diagnosis Indication: {label.upper()}</div>', unsafe_allow_html=True)

                if result_kind == "success":
                    st.success(guidance)
                elif result_kind == "info":
                    st.info(guidance)
                else:
                    st.warning(guidance)

                # Accuracy Check against Dataset
                if true_label:
                    predicted_str = "Malignant" if prediction == 1 else "Benign"
                    if predicted_str == true_label:
                        st.caption(f"**Verification:** The model successfully matched the dataset ground truth ({true_label}).")
                    else:
                        st.caption(f"**Verification:** The model's prediction ({predicted_str}) diverged from the dataset ground truth ({true_label}).")

                # Probability Dashboard
                st.markdown("#### Confidence Metrics")
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Probability of Malignancy", f"{m_prob*100:.1f}%")
                    st.progress(m_prob)
                with metric_col2:
                    st.metric("Probability of Benignity", f"{b_prob*100:.1f}%")
                    st.progress(b_prob)

                with st.expander("View Raw System Output"):
                    st.json(result)

            else:
                st.error(f"System Error {resp.status_code}: {resp.json().get('error', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error(
                "**Connection Refused:** Unable to establish a connection with the prediction engine.\n\n"
                "Please verify the backend service is running."
            )
        except requests.exceptions.Timeout:
            st.error("**Timeout:** The prediction engine exceeded the maximum response time.")
        except Exception as exc:
            st.error(f"**System Exception:** {str(exc)}")

# ─────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">'
    '<strong>Notice:</strong> This application is a machine learning prototype designed for educational purposes. '
    'It does not constitute medical advice, diagnosis, or treatment.'
    '</div>',
    unsafe_allow_html=True,
)
