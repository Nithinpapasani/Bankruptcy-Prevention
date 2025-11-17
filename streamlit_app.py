import streamlit as st

# PAGE CONFIG MUST COME FIRST
st.set_page_config(
    page_title="Bankruptcy Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

import joblib
import pandas as pd
import logging
from datetime import datetime
import os

# Base directory (safe for all environments)
BASE_DIR = os.getcwd()


# Logs directory
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =============================================================================
# LOAD MODEL + FEATURES + METADATA
# =============================================================================

@st.cache_resource
def load_model():
    """Load ML model, feature list, and metadata safely."""

    try:
        # Try files in order: v3 → v2 → v1
        model_paths = [
            os.path.join(BASE_DIR, "model", "bankruptcy_model_v3.pkl"),
            os.path.join(BASE_DIR, "model", "bankruptcy_model_v2.pkl"),
            os.path.join(BASE_DIR, "model", "bankruptcy_model.pkl"),
        ]

        model_path = next((p for p in model_paths if os.path.exists(p)), None)

        if model_path is None:
            raise FileNotFoundError("No model found in /model directory.")

        # Feature names & metadata paths
        features_path = os.path.join(BASE_DIR, "model", "feature_names.pkl")
        metadata_path = os.path.join(BASE_DIR, "model", "model_metadata_v3.pkl")

        model = joblib.load(model_path)
        features = joblib.load(features_path)

        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None

        logging.info("Model loaded successfully: %s", model_path)
        return model, features, metadata

    except Exception as e:
        logging.error("Model loading failed: %s", e)
        st.error(f"❌ Failed to load model: {e}")
        return None, None, None


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():

    # Load the model
    model, features, metadata = load_model()

    if model is None:
        st.error("❌ Could not load the ML model. Check logs.")
        return


    # Custom CSS styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3.4rem;      /* was 2.7rem */
            font-weight: 800;
            color: #1f77b4;
            text-align: center;
        }
        .sub-header {
            font-size: 1.6rem;      /* was 1.2rem */
            color: #555;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        /* Optional: make all H2 headings a bit larger */
        h2 {
            font-size: 2rem !important;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(to right, #ff4b1f, #ff9068);
        }
    </style>
    """, unsafe_allow_html=True)

    # Headers
    st.markdown('<p class="main-header">🏦 Bankruptcy Risk Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Financial Risk Analytics</p>', unsafe_allow_html=True)
    st.markdown("---")

    # =============================================================================
    # SIDEBAR — USER INPUTS
    # =============================================================================

    st.sidebar.header("📊 Configure Risk Factors")

    # 6 business risk factors (must match training features)
    feature_info = {
        "industrial_risk": "Industry volatility & economic instability",
        "management_risk": "Leadership quality & decision-making ability",
        "financial_flexibility": "Liquidity, access to capital & debt capacity",
        "credibility": "Reputation, credit history & trustworthiness",
        "competitiveness": "Market position & brand strength",
        "operating_risk": "Operational efficiency & internal processes"
    }

    risk_options = {
        "🟢 Low (0)": 0.0,
        "🟡 Medium (0.5)": 0.5,
        "🔴 High (1)": 1.0
    }

    inputs = {}

    for feature, description in feature_info.items():
        with st.sidebar.expander(f"ℹ️ {feature.replace('_', ' ').title()}"):
            st.caption(description)
        choice = st.sidebar.selectbox(
            feature.replace('_', ' ').title(),
            list(risk_options.keys()),
            index=0
        )
        inputs[feature] = risk_options[choice]

    st.sidebar.markdown("---")

    # Quick presets
    st.sidebar.subheader("🧪 Quick Test")
    if st.sidebar.button("High Risk Scenario"):
        for k in inputs:
            inputs[k] = 1.0
    if st.sidebar.button("Low Risk Scenario"):
        for k in inputs:
            inputs[k] = 0.0

    st.sidebar.markdown("---")

    predict_button = st.sidebar.button("🔍 Analyze Bankruptcy Risk", type="primary")

    # =============================================================================
    # WELCOME SCREEN BEFORE PREDICTION
    # =============================================================================

    if not predict_button:
        st.info("👈 Select values on the sidebar and click **Analyze Bankruptcy Risk**")

        st.markdown("### 📈 Model Summary")

        colA, colB, colC, colD = st.columns(4)

        if metadata:
            perf = metadata.get("performance", {})

            mean_f1 = perf.get("mean_f1_score", 0)
            mean_auc = perf.get("mean_roc_auc", 0)
            std_auc = perf.get("std_roc_auc", 0)

            model_name = metadata.get("model_name", "Best Model")
            version = metadata.get("model_version", "v3.1")
        else:
            mean_f1 = 0.70
            mean_auc = 0.74
            std_auc = 0.10
            model_name = "ML Model"
            version = "v3.1"

        colA.metric("Mean F1-Score", f"{mean_f1:.1%}")
        colB.metric("Mean ROC-AUC", f"{mean_auc:.1%}", delta=f"± {std_auc:.1%}")
        colC.metric("Selected Model", f"{model_name} {version}")
        colD.metric("Features", "6 Total")

        return  # Stop here until user runs prediction

    # =============================================================================
    # RUN PREDICTION
    # =============================================================================

    try:
        # Input validation
        if not all(v in [0.0, 0.5, 1.0] for v in inputs.values()):
            st.warning("⚠️ Invalid input values.")
            return

        input_df = pd.DataFrame([inputs])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # bankruptcy probability

        logging.info("Prediction made: %s -> %s", inputs, prediction)

        # =============================================================================
        # RESULTS PANEL
        # =============================================================================

        st.markdown("## 📊 Results & Insights")
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        # Prediction label
        with col1:
            label = "Bankruptcy" if prediction == 1 else "Non-Bankruptcy"
            icon = "🔴" if prediction == 1 else "🟢"
            st.metric("Prediction", f"{icon} {label}")

        with col2:
            st.metric("Bankruptcy Probability", f"{probability*100:.1f}%")

        with col3:
            if probability > 0.7:
                risk = ("High Risk", "🔴")
            elif probability > 0.3:
                risk = ("Medium Risk", "🟡")
            else:
                risk = ("Low Risk", "🟢")
            st.metric("Risk Level", f"{risk[1]} {risk[0]}")

        with col4:
            confidence = max(probability, 1 - probability)
            st.metric("Model Confidence", f"{confidence*100:.1f}%")

        st.markdown("---")

        # =============================================================================
        # RECOMMENDATIONS
        # =============================================================================

        if prediction == 1:
            st.error("### 🚨 High Bankruptcy Risk — Immediate Action Required")
            st.markdown("""
            **Recommended Actions:**
            - Reduce operational overhead
            - Improve cash flow liquidity
            - Conduct financial stability audit
            - Strengthen management controls
            - Reevaluate debt structure and risk exposure
            """)
        else:
            st.success("### ✅ Stable — Low Bankruptcy Risk")
            st.markdown("""
            **Recommendations:**
            - Maintain current strategies
            - Monitor financial KPIs regularly
            - Strengthen competitive positioning
            - Explore sustainable growth opportunities
            """)

        st.markdown("---")

        # =============================================================================
        # PROBABILITY PLOT + RISK GAUGE
        # =============================================================================

        left, right = st.columns([2, 1])

        with left:
            st.markdown("### 📈 Bankruptcy Probability Chart")
            prob_df = pd.DataFrame({
                "Category": ["Non-Bankruptcy", "Bankruptcy"],
                "Probability": [1 - probability, probability]
            })
            st.bar_chart(prob_df.set_index("Category"))

        with right:
            st.markdown("### 🎯 Risk Gauge")
            st.progress(float(probability))
            st.caption(f"Bankruptcy Risk: {probability:.1%}")

        st.markdown("---")

        # =============================================================================
        # INPUT SUMMARY
        # =============================================================================

        st.markdown("### 📋 Risk Factor Summary")

        summary_rows = [
            {
                "Risk Factor": f.replace("_", " ").title(),
                "Level": "🔴 High" if v == 1 else "🟡 Medium" if v == 0.5 else "🟢 Low",
            }
            for f, v in inputs.items()
        ]

        st.dataframe(pd.DataFrame(summary_rows), hide_index=True)

        # =============================================================================
        # DOWNLOAD REPORT
        # =============================================================================

        report = f"""
BANKRUPTCY RISK REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Prediction: {label}
Bankruptcy Probability: {probability:.1%}
Risk Level: {risk[0]}
Confidence: {confidence*100:.1f}%

Inputs:
{chr(10).join([f"- {k}: {v}" for k, v in inputs.items()])}
        """

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"bankruptcy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        logging.error("Prediction failed: %s", e)


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()

