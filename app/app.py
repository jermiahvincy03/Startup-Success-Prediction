import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Startup Success Predictor", page_icon="🚀", layout="centered")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    category_freq = joblib.load(os.path.join(MODEL_DIR, "category_freq_map.pkl"))
    state_freq = joblib.load(os.path.join(MODEL_DIR, "state_freq_map.pkl"))
    return model, scaler, feature_columns, category_freq, state_freq


model, scaler, feature_columns, category_freq, state_freq = load_artifacts()

CATEGORY_FLAG_MAP = {
    "software": "is_software", "web": "is_web", "mobile": "is_mobile",
    "enterprise": "is_enterprise", "advertising": "is_advertising",
    "games_video": "is_gamesvideo", "ecommerce": "is_ecommerce",
    "biotech": "is_biotech", "consulting": "is_consulting",
}
STATE_FLAG_MAP = {
    "CA": "is_CA", "NY": "is_NY", "MA": "is_MA", "TX": "is_TX",
}

st.title("🚀 Startup Success Predictor")
st.caption("Predicts whether a startup is likely to be **acquired** or **closed**, based on funding, milestones, and network features. Trained on Crunchbase-style historical startup data.")

st.divider()

with st.form("prediction_form"):
    st.subheader("Company Profile")
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Industry Category", sorted(category_freq.keys()), index=sorted(category_freq.keys()).index("software"))
        state = st.selectbox("State", sorted(state_freq.keys()), index=sorted(state_freq.keys()).index("CA"))
        is_top500 = st.checkbox("Ranked in Startup Top 500?", value=False)

    with col2:
        age_first_funding_year = st.number_input("Age at First Funding (years)", min_value=0.0, max_value=30.0, value=1.5, step=0.1)
        age_last_funding_year = st.number_input("Age at Last Funding (years)", min_value=0.0, max_value=30.0, value=3.0, step=0.1)
        funding_duration_days = st.number_input("Funding Duration (days, first to last round)", min_value=0, max_value=5000, value=365, step=10)

    st.subheader("Traction & Network")
    col3, col4 = st.columns(2)
    with col3:
        relationships = st.number_input("Number of Relationships (investors/partners)", min_value=0, max_value=100, value=5)
        milestones = st.number_input("Number of Milestones Achieved", min_value=0, max_value=20, value=2)
        avg_participants = st.number_input("Avg. Participants per Funding Round", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    with col4:
        age_first_milestone_year = st.number_input("Age at First Milestone (years)", min_value=0.0, max_value=30.0, value=2.0, step=0.1)
        age_last_milestone_year = st.number_input("Age at Last Milestone (years)", min_value=0.0, max_value=30.0, value=4.0, step=0.1)

    st.subheader("Funding Details")
    col5, col6 = st.columns(2)
    with col5:
        funding_total_usd = st.number_input("Total Funding Raised (USD)", min_value=0, max_value=1_000_000_000, value=2_000_000, step=100_000)
        funding_rounds = st.number_input("Number of Funding Rounds", min_value=0, max_value=15, value=2)
    with col6:
        has_VC = st.checkbox("Has VC Funding?", value=True)
        has_angel = st.checkbox("Has Angel Funding?", value=False)
        has_roundA = st.checkbox("Has Round A?", value=True)
        has_roundB = st.checkbox("Has Round B?", value=False)
        has_roundC = st.checkbox("Has Round C?", value=False)
        has_roundD = st.checkbox("Has Round D?", value=False)

    submitted = st.form_submit_button("Predict Outcome", use_container_width=True, type="primary")

if submitted:
    row = {col: 0 for col in feature_columns}

    row["age_first_funding_year"] = age_first_funding_year
    row["age_last_funding_year"] = age_last_funding_year
    row["age_first_milestone_year"] = age_first_milestone_year
    row["age_last_milestone_year"] = age_last_milestone_year
    row["relationships"] = relationships
    row["funding_rounds"] = funding_rounds
    row["funding_total_usd"] = funding_total_usd
    row["milestones"] = milestones
    row["avg_participants"] = avg_participants
    row["is_top500"] = int(is_top500)
    row["has_VC"] = int(has_VC)
    row["has_angel"] = int(has_angel)
    row["has_roundA"] = int(has_roundA)
    row["has_roundB"] = int(has_roundB)
    row["has_roundC"] = int(has_roundC)
    row["has_roundD"] = int(has_roundD)
    row["funding_duration_days"] = funding_duration_days

    if state in STATE_FLAG_MAP:
        row[STATE_FLAG_MAP[state]] = 1
    else:
        row["is_otherstate"] = 1

    if category in CATEGORY_FLAG_MAP:
        row[CATEGORY_FLAG_MAP[category]] = 1
    else:
        row["is_othercategory"] = 1

    round_flags_sum = has_VC + has_angel + has_roundA + has_roundB + has_roundC + has_roundD
    row["total_round_types"] = round_flags_sum
    row["funding_per_relationship"] = funding_total_usd / (relationships + 1)
    row["milestone_velocity"] = milestones / (age_last_milestone_year + 1)
    row["funding_total_usd_log"] = np.log1p(funding_total_usd)
    row["category_code_freq"] = category_freq.get(category, np.mean(list(category_freq.values())))
    row["state_code_freq"] = state_freq.get(state, np.mean(list(state_freq.values())))

    X_input = pd.DataFrame([row])[feature_columns]
    X_scaled = scaler.transform(X_input)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.success(f"✅ **Likely Outcome: Acquired**")
    else:
        st.error(f"⚠️ **Likely Outcome: Closed**")

    col_a, col_b = st.columns(2)
    col_a.metric("Probability of Acquisition", f"{proba[1]*100:.1f}%")
    col_b.metric("Probability of Closure", f"{proba[0]*100:.1f}%")

    st.progress(float(proba[1]))
    st.caption("This prediction is based on historical patterns and should be used as a directional signal, not a guarantee.")

st.divider()
st.caption("Built with scikit-learn, XGBoost & Streamlit · [GitHub](https://github.com/jermiahvincy03/Startup-Success-Prediction)")
