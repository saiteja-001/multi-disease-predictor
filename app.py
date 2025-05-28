import streamlit as st
import pandas as pd
import pickle

# Map disease names to their model files
MODEL_PATHS = {
    "Kidney Disease": "kidney_model.pkl",
    "Diabetes": "diabetes_model.pkl",
    "Heart Disease": "heart_model.pkl",
    "Liver Disease": "liver_model.pkl"
}

# Common input features expected by your models
FEATURES = [
    ("age", "Age"),
    ("bp", "Blood Pressure"),
    ("bgr", "Blood Glucose Random"),
    ("bu", "Blood Urea"),
    ("sc", "Serum Creatinine"),
    ("hemo", "Hemoglobin"),
    ("pcv", "Packed Cell Volume"),
    ("rc", "Red Blood Cell Count"),
    ("wc", "White Blood Cell Count")
]

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

st.title("Multi-Disease Prediction App")

# Disease selector
disease = st.selectbox("Select Disease to Predict", list(MODEL_PATHS.keys()))

# Load corresponding model
model = load_model(MODEL_PATHS[disease])

# Sidebar inputs
st.sidebar.header(f"Input Features for {disease}")
user_inputs = {}
for feature_key, feature_label in FEATURES:
    val = st.sidebar.number_input(feature_label, min_value=0.0, step=0.1)
    user_inputs[feature_key] = val

# Prepare input dataframe for prediction
input_df = pd.DataFrame([user_inputs])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    # Probability prediction if supported
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
    else:
        proba = None

    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.success(f"{disease} detected!")
    else:
        st.info(f"No {disease} detected.")

    if proba is not None:
        st.write(f"Prediction Confidence: {proba:.2f}")