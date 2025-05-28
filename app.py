import streamlit as st
import pandas as pd
import pickle

# Helper function to load models
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Define the features for each disease model
disease_features = {
    'Kidney Disease': ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                       'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                       'htn', 'dm', 'cad', 'appet', 'pe', 'ane'],
    
    'Diabetes': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    
    'Heart Disease': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    
    'Liver Disease': ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                      'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                      'Albumin', 'Albumin_and_Globulin_Ratio']
}

# Load all models upfront (make sure pkl files are in same folder)
models = {
    'Kidney Disease': load_model('kidney_model.pkl'),
    'Diabetes': load_model('diabetes_model.pkl'),
    'Heart Disease': load_model('heart_model.pkl'),
    'Liver Disease': load_model('liver_model.pkl')
}

st.title("Multiple Disease Prediction App")

# Step 1: Select disease
disease = st.selectbox("Select Disease to Predict", list(disease_features.keys()))

# Step 2: Collect inputs dynamically based on selected disease features
input_data = {}
for feature in disease_features[disease]:
    # For demo, make everything a number input; you can customize per feature type
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input dict to dataframe for model
input_df = pd.DataFrame([input_data])

# Step 3: Predict on button click
if st.button("Predict"):
    model = models[disease]
    try:
        prediction = model.predict(input_df)[0]
        # For binary classification models with 0/1 outputs
        if prediction == 1:
            st.error(f"{disease} Detected!")
        else:
            st.success(f"No {disease} Detected!")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
