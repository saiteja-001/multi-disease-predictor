import streamlit as st
import pandas as pd
import pickle

# Step A: Load your trained model (put your .pkl model file in the same folder)
with open('kidney_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Kidney Disease Prediction")

# Step B: Place the Step 2 code (input widgets) here
# Copy-paste the whole input widget code here

id = st.number_input("ID", min_value=0, max_value=1000, value=1)
age = st.number_input("Age", min_value=0, max_value=120, value=45)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.05, value=1.02)
al = st.number_input("Albumin", min_value=0, max_value=5, value=0)
su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
rbc = st.selectbox("Red Blood Cells (0=normal,1=abnormal)", options=[0, 1], index=0)
pc = st.selectbox("Pus Cell (0=normal,1=abnormal)", options=[0, 1], index=0)
pcc = st.selectbox("Pus Cell Clumps (0=no,1=yes)", options=[0, 1], index=0)
ba = st.selectbox("Bacteria (0=no,1=yes)", options=[0, 1], index=0)
bgr = st.number_input("Blood Glucose Random", min_value=0, max_value=1000, value=100)
bu = st.number_input("Blood Urea", min_value=0, max_value=500, value=30)
sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=50.0, value=1.0)
sod = st.number_input("Sodium", min_value=0, max_value=200, value=140)
pot = st.number_input("Potassium", min_value=0.0, max_value=20.0, value=4.5)
hemo = st.number_input("Hemoglobin", min_value=0.0, max_value=30.0, value=15.0)
pcv = st.number_input("Packed Cell Volume", min_value=0, max_value=60, value=40)
wc = st.number_input("White Blood Cell Count", min_value=0, max_value=50000, value=8000)
rc = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, value=4.5)
htn = st.selectbox("Hypertension (0=no,1=yes)", options=[0, 1], index=0)
dm = st.selectbox("Diabetes Mellitus (0=no,1=yes)", options=[0, 1], index=0)
cad = st.selectbox("Coronary Artery Disease (0=no,1=yes)", options=[0, 1], index=0)
appet = st.selectbox("Appetite (0=poor,1=good)", options=[0, 1], index=1)
pe = st.selectbox("Pedal Edema (0=no,1=yes)", options=[0, 1], index=0)
ane = st.selectbox("Anemia (0=no,1=yes)", options=[0, 1], index=0)

input_dict = {
    'id': id,
    'age': age,
    'bp': bp,
    'sg': sg,
    'al': al,
    'su': su,
    'rbc': rbc,
    'pc': pc,
    'pcc': pcc,
    'ba': ba,
    'bgr': bgr,
    'bu': bu,
    'sc': sc,
    'sod': sod,
    'pot': pot,
    'hemo': hemo,
    'pcv': pcv,
    'wc': wc,
    'rc': rc,
    'htn': htn,
    'dm': dm,
    'cad': cad,
    'appet': appet,
    'pe': pe,
    'ane': ane
}

input_df = pd.DataFrame([input_dict])

# Step C: Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("Kidney Disease detected!")
    else:
        st.info("No Kidney Disease detected.")
