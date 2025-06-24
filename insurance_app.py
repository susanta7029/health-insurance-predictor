import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load your trained model and scaler (we’ll train here in one go)
def train_model():
    df = pd.read_csv('insurance.csv')
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    return rf, scaler, X.columns

# Train model
model, scaler, feature_names = train_model()

# Streamlit UI
st.title("💰 Health Insurance Cost Predictor")
st.write("Enter the details below to predict insurance charges.")

# User inputs
age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# One-hot encoding for categorical fields
input_dict = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_male': 1 if sex == 'male' else 0,
    'smoker_yes': 1 if smoker == 'yes' else 0,
    'region_northwest': 1 if region == 'northwest' else 0,
    'region_southeast': 1 if region == 'southeast' else 0,
    'region_southwest': 1 if region == 'southwest' else 0,
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Charges 💡"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Insurance Charges: ₹{round(prediction, 2)}")
