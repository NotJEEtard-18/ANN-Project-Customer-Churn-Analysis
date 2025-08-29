import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# ------------------------------
# Load the trained model
# ------------------------------
try:
    model = tf.keras.models.load_model("model.keras", compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------------------
# Load the encoders and scaler
# ------------------------------
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehotencoder.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scalar.pkl", "rb") as file:
    scaler = pickle.load(file)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ------------------------------
# Prepare input data
# ------------------------------
# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
# only call toarray if result is sparse
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Base input
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

# Merge geography one-hot columns
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# ------------------------------
# Scale input
# ------------------------------
try:
    input_data_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()

# ------------------------------
# Prediction
# ------------------------------
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

st.write(f"### Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.warning("🚨 The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")

