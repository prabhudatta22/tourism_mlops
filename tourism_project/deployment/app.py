"""Streamlit front-end for the Wellness Tourism Package purchase model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "best_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)        # model committed to the repo by the CI pipeline


model = load_model()

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package — Purchase Predictor")
st.write("Enter the customer details to estimate whether they will buy the **Wellness Tourism Package** "
         "before the sales team contacts them.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Customer")
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly income", 1000, 1_000_000, 22000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Has passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
with c2:
    st.subheader("Trip & interaction")
    persons = st.number_input("Number of persons visiting", 1, 10, 3)
    children = st.number_input("Children (< 5 yrs) visiting", 0, 5, 1)
    trips = st.number_input("Average trips per year", 0, 30, 3)
    star = st.selectbox("Preferred property star", [3, 4, 5])
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of pitch (min)", 1, 150, 15)
    followups = st.number_input("Number of follow-ups", 0, 10, 4)
    pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)

# Collect inputs into a DataFrame with the exact training schema
input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": contact, "CityTier": city_tier, "DurationOfPitch": duration,
    "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups, "ProductPitched": product, "PreferredPropertyStar": star,
    "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": int(passport == "Yes"),
    "PitchSatisfactionScore": pitch_score, "OwnCar": int(own_car == "Yes"),
    "NumberOfChildrenVisiting": children, "Designation": designation, "MonthlyIncome": income,
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df)

if st.button("Predict", type="primary"):
    proba = float(model.predict_proba(input_df)[0, 1])
    if proba >= 0.5:
        st.success(f"✅ Likely to BUY the package — probability {proba:.1%}. Prioritise this customer.")
    else:
        st.warning(f"❌ Unlikely to buy — probability {proba:.1%}. Lower priority for outreach.")
