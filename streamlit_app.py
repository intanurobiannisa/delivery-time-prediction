import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model (SVM)
model = joblib.load('data/best_model.pkl')

# Streamlit UI
st.title("🛍️📦 Courier Delivery Time Prediction")
st.write(
    """Created by Intan Nur Robi Annisa – student of Data Science and Data Analyst Bootcamp at Dibimbing.  
    [LinkedIn Profile](https://www.linkedin.com/in/intannurrobiannisa)"""
)

st.subheader("How long customers have to wait? 😌😰😖")
st.write("Input your responses below to see your predicted delivery time.")
    
# Collect user input
user_input = []

Distance_km = st.number_input("Enter your delivery distance in kilometers:", min_value=0, max_value=100, value=5)
user_input.append(Distance_km)

Weather_ = st.selectbox("Select the Weather conditions during the delivery:", ["Clear", "Rainy", "Snowy", "Foggy", "Windy"])
Weather = 0 if Weather_ == "Clear" else 1 if Weather_ == "Foggy" else 2 if Weather_ == "Rainy" else 3 if Weather_ == "Snowy" else 4
user_input.append(Weather)

Traffic_Level_ = st.selectbox("Select the Traffic conditions:", ["Low", "Medium", "High"])
Traffic_Level = 0 if Traffic_Level_ == "Low" else 1 if Traffic_Level_ == "Medium" else 2
user_input.append(Traffic_Level)

Time_of_Day_ = st.selectbox("Select the Time when the delivery took place:", ["Morning", "Afternoon", "Evening", "Night"])
Time_of_Day = 0 if Time_of_Day_ == "Afternoon" else 1 if Time_of_Day_ == "Evening" else 2 if Time_of_Day_ == "Morning" else 3
user_input.append(Time_of_Day)

Vehicle_Type_ = st.selectbox("Select the Type of vehicle used for delivery:", ["Bike", "Scooter", "Car"])
Vehicle_Type = 0 if Vehicle_Type_ == "Bike" else 1 if Vehicle_Type_ == "Car" else 2
user_input.append(Vehicle_Type)

Preparation_Time_min = st.number_input("Enter the Time required to prepare the order (in minutes):", min_value=0, max_value=100, value=10)
user_input.append(Preparation_Time_min)

Courier_Experience_yrs = st.number_input("Enter your Experience as a courier (in years):", min_value=0, max_value=100, value=5)
user_input.append(Courier_Experience_yrs)

# Convert to DataFrame
input_df = pd.DataFrame([user_input], columns=['Distance_km', 'Weather', "Traffic_Level", "Time_of_Day", "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"])

# Display input summary
st.subheader("📋 Your Input Summary")
st.dataframe(input_df)

# Scale if needed
Distance_km_scaler = joblib.load('data/Distance_km_scaler.pkl')
input_df['Distance_km'] = Distance_km_scaler.transform(input_df[['Distance_km']])

Preparation_Time_min_scaler = joblib.load('data/Preparation_Time_min_scaler.pkl')
input_df['Preparation_Time_min'] = Preparation_Time_min_scaler.transform(input_df[['Preparation_Time_min']])

Courier_Experience_yrs_scaler = joblib.load('data/Courier_Experience_yrs_scaler.pkl')
input_df['Courier_Experience_yrs'] = Courier_Experience_yrs_scaler.transform(input_df[['Courier_Experience_yrs']])

# Wait for user to click before predicting
if st.button("Predict Delivery Time"):
    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")

    feature_names = ['Distance_km', 'Weather', "Traffic_Level", "Time_of_Day", "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"]
    # Get coefficients
    coefficients = model.coef_

    # Create a DataFrame for display
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    # Display in Streamlit
    st.subheader("📊 Model Coefficients")
    st.dataframe(coef_df)