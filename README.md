# 📦 Delivery Time Prediction App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://courier-delivery-time-prediction.streamlit.app)

This project is a machine learning-powered web app built with Streamlit that predicts delivery time based on key logistics features like distance, preparation time, and courier experience.


# 🚀 Features

📊 Predict delivery time using a trained regression model (ElasticNet)

📉 Scales input features using MinMaxScaler

🧠 Displays model coefficients for interpretability

🖥️ Interactive UI with real-time predictions


# 🧪 Tech Stack

Python 3.11

Streamlit

scikit-learn

pandas

joblib


# 📁 Project Structure

delivery-time-prediction/

├── data/

│   ├── train.csv

│   ├── test.csv

│   └── best_model.pkl

├── scalers/

│   ├── Distance_km_scaler.pkl

│   ├── Preparation_Time_min_scaler.pkl

│   └── Courier_Experience_yrs_scaler.pkl

├── streamlit_app.py

├── requirements.txt

└── README.md

# ⚙️ Setup Instructions

Clone the repo

bash
```
git clone https://github.com/intanurobiannisa/delivery-time-prediction.git

cd delivery-time-prediction

Install dependencies
```
bash
```
pip install -r requirements.txt

Run the app
```
bash
```
streamlit run streamlit_app.py
```

# 📌 How It Works

Users input:

  Distance: The delivery distance in kilometers.
  
  Weather: Weather conditions during the delivery, including Clear, Rainy, Snowy, Foggy, and Windy.
  
  Traffic Level: Traffic conditions categorized as Low, Medium, or High.
  
  Time of Day: The time when the delivery took place, categorized as Morning, Afternoon, Evening, or Night.
  
  Vehicle Type: Type of vehicle used for delivery, including Bike, Scooter, and Car.
  
  Preparation Time: The time required to prepare the order, measured in minutes.
  
  Courier Experience: Experience of the courier in years.

Inputs are scaled using pre-trained scalers

Model predicts delivery time in minutes

Coefficients are displayed to explain feature impact

# 📈 Model Details

Type: ElasticNet Regression

Target: Delivery Time (minutes)

Training Features:

Distance_km

Weather

Traffic_Level

Time_of_Day

Vehicle_Type

Preparation_Time_min

Courier_Experience_yrs

# 🙋‍♀️ Author

Made with ❤️ by Intan
