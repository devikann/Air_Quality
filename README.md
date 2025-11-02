# Air_Quality
This Deep Learning project uses an LSTM model to forecast PM2.5 air quality levels for a city. It involves data cleaning, scaling, and sequence generation, utilizing TensorFlow/Keras. The model's performance is evaluated using MSE and includes detailed visualization.
# ☁️ Air Quality Prediction (PM2.5 Forecasting) with LSTM


## 📝 Project Overview

This project implements a **Deep Learning** solution for **time-series forecasting of PM2.5 air pollution levels** using a **Long Short-Term Memory (LSTM)** Recurrent Neural Network.  
The model predicts next-day PM2.5 concentrations based on historical trends — enabling insights into **air quality and pollution forecasting**.

Developed as part of a course project, it demonstrates key concepts in:
- Time-series data preprocessing  
- Sequential neural network design  
- Model training and evaluation using TensorFlow/Keras  

The default configuration focuses on **Delhi, India**, one of the most polluted cities globally.

---

## ✨ Key Features

- **Deep Learning Architecture:** Built using a Sequential **LSTM** model to capture temporal dependencies in air quality data.  
- **Time-Series Preprocessing:** Implements `MinMaxScaler` and a sliding window generator (default time steps = 10).  
- **Exploratory Data Analysis (EDA):** Generates trend plots and correlation heatmaps.  
- **Model Evaluation:** Uses **Mean Squared Error (MSE)** for performance measurement.  
- **Interactive Testing:** Allows users to input a custom sequence for live PM2.5 prediction.

---

## 🗂️ Project Structure

