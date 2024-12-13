import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal

def forecasting_cases_page():
    st.title("COVID-19 Daily Cases Prediction :mask:")

    st.markdown("""
    ### To predict the total daily cases from COVID-19.
    Please provide the no of days for which results needs to be fetched:
    """)
    st.write("**Note:** The last known values as on 31st August 2023.")
    # Load the model with custom objects
    custom_objects = {
        'mse': MeanSquaredError(),
        'Orthogonal': Orthogonal,
    }


    # Load the model
    try:
        model = load_model('lstm_model.h5', custom_objects=custom_objects)
    except ValueError as e:
        print(f"Error loading model: {e}")


    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Load the last sequence
    last_sequence = np.load('last_sequence.npy')

    # Input for forecast steps
    SEQ_LENGTH = 14  # Length of input sequences for LSTM
    forecast_steps = st.number_input('Enter the No of days', min_value=1, max_value=1000, value=14)
    button = st.button(label='Predict')
    forecast = []

    if button:
        for _ in range(forecast_steps):
            # Reshape last_sequence for prediction
            prediction = model.predict(last_sequence.reshape(1, SEQ_LENGTH, -1))
            forecast.append(prediction[0, 0])
            
            # Update last_sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, 0] = prediction[0, 0]  # Update with predicted value

        # Prepare for inverse transformation
        forecast_array = np.array(forecast).reshape(-1, 1)  # Reshape forecast to be a column vector
        
        # Create an array with zeros for other features (adjust based on your data)
        num_features = scaler.scale_.shape[0]  # Get number of features from scaler
        additional_features = np.zeros((forecast_steps, num_features - 1))  # Adjust based on your data

        # Concatenate forecast with additional zeros for other features
        combined_forecast = np.concatenate((forecast_array, additional_features), axis=1)

        # Perform inverse transformation
        forecast_rescaled = combined_forecast[:, 0]

        # Create a DataFrame for displaying results
        forecast_df = pd.DataFrame(forecast_rescaled, columns=["Forecasted Values"])
        forecast_df.index.name = "Day"
        
        # Round and convert to integer values for display if necessary
        forecast_df["Forecasted Values"] = forecast_df["Forecasted Values"].apply(lambda x: int(round(x)))

        # Display the forecast results in a table
        st.table(forecast_df)
