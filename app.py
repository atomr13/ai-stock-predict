import streamlit as st
import yfinance as yf 
import datetime
from tensorflow.keras.models import load_model
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained LSTM model for stock price prediction
model = load_model('stock_price_lstm_model.h5')

# Title of the Streamlit App
st.title('Stock Price Prediction App')

# Input fields for user to enter stock ticker and start date
stock = st.text_input('Enter the stock ticker:', 'AAPL') # Default is Apple (AAPL)
start_date = st.date_input('Start date', datetime.date(2015,1,1)) # Default start date

# End date is always set to the current date
end_date = datetime.datetime.now()

# Calculate the predicted date (next day)
predicted_date = end_date + datetime.timedelta(days=1)
predicted_date_str = predicted_date.strftime('%d-%m-%Y')


# Predict Stock Price on Button Click
if st.button('Predict'):
     # Download historical stock data for the given ticker and date range
    data=yf.download(stock, start=start_date, end=end_date)

    # Check if data is available
    if data.empty:
        st.write(f"No data available for {stock}. Please enter a valid ticker symbol.")
    else:

        # Normalize the 'Close' prices using MinMaxScaler to scale values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Prepare the data for the LSTM model by using the last 60 days of stock prices
        time_step = 60
        X_input = scaled_data[-time_step:].reshape(1, time_step, 1) # Create input with required shape for LSTM
    
        # Use the trained model to make a prediction
        prediction = model.predict(X_input)

         # Inverse transform the predicted value to get the actual stock price
        prediction = scaler.inverse_transform(prediction)


        # Display the predicted stock price for the next day
        st.write(f'Predicted Closing Price {predicted_date_str}: ${prediction[0][0]:.2f}')

