import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler



# Get today's date dynamically
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Download Yahoo Finance Apple stock prices from 2015 to today (to train)
data = yf.download('AAPL', start='2015-01-01', end=end_date)

# Uncomment the following lines if you want to preview and visualize the data
#print(data.head()) # Get first 5 rows of the data

# Plotting the 'Close' price to visualize the stock trend
#data['Close'].plot(title="Apple Stock Price History")
#plt.xlabel("Date")
#plt.ylabel("Price (USD)")
#plt.show()

# Normalize stock prices using MinMaxScaler to scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Split data into training and test sets (80% training, 20% test)
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len:]

# Prepare Data for LTSM

def create_dataset(data, time_step=60):
    '''
    Create sequences of data for LSTM input.

    Args:
    data (array-like): The scaled dataset from which to create sequences.
    time_step (int): The number of previous time steps to use as input features.

    Returns:
    tuple: Two numpy arrays, X and y. 
           X is the array of input sequences, and y is the corresponding output labels.

    '''
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])  # Select 'time_step' number of previous values as features
        y.append(data[i, 0]) # Select the current value as the target label
    return np.array(X), np.array(y)

# Create the training and test datasets for LSTM
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape the data to be compatible with LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
