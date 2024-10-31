from tensorflow.keras.models import load_model
from data_preprocces import X_test, data, scaler, training_data_len, test_data
import matplotlib.pyplot as plt

# Load the pre-trained LSTM model for stock price prediction
model = load_model('stock_price_lstm_model.h5')

# Predict stock prices using the test data
predictions = model.predict(X_test)

# Convert the scaled predictions back to the original scale (stock prices)
predictions = scaler.inverse_transform(predictions)

# Set the figure size for the plot
plt.figure(figsize=(12,6))

# Plot the actual stock prices in blue
plt.plot(data.index[training_data_len + 60:], scaler.inverse_transform(test_data[60:]), color='blue', label='Actual Prices')

# Plot the predicted stock prices in red
plt.plot(data.index[training_data_len + 60:], predictions, color='red', label='Predicted Prices')

# Labeling the plot
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Actual vs Predicted Prices')
plt.legend()

# Display the plot
plt.show()
