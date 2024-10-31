from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_preprocces import data, X_train, X_test, y_train, y_test

# Initialize the LSTM model using Sequential API
model = Sequential()

# First LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2)) # Dropout layer to prevent overfitting by setting 20% of neurons to zero

# Second LSTM layer without returning the sequence
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2)) # Another Dropout layer to prevent overfitting

# Output layer (Dense layer) with a single unit for the final prediction
model.add(Dense(units=1))

# Compile the model using the Adam optimizer and Mean Squared Error as the loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with the training dataset
# epochs: Number of times to iterate over the training data
# batch_size: Number of samples per gradient update
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained LSTM model to an .h5 file for future use
model.save('stock_price_lstm_model.h5')
