# Stock Price Prediction App

This project is a stock price prediction application that uses a Long Short-Term Memory (LSTM) model to predict future stock prices based on historical data. The model is built with TensorFlow/Keras and is integrated into a user-friendly web interface using Streamlit.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview
This project allows users to input a stock ticker symbol and get a prediction for the stock's next closing price. The model is trained using historical stock prices downloaded from Yahoo Finance and predicts the stock price using time series forecasting with LSTM neural networks.

## Features
- User-friendly interface built with Streamlit to interact with the model.
- Dynamic input for stock tickers and date ranges.
- Visualization of actual vs. predicted stock prices.
- Pre-trained LSTM model that can be reused to predict different stocks.

## Installation
### Prerequisites
- Python 3.8 or higher
- `pip` (Python package installer)
- Virtual environment (recommended for isolating dependencies)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Activate the virtual environment
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Launch the application using the `streamlit run app.py` command.
2. Enter a valid stock ticker symbol (e.g., `AAPL`, `GOOG`, `MSFT`) and choose the start date for historical data collection.
3. Click the "Predict" button to see the predicted closing price for the next trading day.
4. The app will display the prediction and also provide a visualization comparing actual and predicted prices.

## Project Structure
```
project_root/
    ├── app.py                     # Streamlit application for user interaction
    ├── data_preprocess.py         # Script for data collection and preprocessing
    ├── model_training.py          # Script for defining, training, and saving the LSTM model
    ├── model_predict.py           # Script for loading the model and making predictions
    ├── requirements.txt           # List of required dependencies
    ├── stock_price_lstm_model.h5  # Saved LSTM model
    ├── README.md                  # Project documentation
    └── data/                      # Folder for storing processed data (optional)
```

## Technical Details
- **Data Source**: Historical stock prices are fetched using the Yahoo Finance API (`yfinance` library).
- **Model**: The model is an LSTM neural network built using TensorFlow/Keras.
  - **Layers**: The model consists of two LSTM layers with `50` units each and dropout for regularization, followed by a dense layer to output the predicted value.
  - **Optimizer**: Adam optimizer is used to train the model, with Mean Squared Error (MSE) as the loss function.
- **Interface**: The web app is built with Streamlit, allowing for easy interaction.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Disclaimer

The predictions provided by this application are for informational purposes only and should not be considered as financial advice. Stock market investments are inherently risky, and past performance does not guarantee future results. Always do your own research or consult with a licensed financial advisor before making any investment decisions.