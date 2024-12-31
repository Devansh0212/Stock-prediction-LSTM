import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
MODEL_PATH = '/Users/devanshbrahmbhatt/Desktop/Stock_ML/Stock_ML/Stock Predictions Model.keras'
DEFAULT_STOCK = 'GOOG'
DEFAULT_START_DATE = pd.to_datetime('2012-01-01')
DEFAULT_END_DATE = pd.to_datetime('2022-12-31')

# Load LSTM model
@st.cache_resource
def load_lstm_model(path):
    return load_model(path)

# Fetch stock data
@st.cache
def fetch_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

# Calculate moving averages
def calculate_moving_averages(data):
    return {
        'EMA50': data.Close.ewm(span=50, adjust=False).mean(),
        'EMA100': data.Close.ewm(span=100, adjust=False).mean(),
        'EMA200': data.Close.ewm(span=200, adjust=False).mean(),
    }

# Plot moving averages
def plot_moving_averages(data, moving_averages, selected_mas, stock_symbol):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.Close, label='Stock Price', color='black')
    for ma in selected_mas:
        ax.plot(moving_averages[ma], label=f'{ma}')
    ax.set_title(f'{stock_symbol} Stock Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True)
    st.pyplot(fig)

# Prepare data for prediction
def prepare_data(data, model):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])
    x, y = [], []
    for i in range(100, len(data_scaled)):
        x.append(data_scaled[i-100:i])
        y.append(data_scaled[i, 0])
    x, y = np.array(x), np.array(y)
    predictions = model.predict(x)
    scale_factor = 1 / scaler.scale_[0]
    predictions = predictions * scale_factor
    y = y * scale_factor
    return y, predictions

# Plot predictions
def plot_predictions(y, predictions, stock_symbol):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(y, label='Actual Prices', color='green')
    ax.plot(predictions, label='Predicted Prices', color='red')
    ax.set_title(f'{stock_symbol} Predicted vs Actual Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Calculate indicators
def calculate_indicators(data):
    delta = data.Close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    ema_12 = data.Close.ewm(span=12, adjust=False).mean()
    ema_26 = data.Close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return rsi, macd, signal_line

# Plot indicators
def plot_indicators(rsi, macd, signal_line):
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    # RSI
    ax[0].plot(rsi, label='RSI', color='purple')
    ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
    ax[0].axhline(30, color='green', linestyle='--', label='Oversold')
    ax[0].set_title('Relative Strength Index (RSI)')
    ax[0].legend()
    # MACD
    ax[1].plot(macd, label='MACD', color='blue')
    ax[1].plot(signal_line, label='Signal Line', color='orange')
    ax[1].set_title('MACD & Signal Line')
    ax[1].legend()
    st.pyplot(fig)

# Main App
def main():
    model = load_lstm_model(MODEL_PATH)

    # Sidebar Inputs
    st.sidebar.header('User Input')
    stock_symbol = st.sidebar.text_input('Enter Stock Symbol', DEFAULT_STOCK)
    start_date = st.sidebar.date_input('Select Start Date', DEFAULT_START_DATE)
    end_date = st.sidebar.date_input('Select End Date', DEFAULT_END_DATE)

    # Check date validity
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
        return

    if st.sidebar.button('Fetch Stock Data'):
        with st.spinner('Fetching stock data...'):
            data = fetch_stock_data(stock_symbol, start_date, end_date)

        if data.empty:
            st.error(f"No data found for {stock_symbol} between {start_date} and {end_date}")
            return

        st.success('Data fetched successfully!')
        st.title(f"Stock Market Predictor for {stock_symbol}")
        st.subheader(f"Stock data from {start_date} to {end_date}")

        # Tabs
        tabs = st.tabs(["📈 Stock Prices", "🔮 Predictions", "📊 Indicators", "📄 Raw Data"])

        # Stock Prices Tab
        with tabs[0]:
            st.header("Stock Prices and Moving Averages")
            ma_options = ['EMA50', 'EMA100', 'EMA200']
            selected_mas = st.multiselect('Select Moving Averages to Plot', ma_options, default=ma_options)
            moving_averages = calculate_moving_averages(data)
            plot_moving_averages(data, moving_averages, selected_mas, stock_symbol)

        # Predictions Tab
        with tabs[1]:
            st.header("Predicted vs Actual Prices")
            y, predictions = prepare_data(data, model)
            plot_predictions(y, predictions, stock_symbol)
            st.write(f"MAE: {mean_absolute_error(y, predictions):.2f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y, predictions)):.2f}")

        # Indicators Tab
        with tabs[2]:
            st.header("Technical Indicators (RSI & MACD)")
            rsi, macd, signal_line = calculate_indicators(data)
            plot_indicators(rsi, macd, signal_line)

        # Raw Data Tab
        with tabs[3]:
            st.header("Raw Stock Data")
            st.write(data)
            csv_data = data.to_csv().encode('utf-8')
            st.download_button('Download CSV', csv_data, file_name=f'{stock_symbol}_stock_data.csv', mime='text/csv')

if __name__ == "__main__":
    main()