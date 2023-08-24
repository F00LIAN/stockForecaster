import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU
from datetime import datetime
import logging

# Fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

# Prepare the data for training and testing, 80% train and 20% test, normalize the data
def prepare_data(data, feature='Close'):
    closing_prices = data[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    closing_prices_scaled = scaler.fit_transform(closing_prices)
    
    train_size = int(0.8 * len(closing_prices_scaled))
    train_data = closing_prices_scaled[:train_size]
    test_data = closing_prices_scaled[train_size:]
    
    return train_data, test_data, scaler

# Function to create sequences
def create_sequences(data, sequence_length):
    """
    Create sequences from time series data for RNN training.
    
    Parameters:
        - data (np.ndarray): Time series data.
        - sequence_length (int): The length of each sequence.
        
    Returns:
        - np.ndarray: Sequences of data.
        - np.ndarray: Corresponding target values.
    """
    try: 
        logging.info(f"Creating sequences of length {sequence_length}.")
        sequences, targets = [], []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i+sequence_length])
            targets.append(data[i+sequence_length])
        return np.array(sequences), np.array(targets)
    except Exception as e:
        logging.error(f"Error creating sequences: {e}")
        return None, None
    

# Function to build a generalized RNN model
def build_rnn_model(input_shape, rnn_type='LSTM', units=50):
    model = Sequential()
    if rnn_type == 'LSTM':
        model.add(LSTM(units, activation='relu', input_shape=input_shape))
    elif rnn_type == 'GRU':
        model.add(GRU(units, activation='relu', input_shape=input_shape))
    elif rnn_type == 'RNN':
        model.add(SimpleRNN(units, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
    return model

# Function to forecast future prices
def forecast_future(model, last_sequence, future_steps, scaler):
    forecasted_prices = []
    curr_sequence = last_sequence.copy()
    
    for _ in range(future_steps):
        prediction = model.predict(curr_sequence.reshape(1, -1, 1))
        forecasted_prices.append(prediction[0][0])
        curr_sequence = np.append(curr_sequence[1:], prediction)
        
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))
    return forecasted_prices

# Function to plot results with dates
def plot_results(true_data, predicted_data, future_data, dates, future_dates):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, true_data, label='True Prices')
    ax.plot(dates, predicted_data, label='Predicted Prices')
    ax.plot(future_dates, future_data, label='Forecasted Prices', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Prediction and Forecast')
    ax.legend()
    return fig

# Streamlit app
st.title('Stock Price Prediction and Forecasting')

# Sidebar for user input
st.sidebar.header('User Input and Hyperparameter Tuning')
STOCK_SYMBOL = st.sidebar.text_input('Enter stock symbol:', 'AAPL')
RNN_TYPE = st.sidebar.selectbox('Select RNN Type:', ['LSTM', 'GRU', 'RNN'])
EPOCHS = st.sidebar.slider('Number of Epochs:', min_value=1, max_value=200, value=50)
BATCH_SIZE = st.sidebar.slider('Batch Size:', min_value=1, max_value=100, value=32)
SEQUENCE_LENGTH = 30 # st.sidebar.slider('Sequence Length:', min_value=1, max_value=365, value=30)
RNN_UNITS = st.sidebar.slider('Number of RNN Units:', min_value=1, max_value=100, value=50)
FUTURE_STEPS = st.sidebar.slider('Number of Business Days to Predict:', min_value=1, max_value=50, value=10)

# Buttons for time frame
st.sidebar.header('Choose a Time Frame')

if st.sidebar.button('1 Year'):
    END_DATE = datetime.now()
    START_DATE = END_DATE - pd.DateOffset(years=1)
elif st.sidebar.button('2 Years'):
    END_DATE = datetime.now()
    START_DATE = END_DATE - pd.DateOffset(years=2)
elif st.sidebar.button('3 Years'):
    END_DATE = datetime.now()
    START_DATE = END_DATE - pd.DateOffset(years=3)
elif st.sidebar.button('4 Years'):
    END_DATE = datetime.now()
    START_DATE = END_DATE - pd.DateOffset(years=4)
elif st.sidebar.button('5 Years'):
    END_DATE = datetime.now()
    START_DATE = END_DATE - pd.DateOffset(years=5)




# If any of the time frame buttons is clicked
if 'START_DATE' in locals() and 'END_DATE' in locals():

    # Convert Pandas Timestamps to string format for yfinance
    start_date_str = START_DATE.strftime('%Y-%m-%d')
    end_date_str = END_DATE.strftime('%Y-%m-%d')

    # Fetch and display stock data
    st.header(f'Displaying stock data for {STOCK_SYMBOL}')
    stock_data = fetch_stock_data(STOCK_SYMBOL, start_date_str, end_date_str)

    
    # Filter the stock data to only include the user-specified date range
    stock_data = stock_data[(stock_data.index >= pd.Timestamp(START_DATE)) & (stock_data.index <= pd.Timestamp(END_DATE))]

    # Prepare data
    train_data, test_data, scaler = prepare_data(stock_data)

    # Create sequences
    X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)

    # Build and train model
    st.header('Training Model...')
    model = build_rnn_model((SEQUENCE_LENGTH, 1), rnn_type=RNN_TYPE, units=RNN_UNITS)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Make predictions
    st.header('Making Predictions...')
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Prepare true prices for comparison
    true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Create a list of date indices for plotting
    train_size = int(0.8 * len(stock_data))
    plot_dates = stock_data.index[train_size + SEQUENCE_LENGTH:]
    min_length = min(len(plot_dates), len(true_prices), len(predictions))
    
    # Forecast future prices
    st.header('Forecasting Future Prices...')
    last_sequence = test_data[-SEQUENCE_LENGTH:]
    forecasted_prices = forecast_future(model, last_sequence, FUTURE_STEPS, scaler)

    # Prepare future dates for plotting
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(last_date, periods=FUTURE_STEPS+1, freq='B')[1:]
 
    # Plot results along with future forecast
    st.header('Prediction and Future Forecast Results')
    fig = plot_results(true_prices[:min_length], predictions[:min_length], forecasted_prices, plot_dates[:min_length], future_dates)
    st.pyplot(fig)