from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging

# Fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start, end):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start, end=end)
    return df

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
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)

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
