import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from utils.utils import(fetch_stock_data, prepare_data, 
                        create_sequences, 
                        build_rnn_model, 
                        forecast_future, 
                        plot_results)
import logging

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

    # Plot the recent stock prices for the last year
    st.subheader(f"Recent Stock Prices for {STOCK_SYMBOL} (Last Year)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], label='True Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title(f'{STOCK_SYMBOL} Stock Price Over the Last Year')
    ax.legend()
    st.pyplot(fig)

    # Ensure the stock data index is timezone-naive
    stock_data.index = stock_data.index.tz_localize(None)

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

    # Or if you prefer as a table with 'Day x' keys
    st.subheader('Table of Forecasted Prices')
    forecasted_prices_df = pd.DataFrame(forecasted_prices, columns=["Forecasted Prices"])
    forecasted_prices_df.index = [f"Day {i+1}" for i in range(forecasted_prices_df.shape[0])]
    st.table(forecasted_prices_df)

    # Prepare future dates for plotting
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(last_date, periods=FUTURE_STEPS+1, freq='B')[1:]
 
    # Plot results along with future forecast
    st.header('Prediction and Future Forecast Results')
    fig = plot_results(true_prices[:min_length], predictions[:min_length], forecasted_prices, plot_dates[:min_length], future_dates)
    st.pyplot(fig)
