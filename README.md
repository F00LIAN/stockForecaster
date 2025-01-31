<p align="center">
  
![Image](https://github.com/user-attachments/assets/44e748f6-d53c-48b7-b870-6651fe1a55a0)

</p>

<h1>Stock Price Forecasting using RNNs</h1>
<p>This repository contains a Streamlit application for forecasting stock prices using different types of Recurrent Neural Networks (RNNs)—namely LSTM, GRU, and Simple RNN. The code herein was inspired by the Medium article on The Importance of Sequential Data and the broader context of Recurrent Neural Networks.</p>
<br />

<h2>Try it Now!</h2>

- ### [Forecasting Stock Outcomes Using RNN Methods Website](https://stockforecaster-hha6arjyjviqjw2kd7c9fp.streamlit.app/)

<h2>Environments and Technologies Used</h2>

- Python 3.7+
- Stream-Lit
- Visual Studio Code

<h2>Operating Systems Used </h2>

- Windows 11

<h2>List of Prerequisites</h2>

Before running this project, ensure you have:
- Tensorflow, Y-Finance, Streamlit, Pandas (Look at requirements.txt)
- Y-Finance API

<h2>Running the App</h2>

### 1. Overview
This project demonstrates how Recurrent Neural Networks (RNNs) can be utilized to predict future stock prices. Instead of treating data as independent points, RNNs incorporate sequential information, providing more robust predictions for time-series tasks like financial forecasting.

Key Points:

  - Users can select which RNN variant (LSTM, GRU, or Vanilla RNN) they want to train.
  - The app allows hyperparameter tuning, such as the sequence length, number of units, epochs, and batch size.
  - The model is built in Python using the Keras library, and deployed interactively with Streamlit.

### 2. Key RNN Concepts

**Why Sequential Data Matters**

Traditional machine learning often assumes each data point is independent. However, time-series data is inherently sequential—where the order of data points matters. This is crucial for tasks like:

  - Stock Price Forecasting

  - Language Modeling / Translation

  - Sentiment Analysis

**RNN Types and Applications**

1. Many-to-One
    - Input is sequential, output is a single value (e.g., sentiment classification from text).

2. One-to-Many
    - Input is a single point, output is a sequence (e.g., image captioning).

3. Many-to-Many
    - Input and output are both sequences (e.g., language translation).

**Backpropagation Through Time (BPTT)**

Instead of simple backpropagation (as in a standard ANN), RNNs use Backpropagation Through Time (BPTT). The error gradients propagate through each time step, allowing the network to learn from historical dependencies.

**Common RNN Challenges (Vanishing/Exploding Gradients)**

- Vanishing Gradients: Gradients shrink over many time steps, making it difficult to learn long-term dependencies.

- Exploding Gradients: Gradients become excessively large, causing unstable training.

**Popular Solutions (LSTM, GRU)**

- Long Short-Term Memory (LSTM): Introduces gating mechanisms (input, output, forget gates) to preserve long-term dependencies.

- Gated Recurrent Unit (GRU): A simplified version of LSTM that merges the forget and input gates into a single reset gate, reducing computational overhead.

### 3. Project Structure
```bash
.
├── README.md               <-- You are here
├── requirements.txt        <-- Python dependencies
├── app.py (or main.py)     <-- Primary Streamlit app script
└── ...
```

- The provided code can either live in a single script (e.g., app.py) or be modularized. The example below keeps everything in one file for clarity.

### 4. Dependencies and Installation

Make sure you have Python 3.7+ installed. Install the following libraries:
```bash
pip install streamlit numpy pandas matplotlib yfinance scikit-learn tensorflow keras
```

- Streamlit: For the interactive web app interface.
- NumPy / Pandas: Data manipulation.
- Matplotlib: Plotting utilities.
- yfinance: Fetching stock market data from Yahoo Finance.
- Scikit-learn: Scaling and transformation.
- TensorFlow / Keras: Deep learning framework for building and training the RNN.

### 5. How the Code Works
**Data Fetching**
```python
def fetch_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)
```

- User yfinance to download historical stock data for a chosen ticket symbol (e.g. AAPL).

- The user can specificy the data range via Streamlit interface.

**Data Preparation**
```python
# Prepare the data for training and testing, 80% train and 20% test, normalize the data
def prepare_data(data, feature='Close'):
    closing_prices = data[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    closing_prices_scaled = scaler.fit_transform(closing_prices)
    
    train_size = int(0.8 * len(closing_prices_scaled))
    train_data = closing_prices_scaled[:train_size]
    test_data = closing_prices_scaled[train_size:]
    
    return train_data, test_data, scaler
```

- Extracts the Close price by default.

- Normalizes the prices between 0 and 1 using MinMaxScaler.

- Splits the dataset into 80% training and 20% testing.

**Sequence Generation**
```python
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
```

- Converts a 1D time-series into sequences of length sequence_length for RNN input.

- The targets array holds the future price right after the sequence window.

**Model Building**
```python
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
```

- Builds a one-layer RNN in Keras with either:
    - LSTM
    - GRU
    - SimpleRNN

- A Dense(1) layer is used for the final regression output (predicting one value).

**Forecasting Future Prices**
```python
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
```

- Uses the trained model to predict additional time steps beyond the test set.

- last_sequence is updated with each prediction, simulating rolling forecasts.

**Visualization**
```python
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
```

- Plots the true vs. predicted prices on the test set.

- Plots future forecasts on a separate part of the figure with a dashed line

### 6. Running the App

1. Clone this repository or copy the code into a local virtual environment.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit App:
```bash
streamlit run app.py
```

4. Usage:
  - Enter a stock symbol (e.g., AAPL), choose an RNN type (LSTM, GRU, or RNN).
  - Select hyperparameters such as number of epochs, batch size, sequence length, and number of RNN units.
  - Click the year-range buttons (1 Year, 2 Years, etc.) to fetch and display historical data.
  - The model will train on the chosen data, generate predictions, and forecast future days.
  - A table shows the forecasted prices, and a plot shows both predictions and future forecasts.

### 7. Potential Extensions 

- Add Dropout Layers: To address overfitting in volatile markets.

- Incorporate Fundamental/News Data: Enhance the model with additional signals (e.g., news sentiment, macroeconomic factors).

- Experiment with Multiple Features: Use multiple technical indicators (Volume, Moving Averages, RSI, etc.) in the model input.

- Deploy on the Web: You can deploy the Streamlit app on platforms like Streamlit Cloud or Heroku for public access.

### 8. References

1. Medium Article: The Importance of [Sequential Data](https://medium.com/@sotelojayy13/recurrent-neural-networks-understanding-the-pioneers-of-sequential-data-processing-be1404e89e61)

2. Keras Documentation: [Recurrent Layers](https://keras.io/)

3. Streamlit Documentation: [Streamlit Docs](https://docs.streamlit.io/)

4. yfinance: [yfinance Docs](https://pypi.org/project/yfinance/)

<h2> Thank You! </h2>
<p>Feel free to reach out or contribute if you have ideas for improving the application or expanding its functionality. If you found this helpful, consider following me on social media for more insights into Deep Learning and MLOps!</p>

<p> Disclaimer: This project is for educational purposes. The forecasts generated by this model should not be used as financial advice or the sole basis for investment decisions. Always conduct thorough research and consult professionals.</p>




