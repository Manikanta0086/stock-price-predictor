import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class StockPredictor:
    def _init_(self, symbol='AAPL', period='2y'):
        self.symbol = symbol
        self.period = period
        self.scaler = MinMaxScaler()
        self.model = None
        self.data = None

    def fetch_data(self):
        """Fetch historical stock data"""
        print(f"Fetching data for {self.symbol}...")
        self.data = yf.download(self.symbol, period=self.period)
        self.data = self.data[['Close']].dropna()
        return self.data

    def prepare_data(self, lookback=60):
        """Prepare data for LSTM"""
        scaled_data = self.scaler.fit_transform(self.data)

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = int(0.8 * len(X))
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

    def build_model(self, lookback=60):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, epochs=30, batch_size=32):
        """Train the model"""
        return self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test)
        )

    def predict(self, days=30):
        """Predict future stock prices"""
        test_predictions = self.model.predict(self.X_test)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        last_seq = self.scaler.transform(self.data[-60:])
        last_seq = last_seq.reshape(1, 60, 1)

        future_predictions = []
        for _ in range(days):
            pred = self.model.predict(last_seq)
            future_predictions.append(pred[0, 0])
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, 0] = pred[0, 0]

        return test_predictions, actual, np.array(future_predictions)
