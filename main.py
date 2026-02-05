import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from model import StockPredictor


def run_model():
    predictor = StockPredictor('AAPL', '2y')

    data = predictor.fetch_data()
    predictor.prepare_data()
    predictor.build_model()

    history = predictor.train()

    test_pred, actual, future_pred = predictor.predict(days=30)

    rmse = np.sqrt(mean_squared_error(actual, test_pred))
    print(f"RMSE: {rmse:.2f}")

    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:100], label="Actual")
    plt.plot(test_pred[:100], label="Predicted")
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.show()

    print("\nNext 10 Days Prediction:")
    for i, price in enumerate(future_pred[:10]):
        print(f"Day {i + 1}: ${price:.2f}")
