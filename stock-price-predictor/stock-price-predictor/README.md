# Stock Price Predictor

This project implements a stock price prediction model using LSTM (Long Short-Term Memory) networks. The model is designed to predict future stock prices based on historical data.

## Project Structure

- `data_loader.py`: Contains the `StockDataManager` class responsible for loading and preparing stock data for training. It includes methods for fetching data, normalizing it, and splitting it into training and testing sets.

- `train_model.py`: Includes the `LSTMModel` class that defines the architecture of the LSTM model, along with the `train_model` function that handles the training process, including loss calculation and optimization. It also saves the trained model to a file.

- `evaluate_model.py`: Contains functions for evaluating the model's predictions, visualizing training history, and loading saved models. It includes the `evaluate_predictions` function for calculating performance metrics and the `plot_training_history` function for visualizing training and test losses.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock-price-predictor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Load and prepare the data using `data_loader.py`.
2. Train the model using `train_model.py`.
3. Evaluate the model's performance and visualize results using `evaluate_model.py`.

## License

This project is licensed under the MIT License.