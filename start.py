import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch


def fetch_stock_data(symbols, days=60):
    """Fetch stock data for given symbols for the past N days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = {}
    for symbol in symbols.split(','):
        symbol = symbol.strip()
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            data[symbol] = df['Close'].values
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    
    return data


def prepare_training_data(stock_data):
    """Convert stock data to PyTorch tensors for model training"""
    tensors = {}
    for symbol, prices in stock_data.items():
        # Normalize the prices
        normalized = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
        tensors[symbol] = torch.tensor(normalized, dtype=torch.float32).unsqueeze(-1)
    return tensors

# Fetch stock data for the specified symbols
symbols = 'TSM,AAPL,MSFT,AMZN'
stock_data = fetch_stock_data(symbols, days=60)

# Prepare training data as PyTorch tensors
training_data = prepare_training_data(stock_data)

# Display loaded data
for symbol, tensor in training_data.items():
    print(f"{symbol}: shape {tensor.shape}")
