import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import pickle
import os
from typing import List, Tuple, Optional


class StockDataManager:
    """
    Manages stock data fetching, preprocessing, and storage for LSTM model training.
    
    Data is structured as: (num_samples, seq_length, num_features)
    Features: [Open, High, Low, Close, Volume]
    """
    
    def __init__(self, save_dir='stock_data'):
        """
        Initialize the StockDataManager.
        
        Args:
            save_dir: Directory to save/load data
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.raw_data = None
        self.X = None
        self.y = None
        self.symbols = None
        self.mean_X = None
        self.std_X = None
        self.mean_y = None
        self.std_y = None
        
    def fetch_data(self, 
                   symbols: List[str], 
                   seq_length: int = 60,
                   prediction_days: int = 1,
                   total_days: int = 365) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch stock data from Yahoo Finance and prepare for LSTM training.
        
        Args:
            symbols: List of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            seq_length: Number of days to use as input sequence (default: 60)
            prediction_days: Number of days ahead to predict (default: 1)
            total_days: Total days of historical data to fetch (default: 365)
        
        Returns:
            X: Input sequences (num_samples, seq_length, 5)
            y: Target prices (num_samples, 1)
        """
        print(f"Fetching data for {len(symbols)} stocks...")
        print(f"Sequence length: {seq_length}, Total days: {total_days}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days + seq_length)
        
        self.symbols = symbols
        self.raw_data = {}
        
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                print(f"  Downloading {symbol}...", end=' ')
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if len(df) < seq_length + prediction_days:
                    print(f"SKIPPED (insufficient data: {len(df)} days)")
                    continue
                
                self.raw_data[symbol] = df
                print(f"OK ({len(df)} days)")
                
            except Exception as e:
                print(f"ERROR: {e}")
        
        if not self.raw_data:
            raise ValueError("No stock data was successfully fetched!")
        
        # Create training sequences
        X, y = self._create_sequences(seq_length, prediction_days)
        
        self.X = X
        self.y = y
        
        print(f"\nCreated {len(X)} training sequences")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        return X, y
    
    def _create_sequences(self, seq_length: int, prediction_days: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sliding window sequences from raw stock data.
        
        Args:
            seq_length: Length of input sequences
            prediction_days: Days ahead to predict
        
        Returns:
            X: Input sequences with OHLCV features
            y: Target closing prices
        """
        all_X = []
        all_y = []
        
        for symbol, df in self.raw_data.items():
            # Extract OHLCV features 
            open_prices = df['Open'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values
            volume = df['Volume'].values
            
            # Create sliding windows
            for i in range(len(df) - seq_length - prediction_days + 1):
                # Input sequence (seq_length days of OHLCV)
                sequence = np.column_stack([
                    open_prices[i:i + seq_length],
                    high_prices[i:i + seq_length],
                    low_prices[i:i + seq_length],
                    close_prices[i:i + seq_length],
                    volume[i:i + seq_length]
                ])
                
                # Target (closing price N days ahead)
                target = close_prices[i + seq_length + prediction_days - 1]
                
                all_X.append(sequence)
                all_y.append(target)
        
        X = torch.FloatTensor(np.array(all_X))
        y = torch.FloatTensor(np.array(all_y)).unsqueeze(-1)
        
        return X, y
    
    def normalize_data(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize features to mean=0, std=1.
        
        Args:
            X: Input sequences
            y: Target values
        
        Returns:
            X_norm: Normalized inputs
            y_norm: Normalized targets
        """
        print("\nNormalizing data...")
        
        # Normalize X (reshape to 2D, normalize, reshape back)
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.mean_X = X_reshaped.mean(dim=0, keepdim=True)
        self.std_X = X_reshaped.std(dim=0, keepdim=True)
        X_norm = (X - self.mean_X) / (self.std_X + 1e-8)
        
        # Normalize y
        self.mean_y = y.mean()
        self.std_y = y.std()
        y_norm = (y - self.mean_y) / (self.std_y + 1e-8)
        
        print(f"X normalized: mean={self.mean_X.mean().item():.4f}, std={self.std_X.mean().item():.4f}")
        print(f"y normalized: mean={self.mean_y.item():.4f}, std={self.std_y.item():.4f}")
        
        return X_norm, y_norm
    
    def split_data(self, 
                   X: torch.Tensor, 
                   y: torch.Tensor, 
                   train_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into training and test sets.
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio of data to use for training (default: 0.8)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        train_size = int(len(X) * train_ratio)
        
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_data(self, filename: str = 'stock_data.pkl'):
        """
        Save all data and normalization parameters to file.
        
        Args:
            filename: Name of file to save (will be saved in save_dir)
        """
        filepath = os.path.join(self.save_dir, filename)
        
        data_dict = {
            'X': self.X,
            'y': self.y,
            'symbols': self.symbols,
            'raw_data': self.raw_data,
            'mean_X': self.mean_X,
            'std_X': self.std_X,
            'mean_y': self.mean_y,
            'std_y': self.std_y
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"\nData saved to {filepath}")
    
    def get_normalization_params(self) -> dict:
        """
        Get normalization parameters (needed for predictions).
        
        Returns:
            Dictionary with mean_X, std_X, mean_y, std_y
        """
        if self.mean_X is None:
            raise ValueError("Data not normalized yet. Call normalize_data() first.")
        
        return {
            'mean_X': self.mean_X,
            'std_X': self.std_X,
            'mean_y': self.mean_y,
            'std_y': self.std_y
        }
    
    def prepare_for_training(self,
                           symbols: List[str],
                           seq_length: int = 60,
                           prediction_days: int = 1,
                           total_days: int = 365,
                           train_ratio: float = 0.8,
                           normalize: bool = True,
                           save: bool = True) -> dict:
        """
        All-in-one function to fetch, process, and prepare data for training.
        
        Args:
            symbols: List of stock tickers
            seq_length: Input sequence length (days)
            prediction_days: Days ahead to predict
            total_days: Total historical days to fetch
            train_ratio: Train/test split ratio
            normalize: Whether to normalize data
            save: Whether to save data to disk
        
        Returns:
            Dictionary with all training data and parameters
        """
        print("=" * 70)
        print("PREPARING STOCK DATA FOR TRAINING")
        print("=" * 70)
        
        # Fetch data
        X, y = self.fetch_data(symbols, seq_length, prediction_days, total_days)
        
        # Normalize
        if normalize:
            X, y = self.normalize_data(X, y)
            self.X = X
            self.y = y
        
        # Split
        X_train, X_test, y_train, y_test = self.split_data(X, y, train_ratio)
        
        # Save
        if save:
            self.save_data()
        
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_full': X,
            'y_full': y,
            'symbols': symbols,
            'seq_length': seq_length,
            'num_features': 5,  # OHLCV
            'normalization': self.get_normalization_params() if normalize else None
        }
        
        print("\n" + "=" * 70)
        print("DATA PREPARATION COMPLETE")
        print("=" * 70)
        
        return result
