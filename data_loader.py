import torch
import pickle
import os
from typing import Tuple, Optional


class DataLoader:
    """
    Handles loading and preprocessing of stock data from preset paths.
    """
    
    def __init__(self, data_dir: str = 'stock_data'):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing saved data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_data(self, filename: str = 'stock_data.pkl') -> dict:
        """
        Load previously saved stock data.
        
        Args:
            filename: Name of file to load (from data_dir)
        
        Returns:
            Dictionary with loaded data including X, y, symbols, and normalization params
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        print(f"Data loaded from {filepath}")
        print(f"Symbols: {data_dict['symbols']}")
        print(f"X shape: {data_dict['X'].shape}")
        print(f"y shape: {data_dict['y'].shape}")
        
        return data_dict
    
    def prepare_training_data(self, 
                             data_dict: dict, 
                             train_ratio: float = 0.8,
                             reshape_y: bool = True) -> dict:
        """
        Prepare loaded data for training by splitting and optionally reshaping.
        
        Args:
            data_dict: Loaded data dictionary
            train_ratio: Ratio of data to use for training
            reshape_y: Whether to reshape y tensors to (batch_size, 1)
        
        Returns:
            Dictionary with prepared training data
        """
        X = data_dict['X']
        y = data_dict['y']
        
        # Split data
        train_size = int(len(X) * train_ratio)
        
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Reshape y if needed
        if reshape_y:
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()
            
            if y_train.dim() == 1:
                y_train = y_train.unsqueeze(1)
            if y_test.dim() == 1:
                y_test = y_test.unsqueeze(1)
        
        print(f"\nData preparation complete:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_full': X,
            'y_full': y,
            'symbols': data_dict['symbols'],
            'raw_data': data_dict['raw_data'],
            'normalization': {
                'mean_X': data_dict['mean_X'],
                'std_X': data_dict['std_X'],
                'mean_y': data_dict['mean_y'],
                'std_y': data_dict['std_y']
            }
        }
    
    def get_data_info(self, data_dict: dict) -> dict:
        """
        Get information about the loaded data.
        
        Args:
            data_dict: Loaded data dictionary
        
        Returns:
            Dictionary with data information
        """
        return {
            'num_samples': len(data_dict['X']),
            'seq_length': data_dict['X'].shape[1],
            'num_features': data_dict['X'].shape[2],
            'symbols': data_dict['symbols'],
            'has_normalization': all(key in data_dict for key in ['mean_X', 'std_X', 'mean_y', 'std_y'])
        }
    
    def validate_data(self, data_dict: dict) -> bool:
        """
        Validate that the loaded data has the expected structure.
        
        Args:
            data_dict: Loaded data dictionary
        
        Returns:
            True if data is valid, raises exception if invalid
        """
        required_keys = ['X', 'y', 'symbols', 'raw_data']
        missing_keys = [key for key in required_keys if key not in data_dict]
        
        if missing_keys:
            raise ValueError(f"Missing required keys in data: {missing_keys}")
        
        X = data_dict['X']
        y = data_dict['y']
        
        # Check tensor shapes
        if X.dim() != 3:
            raise ValueError(f"Expected X to be 3D tensor, got {X.dim()}D")
        
        if y.dim() != 2 or y.shape[-1] != 1:
            raise ValueError(f"Expected y to be 2D tensor with last dimension 1, got shape {y.shape}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of samples: {len(X)} vs {len(y)}")
        
        # Check for NaN or infinite values
        if torch.isnan(X).any():
            raise ValueError("X contains NaN values")
        if torch.isinf(X).any():
            raise ValueError("X contains infinite values")
        if torch.isnan(y).any():
            raise ValueError("y contains NaN values")
        if torch.isinf(y).any():
            raise ValueError("y contains infinite values")
        
        print("Data validation passed!")
        return True
    
    def create_sample_batch(self, data_dict: dict, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a sample batch for testing model architecture.
        
        Args:
            data_dict: Loaded data dictionary
            batch_size: Size of sample batch
        
        Returns:
            Tuple of (X_batch, y_batch)
        """
        X = data_dict['X']
        y = data_dict['y']
        
        # Take first batch_size samples
        X_batch = X[:batch_size]
        y_batch = y[:batch_size]
        
        # Reshape y to match training format
        y_batch = y_batch.squeeze()
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)
        
        print(f"Created sample batch:")
        print(f"  X_batch shape: {X_batch.shape}")
        print(f"  y_batch shape: {y_batch.shape}")
        
        return X_batch, y_batch
