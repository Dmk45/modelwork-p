# Stock Price Predictor

A comprehensive LSTM-based stock price prediction system with modular architecture.

## Architecture

The codebase is organized into 4 main modules:

### 1. `data_extractor.py` - Data Extraction
- **Purpose**: Extracts stock data from web sources
- **Key Class**: `StockDataManager`
- **Features**:
  - Fetches historical stock data from Yahoo Finance
  - Creates sliding window sequences for LSTM training
  - Handles data normalization and preprocessing
  - Saves/loads processed data

### 2. `data_loader.py` - Data Loading
- **Purpose**: Loads data from preset paths
- **Key Class**: `DataLoader`
- **Features**:
  - Loads preprocessed stock data from disk
  - Validates data integrity
  - Prepares training/test splits
  - Creates sample batches for testing

### 3. `model_trainer.py` - Model Training
- **Purpose**: Takes data from loader and saves model file
- **Key Classes**: `LSTMModel`, `ModelTrainer`
- **Features**:
  - Flexible LSTM architecture with configurable layers
  - Comprehensive training pipeline with validation
  - Model saving/loading functionality
  - Device management (CPU/GPU)

### 4. `visualizer_evaluator.py` - Visualization & Evaluation
- **Purpose**: Used for testing the model
- **Key Class**: `ModelEvaluator`
- **Features**:
  - Comprehensive model evaluation metrics
  - Training history visualization
  - Prediction vs actual plotting
  - Residual analysis
  - Detailed evaluation reports

## Usage Example

```python
from data_extractor import StockDataManager
from data_loader import DataLoader
from model_trainer import LSTMModel, ModelTrainer
from visualizer_evaluator import ModelEvaluator

# 1. Extract and prepare data
extractor = StockDataManager()
data = extractor.prepare_for_training(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    seq_length=60,
    total_days=365
)

# 2. Load and validate data
loader = DataLoader()
prepared_data = loader.prepare_training_data(data)

# 3. Create and train model
model = LSTMModel()
model.create(prepared_data['X_train'], [
    {"type": "LSTM", "input_size": 5, "hidden_size": 128, "num_layers": 2, "output_size": 128},
    {"type": "linear", "input_size": 128, "hidden_size": 64, "output_size": 64},
    {"type": "linear", "input_size": 64, "hidden_size": 1, "output_size": 1}
])

trainer = ModelTrainer(model)
train_losses, test_losses = trainer.train_model(
    prepared_data['X_train'], 
    prepared_data['y_train'],
    prepared_data['X_test'],
    prepared_data['y_test'],
    epochs=100
)

# 4. Evaluate and visualize
evaluator = ModelEvaluator(model)
report = evaluator.generate_report(
    prepared_data['X_test'],
    prepared_data['y_test'],
    data['normalization']['mean_y'],
    data['normalization']['std_y'],
    train_losses,
    test_losses
)
```

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` - Deep learning framework
- `yfinance` - Stock data fetching
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `scipy` - Statistical analysis (for evaluation)

## Features

- **Modular Design**: Clean separation of concerns across 4 main modules
- **Flexible Architecture**: Configurable LSTM model structure
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Data Management**: Robust data loading, validation, and preprocessing
- **Device Support**: Automatic GPU/CPU detection and usage

## Model Architecture

The LSTM model supports:
- Multiple RNN types (LSTM, GRU, RNN)
- Configurable layer stacks
- Automatic sequence handling
- Flexible output dimensions

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score
- Direction Accuracy
- Residual Analysis

## File Structure

```
Modelwork/
|-- data_extractor.py      # Web data extraction
|-- data_loader.py         # Data loading and validation
|-- model_trainer.py       # Model training and management
|-- visualizer_evaluator.py # Evaluation and visualization
|-- requirements.txt       # Dependencies
|-- README.md             # This file
|-- stock_data/           # Data storage directory
|-- stock_predictor_model.pth  # Trained model checkpoint
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the example usage code above
3. Check the `stock_data/` directory for processed data
4. Look for `stock_predictor_model.pth` for the trained model

The system is designed to be modular and extensible, allowing for easy customization of data sources, model architectures, and evaluation methods.
