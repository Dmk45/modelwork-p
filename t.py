import struct
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.layers = nn.ModuleList()
    
    def layer_maker(self, type, input_size, hidden_size, num_layers):
        if type == 'LSTM':
            return nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'GRU':
            return nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'RNN':
            return nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'linear':
            return nn.Linear(input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported layer type: {type}")
    
    def create(self, x, struct: list[dict]):
        self.validate_struct(struct, expected_input_size=x.size(-1))
        for layer in struct:
            self.layers.append(
                self.layer_maker(
                    type=layer["type"], 
                    input_size=layer["input_size"], 
                    hidden_size=layer.get("hidden_size"), 
                    num_layers=layer.get("num_layers")
                )
            )
        return struct, self.layers
    def forward(self, x):
        lstm_outputs_processed = False
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
                
            elif isinstance(layer, nn.Linear):
                # If this is the first Linear layer after LSTM(s)
                # we need to collapse the sequence dimension
                if not lstm_outputs_processed and x.dim() == 3:
                    x = x[:, -1, :]
                    lstm_outputs_processed = True
                
                # Apply linear transformation
                x = layer(x)  # (batch, hidden_size) → (batch, output_size)
        
        return x
    
    def validate_struct(self, struct: list[dict], expected_input_size: int):
        prev_output_size = expected_input_size

        for i, layer in enumerate(struct):
            required = {"type", "input_size", "output_size"}
            missing = required - layer.keys()
            if missing:
                raise ValueError(f"Layer {i} missing keys: {missing}")

            if layer["input_size"] != prev_output_size:
                raise ValueError(
                    f"Layer {i} input_size {layer['input_size']} "
                    f"does not match previous output_size {prev_output_size}"
                )

            if layer["type"] in {"LSTM", "GRU", "RNN"}:
                if "hidden_size" not in layer or "num_layers" not in layer:
                    raise ValueError(f"Layer {i} missing hidden_size or num_layers")

                if layer["output_size"] != layer["hidden_size"]:
                    raise ValueError(
                        f"Layer {i} output_size must equal hidden_size for RNN layers"
                    )

            prev_output_size = layer["output_size"]

    
import numpy as np
def generate_sample_stock_data(num_samples=32, seq_length=60, num_features=5):
    """
    Generate sample stock data.
    
    Args:
        num_samples: Number of sequences (batch size)
        seq_length: Number of days in each sequence
        num_features: Number of features (OHLCV = 5)
    
    Returns:
        X: Stock data (num_samples, seq_length, num_features)
    """
    np.random.seed(42)
    
    X = []
    for _ in range(num_samples):
        # Starting price around $100-$500
        start_price = np.random.uniform(100, 500)
        
        # Generate price trend over time
        sequence = []
        for day in range(seq_length):
            # Simulate daily price movement
            price = start_price + np.random.normal(0, 5) * day * 0.1
            
            # OHLCV features
            open_price = price + np.random.uniform(-2, 2)
            high_price = price + np.random.uniform(0, 3)
            low_price = price - np.random.uniform(0, 3)
            close_price = price + np.random.uniform(-1, 1)
            volume = np.random.uniform(1000000, 10000000)
            
            sequence.append([open_price, high_price, low_price, close_price, volume])
        
        X.append(sequence)
    
    return torch.FloatTensor(X)



structt = [
    {
        "type": "LSTM",
        "input_size": 5,
        "hidden_size": 128,
        "num_layers": 8,
        "output_size": 128
    },
    {
        "type": "LSTM",
        "input_size": 128,
        "hidden_size": 64,
        "num_layers": 3,
        "output_size": 64
    },
    {
        "type": "linear",
        "input_size": 64,
        "hidden_size": 32,
        "output_size": 32
    },
    {
        "type": "linear",
        "input_size": 32,
        "hidden_size": 1,
        "output_size": 1
    }
]

def train_model(model, X_train, y_train, epochs=50, batch_size=32, lr=0.001):    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []

    
    num_batches = len(X_train) // batch_size
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Batches per epoch: {num_batches}")
    print("=" * 70)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0
        
        # Shuffle training data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            batch_X = X_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}", end="\r")
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
    print("=" * 70)
    return train_losses


def evaluate_model(model, X_test, y_test, mean_y, std_y):
    """Evaluate model and show sample predictions."""
    model.eval()
    
    with torch.no_grad():
        predictions_norm = model(X_test)
    
    # Denormalize predictions and targets
    predictions = predictions_norm * std_y + mean_y
    targets = y_test * std_y + mean_y
    
    # Calculate metrics
    mse = ((predictions - targets) ** 2).mean().item()
    mae = (predictions - targets).abs().mean().item()
    
    print("\nEvaluation Results:")
    print("=" * 70)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"\nSample Predictions (denormalized to actual prices):")
    print("-" * 70)
    print(f"{'Predicted':>12} | {'Actual':>12} | {'Difference':>12}")
    print("-" * 70)
    
    for i in range(min(10, len(predictions))):
        pred = predictions[i].item()
        actual = targets[i].item()
        diff = abs(pred - actual)
        print(f"{pred:12.2f} | {actual:12.2f} | {diff:12.2f}")



from start import StockDataManager
model = LSTMModel()


#evaluate_model(model, generate_sample_stock_data(), torch.rand(32, 1), mean_y=250, std_y=100)


print("\n### EXAMPLE 1: Fetch and Prepare Data ###\n")
    
manager = StockDataManager(save_dir='stock_data')
    
data = manager.prepare_for_training(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        seq_length=60,           # Use 60 days to predict next day
        prediction_days=1,       # Predict 1 day ahead
        total_days=365,          # Fetch 1 year of data
        train_ratio=0.8,         # 80% train, 20% test
        normalize=True,          # Normalize data
        save=True                # Save to disk
    )
    
    # Access the data
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
    
structt, layers = model.create(X_train, structt)
train_model(X_train, y_train, epochs=5)