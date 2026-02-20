import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from start import StockDataManager






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
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            Output tensor (batch, 1)
        """
        sequence_collapsed = False
        
        for layer in self.layers:
            if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
                
            elif isinstance(layer, nn.Linear):
                if not sequence_collapsed and x.dim() == 3:
                    x = x[:, -1, :]
                    sequence_collapsed = True
                
                x = layer(x)
        
        return x


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, X_train, y_train, X_test=None, y_test=None, epochs=50, batch_size=32, lr=0.001):    
    """
    Train the LSTM model.
    
    Args:
        model: LSTM model to train
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data (optional)
        y_test: Test target data (optional)
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    num_batches = len(X_train) // batch_size
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Batches per epoch: {num_batches}")
    print(f"  Training samples: {len(X_train)}")
    if X_test is not None:
        print(f"  Test samples: {len(X_test)}")
    print("=" * 70)
    
    for epoch in range(epochs):
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
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        test_loss = None
        if X_test is not None and y_test is not None:
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test)
                test_loss = criterion(test_predictions, y_test).item()
                test_losses.append(test_loss)
        
        # Print progress
        if epochs <= 20 or (epoch + 1) % 5 == 0 or epoch == 0:
            if test_loss is not None:
                print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Test Loss: {test_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {avg_train_loss:.4f}")
    
    print("=" * 70)
    print("Training Complete!")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    if test_losses:
        print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print("=" * 70)
    
    return train_losses, test_losses



def evaluate_predictions(model, X_test, y_test, mean_y, std_y, show_all=False):

    print("\n" + "="*70)
    print("TEST SET PREDICTIONS vs ACTUAL PRICES")
    print("="*70)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        test_predictions_norm = model(X_test)
    
    # Denormalize to get actual dollar prices
    predictions_actual = test_predictions_norm * std_y + mean_y
    targets_actual = y_test * std_y + mean_y
    
    # Calculate metrics
    differences = predictions_actual - targets_actual
    mae = differences.abs().mean().item()
    rmse = (differences ** 2).mean().sqrt().item()
    mape = (differences.abs() / targets_actual * 100).mean().item()
    
    print(f"\nTest Set Performance:")
    print(f"  Total samples: {len(X_test)}")
    print(f"  Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Determine how many to show
    num_to_show = len(predictions_actual) if show_all else min(20, len(predictions_actual))
    
    print(f"\n{'First ' + str(num_to_show) if not show_all else 'All ' + str(num_to_show)} Predictions:")
    print("-"*70)
    print(f"{'#':>4} | {'Predicted':>12} | {'Actual':>12} | {'Difference':>12} | {'Error %':>10}")
    print("-"*70)
    
    for i in range(num_to_show):
        pred = predictions_actual[i].item()
        actual = targets_actual[i].item()
        diff = pred - actual
        error_pct = abs(diff) / actual * 100
        
        print(f"{i+1:4d} | ${pred:11.2f} | ${actual:11.2f} | ${diff:+11.2f} | {error_pct:9.2f}%")
    
    if not show_all and len(predictions_actual) > num_to_show:
        print(f"  ... ({len(predictions_actual) - num_to_show} more samples)")
    
    print("-"*70)
    avg_diff = differences.mean().item()
    avg_error_pct = mape
    print(f"{'AVG':>4} | {'':>12} | {'':>12} | ${avg_diff:+11.2f} | {avg_error_pct:9.2f}%")
    print("="*70)


def plot_training_history(train_losses, test_losses):
    """Plot training and test loss over epochs."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Regular scale
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log scale (better for seeing details)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


