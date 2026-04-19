import torch
import torch.nn as nn
import os
from typing import List, Tuple, Optional


class LSTMModel(nn.Module):
    """
    Flexible LSTM model that can be configured with different layer architectures.
    """
    
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.layers = nn.ModuleList()
    
    def layer_maker(self, type: str, input_size: int, hidden_size: int, num_layers: int) -> nn.Module:
        """
        Create a layer based on type and parameters.
        
        Args:
            type: Layer type ('LSTM', 'GRU', 'RNN', 'linear')
            input_size: Input size for the layer
            hidden_size: Hidden size for the layer
            num_layers: Number of layers (for RNN types)
        
        Returns:
            PyTorch layer module
        """
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
    
    def create(self, x: torch.Tensor, struct: List[dict]) -> Tuple[List[dict], nn.ModuleList]:
        """
        Create model architecture based on structure definition.
        
        Args:
            x: Sample input tensor to determine input size
            struct: List of layer dictionaries defining architecture
        
        Returns:
            Tuple of (structure, layers)
        """
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
    
    def validate_struct(self, struct: List[dict], expected_input_size: int) -> None:
        """
        Validate the architecture structure.
        
        Args:
            struct: List of layer dictionaries
            expected_input_size: Expected input size for first layer
        """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ModelTrainer:
    """
    Handles model training, saving, and loading.
    """
    
    def __init__(self, model: LSTMModel, device: Optional[str] = None):
        """
        Initialize the ModelTrainer.
        
        Args:
            model: LSTM model to train
            device: Device to use for training ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_model(self, 
                   X_train: torch.Tensor, 
                   y_train: torch.Tensor, 
                   X_test: Optional[torch.Tensor] = None, 
                   y_test: Optional[torch.Tensor] = None, 
                   epochs: int = 50, 
                   batch_size: int = 32, 
                   lr: float = 0.001) -> Tuple[List[float], List[float]]:
        """
        Train the LSTM model.
        
        Args:
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
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if X_test is not None:
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        test_losses = []
        
        num_batches = len(X_train) // batch_size
        
        print(f"\nTraining Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Batches per epoch: {num_batches}")
        print(f"  Training samples: {len(X_train)}")
        if X_test is not None:
            print(f"  Test samples: {len(X_test)}")
        print("=" * 70)
        
        for epoch in range(epochs):
            self.model.train()
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
                predictions = self.model(batch_X)
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
                self.model.eval()
                with torch.no_grad():
                    test_predictions = self.model(X_test)
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
    
    def save_model(self, 
                   filepath: str, 
                   struct: List[dict], 
                   train_losses: List[float], 
                   test_losses: List[float], 
                   final_train_loss: float, 
                   final_test_loss: float,
                   normalization: Optional[dict] = None) -> None:
        """
        Save model checkpoint with all necessary information.
        
        Args:
            filepath: Path to save the model
            struct: Model architecture structure
            train_losses: Training loss history
            test_losses: Test loss history
            final_train_loss: Final training loss
            final_test_loss: Final test loss
            normalization: Normalization parameters
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'struct': struct,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'normalization': normalization,
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to '{filepath}'")
    
    @staticmethod
    def load_model(filepath: str, device: Optional[str] = None) -> Tuple[LSTMModel, dict]:
        """
        Load a saved model checkpoint.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on
        
        Returns:
            Tuple of (model, checkpoint_data)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(filepath, weights_only=False, map_location=device)
        
        # Create model with the same structure
        model = LSTMModel()
        
        # Create a dummy input to reconstruct the model
        dummy_input = torch.randn(1, 60, 5)  # Default seq_length=60, features=5
        model.create(dummy_input, checkpoint['struct'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from '{filepath}'")
        print(f"Device: {device}")
        
        return model, checkpoint
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_layers': len(self.model.layers),
            'device': self.device,
            'layer_types': [type(layer).__name__ for layer in self.model.layers]
        }
