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


def train_model(model, X_train, y_train, X_test=None, y_test=None, epochs=50, batch_size=32, lr=0.001):    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    num_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        if X_test is not None and y_test is not None:
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test)
                test_loss = criterion(test_predictions, y_test).item()
                test_losses.append(test_loss)
        
        if epochs <= 20 or (epoch + 1) % 5 == 0 or epoch == 0:
            if test_loss is not None:
                print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Test Loss: {test_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {avg_train_loss:.4f}")
    
    return train_losses, test_losses


def save_model(model, struct, train_losses, test_losses, normalization, filename='stock_predictor_model.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'struct': struct,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'normalization': normalization,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }
    
    torch.save(checkpoint, filename)
    print(f"Model saved to '{filename}'")