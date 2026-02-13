import struct
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
    
    def layer_maker(self, type, input_size, hidden_size, num_layers, output_size):
        if type == 'LSTM':
            lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'GRU':
            gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'RNN':
            rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'linear':
            lin = nn.Linear(input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported layer type: {type}")
    def create(self, x, struct: list[dict]):
        self.validate_struct(struct, expected_input_size=x.size(-1))
        layers = []
        for layer in struct:
            layers.append(self.layer_maker(type=layer["type"], input_size=layer["input_size"], hidden_size=layer.get("hidden_size"), num_layers=layer.get("num_layers")))
        return struct, layers
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


