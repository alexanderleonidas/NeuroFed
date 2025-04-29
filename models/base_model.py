import torch.nn as nn
import torch.nn.functional as f


class FlexibleNet(nn.Module):
    def __init__(self, layer_sizes, name='BP'):
        super(FlexibleNet, self).__init__()
        self.name = name
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.flatten = nn.Flatten()
        # Dynamically create and register layers as attributes
        for i in range(self.num_layers):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            setattr(self, f'fc{i+1}', layer)
        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.flatten(x)
        self.inputs = [x]
        self.activations = []

        for i in range(1, self.num_layers):
            layer = getattr(self, f'fc{i}')
            x = layer(x)
            x = f.relu(x)
            self.inputs.append(x)
            self.activations.append((x > 0).float())

        # Last layer (output layer, no activation)
        x = getattr(self, f'fc{self.num_layers}')(x)
        self.out = x
        return x