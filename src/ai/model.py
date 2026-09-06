import torch
import torch.nn as nn

class TetrisAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(200, 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 5)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = self.input_layer(x)
        x = self.ReLU(x)

        x = self.hidden_layer(x)
        x = self.ReLU(x)

        x = self.output_layer(x)
        
        return x