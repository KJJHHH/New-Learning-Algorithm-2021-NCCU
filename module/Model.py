import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4, dtype = torch.float64):
        super(TwoLayerNet, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim).to(dtype = dtype)
        self.layer_out = nn.Linear(hidden_dim, output_dim).to(dtype = dtype)
        self.relu = nn.ReLU().to(dtype = dtype)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.layer_out(x)    
        return x