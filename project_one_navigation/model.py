import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [50, 50], drop_p=0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        print('versao 1.0')
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)
        
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        return self.output(x)


