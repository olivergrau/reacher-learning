import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -1
LOG_STD_MAX = 1
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom weight initialization
def weights_init_(m):
    if isinstance(m, nn.Linear):
        if m.out_features > 1:  # Hidden layers
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
        else:  # Final layer (mean output)
            torch.nn.init.uniform_(m.weight, -3e-3, 3e-3)
            torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_size=33, action_size=4, hidden1=256, hidden2=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)  # Stabilizes learning
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)

        self.mean_fc = nn.Linear(hidden2, action_size)  # Outputs mean
        self.log_std_fc = nn.Linear(hidden2, action_size)  # Outputs log std deviation

        # Apply correct weight initialization
        self.fc1.apply(weights_init_)
        self.fc2.apply(weights_init_)
        self.mean_fc.apply(weights_init_)
        torch.nn.init.uniform_(self.log_std_fc.weight, -3e-3, 3e-3)
        torch.nn.init.constant_(self.log_std_fc.bias, -2.0)  # Log std bias = -2

    def forward(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        mean = self.mean_fc(x)  # Mean action values
        log_std = self.log_std_fc(x).clamp(LOG_STD_MIN, LOG_STD_MAX)  # Keep log_std within a reasonable range

        return mean, log_std

    def sample(self, state):
        """
        Given a state, sample an action using the reparameterization trick.
        Returns:
            - action: The squashed action via tanh, within [-1, 1]
            - log_prob: Log probability of the action (with tanh correction)
            - pre_tanh_value: The raw action values before applying tanh (for diagnostics)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick: sample from Normal using rsample()
        pre_tanh = normal.rsample()  # mean + std * noise
        action = torch.tanh(pre_tanh)

        # Compute log-probability with tanh correction
        log_prob = normal.log_prob(pre_tanh)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Tanh correction: log(1 - tanh(x)^2)
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True)
        
        return action, log_prob, pre_tanh, std