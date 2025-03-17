import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        if m.out_features == 1:  # Output layers
            torch.nn.init.uniform_(m.weight, -3e-3, 3e-3)  # Small range
            torch.nn.init.constant_(m.bias, 0)
        else:
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    def __init__(self, state_size=33, action_size=4, hidden1=256, hidden2=256):
        super(Critic, self).__init__()
        
        # Q-network 1
        self.fc1_1 = nn.Linear(state_size + action_size, hidden1)
        self.ln1_1 = nn.LayerNorm(hidden1)
        self.fc1_2 = nn.Linear(hidden1, hidden2)
        self.ln1_2 = nn.LayerNorm(hidden2)
        self.fc1_3 = nn.Linear(hidden2, 1)
        
        # Q-network 2
        self.fc2_1 = nn.Linear(state_size + action_size, hidden1)
        self.ln2_1 = nn.LayerNorm(hidden1)
        self.fc2_2 = nn.Linear(hidden1, hidden2)
        self.ln2_2 = nn.LayerNorm(hidden2)
        self.fc2_3 = nn.Linear(hidden2, 1)
        
        self.fc1_1.apply(weights_init_)
        self.fc1_2.apply(weights_init_)
        self.fc1_3.apply(weights_init_)
        
        self.fc2_1.weight.data.copy_(self.fc1_1.weight.data)
        self.fc2_2.weight.data.copy_(self.fc1_2.weight.data)
        self.fc2_3.weight.data.copy_(self.fc1_3.weight.data)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        
        # Q1 computation
        x1 = self.fc1_1(xu)
        x1 = self.ln1_1(x1)
        x1 = F.relu(x1)
        
        x1 = self.fc1_2(x1)
        x1 = self.ln1_2(x1)
        x1 = F.relu(x1)
        q1 = self.fc1_3(x1)
        
        # Q2 computation
        x2 = self.fc2_1(xu)
        x2 = self.ln2_1(x2)
        x2 = F.relu(x2)
        
        x2 = self.fc2_2(x2)
        x2 = self.ln2_2(x2)
        x2 = F.relu(x2)
        q2 = self.fc2_3(x2)
        
        return q1, q2