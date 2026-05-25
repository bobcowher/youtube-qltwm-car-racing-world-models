import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class DynamicsModel(BaseModel):
    """
    Predicts the next latent state from the current latent state and the action. 
    """

    def __init__(self, embed_dim=1024, n_actions=4, hidden_dim=2048):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_actions = n_actions

        self.fc1 = nn.Linear(embed_dim + n_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.output = nn.Linear(hidden_dim, embed_dim)

    def forward(self, embed, action):
        x = torch.cat([embed, action], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.output(x)


