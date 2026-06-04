import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.base import BaseModel

class QModel(BaseModel):

    def __init__(self, action_dim, hidden_dim=256, embed_dim=1024):
        super(QModel, self).__init__()

        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, action_dim)

        self.apply(self._weights_init)

        print(f"Q-Model initialized")
        print(f"    Input: {embed_dim} embeddings")
        print(f"    Hidden: {hidden_dim}")
        print(f"    Output: {action_dim} actions")


    def forward(self, embeddings):
        x = F.relu(self.fc1(embeddings))
        x = F.relu(self.fc2(x))
        return self.output(x)

    def _weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

