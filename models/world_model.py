import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel
from models.encoder import Encoder, Decoder
from models.dynamics_model import DynamicsModel
from models.ssim_loss import ssim_loss

class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024, n_actions=4, feature_dim=None):
        super().__init__()

        if feature_dim is None:
            feature_dim = embed_dim

        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)
        self.decoder = Decoder(embed_dim=embed_dim,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels())

        self.dynamics = DynamicsModel(embed_dim=embed_dim, n_actions=n_actions, hidden_dim=2048)

        self.embed_norm_layer = nn.LayerNorm(embed_dim)

        self.reward_pred = nn.Linear(embed_dim + n_actions, 1)
        self.done_pred = nn.Linear(embed_dim + n_actions, 1)

        self.embed_dim = embed_dim
        self.n_actions = n_actions


        





