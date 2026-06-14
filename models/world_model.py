import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel
from models.encoder import Encoder, Decoder
from models.dynamics_model import DynamicsModel
from models.ssim_loss import ssim_loss

def gradient_loss(pred, target):
    pred_dx =  pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx =  target[:, :, :, 1:] - target[:, :, :, :-1]
    
    pred_dy =  pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy =  target[:, :, 1:, :] - target[:, :, :-1, :]

    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024, n_actions=4, feature_dim=None):
        super().__init__()

        if feature_dim is None:
            feature_dim = embed_dim

        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)

        print(f"Encoder output shape: {self.encoder.get_output_shape()}")
        self.decoder = Decoder(embed_dim=embed_dim,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels())

        self.dynamics = DynamicsModel(embed_dim=embed_dim, n_actions=n_actions, hidden_dim=2048)

        self.embed_norm_layer = nn.LayerNorm(embed_dim)

        self.reward_pred = nn.Linear(embed_dim + n_actions, 1)
        self.done_pred = nn.Linear(embed_dim + n_actions, 1)

        self.embed_dim = embed_dim
        self.n_actions = n_actions


    def normalize_embedding(self, embed):
        return self.embed_norm_layer(embed)

    def encode(self, obs):
        embed = self.encoder(obs)
        embed = self.normalize_embedding(embed)
        return embed

    def decode(self, embeds):
        return self.decoder(embeds)

    def imagine_step(self, embed, action):

        delta = self.dynamics(embed, action)
        next_embed = embed + delta
        next_embed = self.normalize_embedding(next_embed)

        embed_action = torch.cat([embed, action], dim=-1)

        reward = self.reward_pred(embed_action)
        done = torch.sigmoid(self.done_pred(embed_action))

        return next_embed, reward, done

    def forward(self, obs, action):

        embeds = self.encode(obs)

        embeds_flat = embeds.view(-1, embeds.shape[-1])
        recon = self.decode(embeds_flat)

        next_embed_pred = self.dynamics(embeds_flat, action)
        next_embed_pred = embeds_flat + next_embed_pred
        next_embed_pred = self.normalize_embedding(next_embed_pred)

        embed_action = torch.cat([embeds_flat, action], dim=-1)
        reward_pred = self.reward_pred(embed_action)

        done_pred = torch.sigmoid(self.done_pred(embed_action))

        return recon, embeds, next_embed_pred, reward_pred, done_pred

    def compute_loss(self, obs, actions, rewards, next_obs, dones):

        obs_normalized = obs.float() / 255.0
        next_obs_normalized = next_obs.float() / 255.0

        if obs_normalized.ndim == 5:
            obs_normalized = obs_normalized.squeeze(1)
        if next_obs_normalized.ndim == 5:
            next_obs_normalized = next_obs_normalized.squeeze(1)

        action_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()

        recon, _, next_embed_pred, reward_pred, done_pred = self.forward(obs_normalized, action_onehot)

        recon_loss = F.l1_loss(recon, obs_normalized) + 0.2 * ssim_loss(recon, obs_normalized) + 0.1 * gradient_loss(recon, obs_normalized)

        next_embeds = self.encode(next_obs_normalized)
        next_embed_target = next_embeds.view(-1, next_embeds.shape[-1])

        dynamics_loss = F.mse_loss(next_embed_pred, next_embed_target.detach())

        reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards.float())
        
        done_loss = F.binary_cross_entropy(done_pred.squeeze(-1), dones.float())

        combined_loss = (
            1.0 * recon_loss +
            1.0 * dynamics_loss +
            2.0 * reward_loss +
            0.5 * done_loss
        )

        return combined_loss, {
            "total": combined_loss.item(),
            "recon": recon_loss.item(),
            "dynamics": dynamics_loss.item(),
            "reward": reward_loss.item(),
            "done": done_loss.item(),
        }






