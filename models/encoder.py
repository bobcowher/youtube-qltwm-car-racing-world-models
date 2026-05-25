import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class Encoder(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape, dtype=torch.uint8)
            feats = self._conv_features(dummy)
            self.conv_output_shape = feats.shape[1:]
            self.flattened_dim = feats.numel() // 1
            print(f"Conv output shape: {feats.shape}, flattened dim: {self.flattened_dim}")

        self.fc_enc = nn.Linear(self.flattened_dim, embed_dim)

        print(f"Encoder initialized. Input shape {observation_shape}")

        
    def _conv_features(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        return x.flatten(1)

    def _conv_forward(self, x):
        x = self._conv_features(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._conv_forward(x)
        x = self.fc_enc(x)
        return x


class Decoder(BaseModel):

    def __init__(self, embed_dim=1024, conv_output_shape=(128, 12, 12)):
        super().__init__()

        self.conv_output_shape=conv_output_shape

        conv_flat_size = conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2]

        self.fc_dec = nn.Linear(embed_dim, conv_flat_size)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1)

    def _deconv_forward(self, x):

        x = x.view(-1, *self.conv_output_shape)
        x = F.elu(self.deconv1(x))
        x = F.elu(self.deconv2(x))
        x = self.deconv3(x)

        return x

    def forward(self, x):
        x = self.fc_dec(x)
        x = self._deconv_forward(x)
        x = torch.sigmoid(x)
        return x
            

    
    


