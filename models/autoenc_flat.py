import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Band_generator_flat(nn.Module):
    def __init__(self, latent_dim=4):
        super(self.__class__, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,latent_dim),
            torch.nn.ReLU()
        )
        
        # Decoder

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,100),
            nn.Unflatten(1, (1, 10, 10))

        )


    def decode(self, z):
        z = self.decoder(z)
        z = (z + torch.transpose(z, 2, 3) )*0.5
        z = (z + torch.rot90( 
                torch.transpose( 
                                torch.rot90(z,1, [2, 3]), 2, 3
                                ),
                                3, [2, 3])
            )*0.5
        return z


    def forward(self,x):    
        z = self.encoder(x)
        
        y = self.decode(z)
        # print(y.shape)
        
        return y
