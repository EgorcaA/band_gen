###
# Autoencoder training
###

#  Train a variational autoencoder so that then we can sample bands from the latent space

import torch
import torch.utils
import torch.utils.data
import csv
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Autoencoder with one hidden layer (= latent space)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(21, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,8),
            torch.nn.ReLU()
        )

        self.encode_mu = torch.nn.Sequential(
            torch.nn.Linear(8,4),
            torch.nn.ReLU()
        )

        self.encode_logvar = torch.nn.Sequential(
            torch.nn.Linear(8,4)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4,8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,21)
        )

        self.mu = torch.Tensor(size=(4,))
        self.logvar = torch.Tensor(size=(4,))
    
    def reparameterise(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self,x):    
        encoded = self.encoder(x)
        self.mu, self.log_var = self.encode_mu(encoded), self.encode_logvar(encoded)
        z = self.reparameterise(self.mu, self.log_var)
        y = self.decode(z)
        return y

class myDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            data = []
            for row in reader:
                if row:
                    data.append(row)

        # Put everything into a tensor and transpose
        self.data = torch.zeros((len(data),len(data[0])))
        for i in range(len(data)):
            for j in range(len(data[0])):
                self.data[i,j] = float(data[i][j])


    def __getitem__(self, idx):
        return self.data[idx,:]
    
    def __len__(self):
        return len(self.data)

def train():
    d = myDataset('material_data.csv')

    loader = torch.utils.data.DataLoader(d, batch_size=2048, shuffle=True)
    model = Autoencoder()

    load_mode = False
    if load_mode == True:
        model.load_state_dict(torch.load('autoencoder_weights.pth'))
    loss_fcn = torch.nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)

    epochs = 300
    outs = []
    losses = []

    for epoch in range(epochs):
        if epoch % 10 == 0: print("epoch:", epoch)
        for instance in loader:

            reconstructed = model(instance)
            loss = loss_fcn(reconstructed, instance)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.detach())
        outs.append((epochs, instance, reconstructed))

    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration"); plt.ylabel("loss"); plt.title("autoencoder performance")
    plt.show()

    # Save model weights
    torch.save(model.state_dict(), 'autoencoder_weights.pth')
    return model


def generate(model, n_samples=1):
    model.eval()

    with torch.no_grad():
        eps = torch.randn(n_samples, 4)
        generated = model.decode(eps)
    return generated.detach()


model = train()
poly_band = generate(model)
ks = list(np.linspace(-torch.pi, torch.pi, 1000 ))
p_eval = np.polyval(poly_band.T, ks)
plt.plot(ks, p_eval)
plt.show()

