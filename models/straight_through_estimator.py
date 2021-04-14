# https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


# defining networks

class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 4),
                      stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(4, 4),
                      stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(4, 4),
                      stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=(4, 4),
                      stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            StraightThroughEstimator(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(
                5, 5), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=(
                5, 5), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=(
                4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=(
                4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, x, encode=False, decode=False):
        if encode:
            x = self.encoder(x)
        elif decode:
            x = self.decoder(x)
        else:
            encoding = self.encoder(x)
            x = self.decoder(encoding)
        return x


def train_code():

    from tqdm import tqdm
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    net = Autoencoder().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion_MSE = nn.MSELoss().to(device)

    trainloader, testloader = prep_data()

    # train loop

    epoch = 5
    for e in range(epoch):
        print(f'Starting epoch {e} of {epoch}')
        
        # for X, y in trainloader:
        for X, y in tqdm(trainloader):
            optimizer.zero_grad()
            X = X.to(device)
            reconstruction = net(X)
            loss = criterion_MSE(reconstruction, X)
            loss.backward()
            optimizer.step()
        print(f'Loss: {loss.item()}')

    # test loop

    i = 1
    fig = plt.figure(figsize=(10, 10))

    for X, y in testloader:
        X_in = X.to(device)
        recon = net(X_in).detach().cpu().numpy()

        if i >= 10:
            break

        fig.add_subplot(5, 2, i).set_title('Original')
        plt.imshow(X[0].reshape((28, 28)), cmap="gray")
        fig.add_subplot(5, 2, i+1).set_title('Reconstruction')
        plt.imshow(recon[0].reshape((28, 28)), cmap="gray")

        i += 2
    fig.tight_layout()
    plt.show()


# dataset preparation
def prep_data():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    trainset = datasets.MNIST('dataset/', train=True,
                            download=True, transform=transform)
    testset = datasets.MNIST('dataset/', train=False,
                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True)
    return trainloader, testloader

if __name__ == "__main__":
    train_code()
