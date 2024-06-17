from torch import nn

class Autoencoder_instance(nn.Module):
    def __init__(self, Input_shape):
        super(Autoencoder_instance, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(Input_shape, 4096),
            # nn.BatchNorm1d(4096*2),
            nn.InstanceNorm1d(4096),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            # nn.BatchNorm1d(4096),
            nn.InstanceNorm1d(2048),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(2048, 1024),

        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            # nn.BatchNorm1d(4096),
            nn.InstanceNorm1d(2048),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(2048, 4096),
            # nn.BatchNorm1d(4096*2),
            nn.InstanceNorm1d(4096),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(4096, Input_shape),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded