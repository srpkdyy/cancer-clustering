import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchsummary import summary


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, downsample=None, bias=False):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
#            conv3x3(in_planes, planes, bias=bias),
#            nn.BatchNorm2d(planes),
#            nn.ReLU(True),
            ConvBlock(in_planes, planes, bias=bias),
            conv3x3(planes, planes, bias=bias),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        identify = x

        out = self.layers(x)

        if self.downsample is not None:
            identify = self.downsample(x)

        out += identify
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, same=False, bias=True):
        super(ConvBlock, self).__init__()

        if same:
            kernel_size = 3
            stride = 1
        else:
            kernel_size = 4
            stride = 2  

        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride,
                      padding=1, bias=bias),
            nn.BatchNorm2d(n_out),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvTBlock(nn.Module):
    def __init__(self, in_planes, out_planes, bias=True):
        super(ConvTBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)



class VAE(nn.Module):
    def __init__(self, in_channels=3, h_dims: list = None, z_dim=32):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim

        if h_dims is None:
            h_dims = [32, 64, 128, 256, 256, 512, 512]

        # build encoder
        encoder = []
        for h_dim in h_dims:
            encoder.append(ResBlock(in_channels, h_dim,
                           downsample=nn.Conv2d(in_channels, h_dim,
                           kernel_size=4, stride=2, padding=1, bias=False)))
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder)

        width = 256 // (2 ** len(h_dims))
        self.fc_mu = nn.Linear(h_dims[-1] * width**2, z_dim)
        self.fc_logvar = nn.Linear(h_dims[-1] * width**2, z_dim)
        self.fc = nn.Linear(z_dim, h_dims[-1] * width**2)

        in_channels = h_dims[-1]
        del h_dims[-1]
        h_dims.reverse()
        
        # build decoder
        decoder = []
        for h_dim in h_dims:
            decoder.append(ConvTBlock(in_channels, h_dim))
            in_channels = h_dim
        decoder.extend([
            nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        ])
        self.decoder = nn.Sequential(*decoder)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, a=1e-2, mode='fan_out')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, n_samples: int, device):
        z = torch.randn(n_samples, self.z_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples


if __name__ == "__main__":
    img = (3, 256, 256)
    model = VAE()
    summary(model, img, device='cpu')
