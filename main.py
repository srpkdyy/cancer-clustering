import argparse
import datetime
import numpy as np
import torch
import torch.utils as utils
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, datasets
from torchvision.utils import save_image

from vae import VAE


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('-z', '--z-dim', type=int, default=64)
parser.add_argument('--out', type=str, default='latest')
parser.add_argument('--interval', type=int, default=10)
args = parser.parse_args()

torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')




img_width = 256
normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

train_transforms = transforms.Compose([
    transforms.Resize(img_width),
    transforms.CenterCrop(img_width),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
])
'''
train_ds = datasets.STL10(
    './data/examples',
    split='train',
    download=False,
    transform=train_transforms
)

test_transforms = transforms.Compose([
    transforms.Resize(img_width),
    transforms.CenterCrop(img_width),
    transforms.ToTensor(),
    normalize
])
test_ds = datasets.STL10(
    './data/examples',
    split='test',
    download=False,
    transform=test_transforms
)
'''
ds = datasets.ImageFolder(
    './dataset/all',
    train_transforms
)

kwargs = {
    'num_workers': 4,
    'pin_memory': True
} if use_cuda else {}
'''
train_loader = data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = data.DataLoader(
    test_ds,
    batch_size=args.batch_size,
    shuffle=False,
    **kwargs
)
'''
data_loader = data.DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
'''
t = data_loader.__iter__()
x, _ = t.next()
print(x[0])
print(x[0].max())
print(x[0].min())
'''
model = VAE(z_dim=args.z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.8)], gamma=0.1) # 10 < 20 < 40


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()))
    
    return MSE, KLD


def train(epoch):
    model.train()
    train_loss = 0
    mse_loss = 0
    kld_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        mse, kld = loss_function(recon_batch, data, mu, logvar)
        loss = mse + kld
        loss.backward()
        optimizer.step()

        mse_loss += mse
        kld_loss += kld
        train_loss += loss
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))
    return mse_loss / len(data_loader.dataset), kld_loss / len(data_loader.dataset)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':

    mse_loss = list()
    kld_loss = list()
    for epoch in range(1, args.epochs + 1):
        mse, kld = train(epoch)
        mse_loss.append(mse.cpu().detach().clone().numpy())
        kld_loss.append(kld.cpu().detach().clone().numpy())
        scheduler.step()
        #test(epoch)
        if epoch % args.interval == 0:
            with torch.no_grad():
                sample = model.sample(64, device=device).cpu()
                sample = 0.5 * (sample + 1)
                sample = sample.clamp(0, 1)
                save_image(sample, 'result/recon/sample_' + str(epoch) + '.png')
    loss = np.array([mse_loss, kld_loss])
    np.save('result/loss/' + args.out, loss)

    features = list()
    labels = list()
    model.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            features.extend(mu.cpu().clone().tolist())
            labels.extend(label.tolist())
    features = np.array([x for x in zip(np.array(features), labels)])
    np.save('result/features/' + args.out, features)
        