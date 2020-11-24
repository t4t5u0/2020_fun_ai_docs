import argparse
from datetime import datetime
from pathlib import Path
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# データローダの個数
# workers = 2
workers = 16

# バッチサイズ
# batch_size = 128
batch_size = 512

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# カラーチャンネル
nc = 3

# 潜在ベクトルの大きさ (i.e. size of generator input)
nz = 100

# ジェネレーターのフィーチャーマップの大きさ
ngf = 64

# ディスクリミネーターのフィーチャーマップの大きさ
ndf = 64

# トレーニングエポックの大きさ
num_epochs = 5

# オプティマイザの学習率
lr = 0.0002

# Adamのハイパーパラメータ
beta1 = 0.5

# GPUの個数. 0ならCPU
ngpu = 1


time = datetime.now()
param_path = f'./05/param/{time.year}{time.month:02}{time.day:02}{time.hour:02}{time.minute:02}'
writer = SummaryWriter(param_path)

# img_folder = Path.cwd() / "dataset"
img_folder = "/home/t4t5u0/Develop/2020_fun_ai_docs/05/dataset"
dataset = ImageFolder(
    img_folder,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=workers)

device = torch.device("cuda:0")
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
#            :64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.savefig("/home/t4t5u0/Develop/2020_fun_ai_docs/05/hoge.jpg")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 入力はz
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=ngf*8),
            nn.ReLU(inplace=True),

            # テンソルのサイズ (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # テンソルのサイズ　(ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # テンソルのサイズ (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # テンソルのサイズ　(ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # テンソルのサイズ (nc) x 64 x 64

        )

    def forward(self, x):
        return self.main(x)


# ジェネレーターをインスタンス化
netG = Generator(ngpu).to(device)

if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

# print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu) -> None:
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 入力 (nc) * 64 * 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)
print(netD)


# criterion = nn.BCELoss()
criterion = nn.MSELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr,
                        betas=(beta1, 0.999), weight_decay=1e-5)
optimizerG = optim.Adam(netG.parameters(), lr=lr,
                        betas=(beta1, 0.999), weight_decay=1e-5)


print('Starting Training Loop ...')


def train_GAN():
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    # for data, _ in tqdm(dataloader):
    for data in tqdm(dataloader):
        real_image = data[0].to(device)
        sample_size = real_image.size(0)
        # print(f'{sample_size=}')

        netD.zero_grad()
        # print(real_image.size())
        label = torch.full((sample_size,), real_label,
                           dtype=torch.float, device=device)
        # print(f'{label=}')
        output = netD(real_image).view(-1)
        # print(f'{output.size()=}')

        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(sample_size, nz, 1, 1, device=device)

        fake_image = netG(noise)
        label.fill_(fake_label)

        output = netD(fake_image.detach()).view(-1)
        errD_fake = criterion(output, label)

        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        D_losses.append(errD.item())

        optimizerD.step()

        ########################################

        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake_image).view(-1)

        errG = criterion(output, label)

        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()
        G_losses.append(errG.item())

        if epoch % 10 == 0:
            pass

        if writer is not None:
            writer.add_scalars('loss', {
                'D': D_losses[-1], 'G': G_losses[-1]
            }, epoch)


for epoch in range(1000):
    # train_dcgan(g, d, opt_g, opt_d, img_loader, writer)
    train_GAN()

    if epoch % 10 == 0:
        # パラメータの保存
        torch.save(
            netG.state_dict(),
            f"{param_path}/g_{epoch:04d}.prm",
            pickle_protocol=4
        )
        torch.save(
            netD.state_dict(),
            f"{param_path}/d_{epoch:04d}.prm",
            pickle_protocol=4
        )
        generated_img = netG(fixed_noise)
        # generated_img = g(fixed_z)
        save_image(generated_img,
                   f"{param_path}/{epoch:04d}.jpg")
