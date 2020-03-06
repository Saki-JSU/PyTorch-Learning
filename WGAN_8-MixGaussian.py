# an example for GAN: learning 8-centers mixture Gaussian distribution
import torch
from torch import nn, optim, autograd
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt
import random

# global parameter
h_dim = 400
batchsz = 512

# network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )

    def forward(self, z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

# data generator
def data_generator():
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    # this loop is cool
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(batchsz):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev
        yield dataset

# plot
def generate_image(D, G, xr):
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda()  # [16384, 2]
        disc_map = D(points).cpu().numpy()  # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)

    # draw samples
    with torch.no_grad():
        z = torch.randn(batchsz, 2).cuda()  # [b, 2]
        samples = G(z).cpu().numpy()  # [b, 2]
        xr = xr.cpu().numpy()
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
    plt.show()

# initialized weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

# Wasserstein GAN penalty
def gradient_penalty(D, xr, xf):
    LAMBDA = 0.3

    # only constraint for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 2]
    alpha = torch.rand(batchsz, 1).cuda()
    alpha = alpha.expand_as(xr)

    mid = alpha * xr + ((1 - alpha) * xf)
    mid.requires_grad_()

    pred = D(mid)

    gradients = autograd.grad(outputs=pred, inputs=mid,
                              grad_outputs=torch.ones_like(pred),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp

# main function
def main():
    # random seed
    torch.manual_seed(23)
    np.random.seed(23)

    # create network and initialization
    G = Generator().cuda()
    D = Discriminator().cuda()
    G.apply(weights_init)
    D.apply(weights_init)

    # optimizer
    optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    # generate data
    data_iter = data_generator()

    for epoch in range(5000):

        # 1. train discriminator for k steps
        for _ in range(5):
            # train on real data
            x = next(data_iter)
            xr = torch.from_numpy(x).cuda()
            predr = D(xr)
            # max log(lossr)
            lossr = -predr.mean()

            # train on fake data
            z = torch.randn(batchsz, 2).cuda()
            xf = G(z).detach()
            predf = D(xf)
            # min predf
            lossf = predf.mean()

            # gradient penalty
            gp = gradient_penalty(D, xr, xf)

            # loss of Discriminator
            loss_D = lossr + lossf + gp
            # training
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()


        # 2. train Generator
        z = torch.randn(batchsz, 2).cuda()
        xf = G(z)
        predf = D(xf)
        # max predf
        loss_G = -predf.mean()
        # training
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        # print
        if epoch % 100 == 0:
            print(loss_D.item(), loss_G.item())

    # plot
    generate_image(D, G, xr)


if __name__ == '__main__':
    main()

