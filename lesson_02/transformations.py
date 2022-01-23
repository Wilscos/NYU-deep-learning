import torch
import torch.nn as nn
from matplotlib import pyplot as plt


def scatter_plot(x, colors):
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.title(f'{len(x)} points from Gaussian dist')
    plt.show()
    plt.close()


def main():
    n_points = 1000
    x = torch.randn(n_points, 2)
    colors = x[:, 0]
    scatter_plot(x, colors)


if __name__ == '__main__':
    main()
