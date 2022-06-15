import numpy as np
import random
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from rich import print


def compute_bases(dim=2):
    # Make empty square matrix
    zeros = torch.zeros(dim, dim)
    # Make unit matrix
    unit = torch.eye(dim)
    # Compute basis
    bases = torch.cat((zeros, unit))

    return bases


def compute_linear_transformations(x, bases, dim=2):
    # Linear layer
    linear_layer = nn.Linear(dim, dim, bias=False)
    # Weights
    weights = linear_layer.weight
    # SVD decomposition
    u, s, v = compute_svd(weights)
    weights_determinant = np.linalg.det(weights.detach().numpy())
    if weights_determinant > 0:
        print(f'\nDeterminant < 0: {weights_determinant}\n'
              f'- U and V.T can both be rotations with/without reflections\n')
    elif weights_determinant < 0:
        print(f'\nDeterminant < 0: {weights_determinant}\n'
              f'- One of U and V.T will have a reflection\n')
    elif - 0.01 < weights_determinant < 0.01:
        print(f'\nDeterminant approx= 0: {weights_determinant}\n'
              f'- U and V.T can be either rotation or reflection independently\n')

    print('\n- [red]Weight matrix[/red]: ')
    print(weights)

    print('\n- U matrix: ')
    print(u)

    print('\n- S matrix: ')
    print(s)

    print('\n- V matrix: ')
    print(v)

    model = nn.Sequential(
        linear_layer
    )

    with torch.no_grad():
        y = model(x)
        y_bases = model(bases)

    return y, y_bases, u, s ,v


def compute_non_linear_transformations(x, bases, dim=2):
    # Linear layer
    linear_layer = nn.Linear(dim, dim, bias=False)
    # Weights
    weights = linear_layer.weight

    # List of most common activation functions
    common_activation_functions = [nn.Tanh(), nn.Sigmoid(), nn.ReLU()]
    # Getting one random activation function
    random_activation = random.choice(common_activation_functions)
    # Computing non-linear transformations on weights from previous layer
    squashed_weights = random_activation(weights)

    model = nn.Sequential(
        linear_layer,
        random_activation
    )

    linear_u, linear_s, linear_v = compute_svd(weights)
    u, s, v = compute_svd(squashed_weights)

    print('\n- [blue]Weight matrix[/blue]: ')
    print(weights)

    print('\n- U matrix after squashing: ')
    print(linear_u)

    print('\n- S matrix after squashing: ')
    print(linear_s)

    print('\n- V matrix after squashing: ')
    print(linear_v)

    print(f'\n- Weight matrix after {random_activation}: ')
    print(squashed_weights)

    print('\n- U matrix after squashing: ')
    print(u)

    print('\n- S matrix after squashing: ')
    print(s)

    print('\n- V matrix after squashing: ')
    print(v)

    with torch.no_grad():
        y = model(x)
        y_bases = model(bases)

    return y, y_bases, random_activation


def compute_svd(w):
    u, s, v = torch.svd(w)

    return u, s, v


def scatter_plot(x, colors, title: str, show=False):
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.axvline()
    plt.axhline()
    plt.title(title)

    if show:
        plt.show()


def plot_bases(bases, width=0.04, show=False):
    # *bases[0] for 2x2 matrix: tensor(0.) tensor(0.) instead of tensor([0., 1.])
    plt.arrow(*bases[0], *bases[2],
              width=width,
              color=(1, 0, 0),
              zorder=10,
              length_includes_head=True,
              label=f'{round(bases[2][0].item(), 3), round(bases[2][1].item(), 3)}')
    plt.arrow(*bases[1], *bases[3],
              width=width,
              color=(0, 1, 0),
              zorder=10,
              length_includes_head=True,
              label=f'{round(bases[3][0].item(), 3), round(bases[3][1].item(), 3)}')
    plt.legend()

    if show:
        plt.show()
        plt.close()


def main():
    n_points = 1000
    x = torch.randn(n_points, 2)
    bases = compute_bases()
    print('\n- Bases: ')
    print(bases)
    colors = x[:, 0]
    scatter_plot(x,
                 colors,
                 f'{len(x)} points from Gaussian dist.')
    plot_bases(bases, show=True)

    # Linear transform
    x_transform, bases_transform, u, s, v = compute_linear_transformations(x, bases)
    print('\n- Bases after linear transformation: ')
    print(bases_transform)

    # Plot SVD transformations (to explain what happens step by step)
    # - U transformation
    xu = x @ u
    xu = xu.detach().numpy()
    bases_u = bases @ u
    bases_u = bases_u.detach().numpy()
    scatter_plot(xu,
                 colors,
                 f'U transformation (SVD) [Rotation/Reflection]')
    plot_bases(bases_u, show=True)
    # - V transformation
    xv = x @ v.T
    xv = xv.detach().numpy()
    bases_v = bases @ v
    bases_v = bases_v.detach().numpy()
    scatter_plot(xv,
                 colors,
                 f'V.T transformation (SVD) [Rotation/Reflection]')
    plot_bases(bases_v, show=True)
    # - S transformation
    xs = x @ s
    xs = xs.detach().numpy()
    xs = xs[:, np.newaxis]
    xs = np.insert(xs, 1, x[:, 0], axis=1)
    xs = xs[:, ::-1]
    bases_s = bases @ s
    bases_s = bases_s.detach().numpy()
    bases_s = bases_s[:, np.newaxis]
    bases_s = np.insert(bases_s, 1, bases[:, 0], axis=1)
    bases_s = bases_s[:, ::-1]
    scatter_plot(xs,
                 colors,
                 f'S transformation (SVD) [Scaling]')
    plot_bases(bases_s, show=True)

    # Plot transformations altogether
    scatter_plot(x_transform,
                 colors,
                 f'Linear transformation')
    plot_bases(bases_transform, show=True)

    # Squashing
    x_squashed, bases_squashed, activation = compute_non_linear_transformations(x, bases)
    print('\n- Bases after non-linear transformation: ')
    print(bases_squashed)
    # Plot transformations
    scatter_plot(x_squashed,
                 colors,
                 f'Non-linear transformation with {activation}')
    plot_bases(bases_squashed, show=True)


if __name__ == '__main__':
    main()
