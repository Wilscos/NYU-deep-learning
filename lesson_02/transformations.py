import random
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


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

    print('\n- Weight matrix: ')
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

    return y, y_bases


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

    print('\n- Weight matrix: ')
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

    return y, y_bases


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
    x_transform, bases_transform = compute_linear_transformations(x, bases)
    print('\n- Bases after linear transformation: ')
    print(bases_transform)
    # Plot transformations
    scatter_plot(x_transform,
                 colors,
                 f'Linear transformation')
    plot_bases(bases_transform, show=True)

    # Squashing
    x_squashed, bases_squashed = compute_non_linear_transformations(x, bases)
    print('\n- Bases after non-linear transformation: ')
    print(bases_squashed)
    # Plot transformations
    scatter_plot(x_squashed,
                 colors,
                 f'Non-linear transformation')
    plot_bases(bases_squashed, show=True)


if __name__ == '__main__':
    main()
