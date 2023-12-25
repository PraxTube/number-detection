import numpy as np
import matplotlib.pyplot as plt


N = 10
N_row = 2
N_col = 5


def empirical_mean(X):
    """Calculate mean of X, it is expected that X is of shape d x D,
    meaning the data is in C-major matrix style."""
    return X.mean(axis=0)


def empirical_variance(X):
    """Calculate the empirical variance of X.
    It is expected that the data points are row vectors in X,
    meaning that the matrix uses C-major style."""
    b = empirical_mean(X)
    Y = np.power(X - b, 2)
    return Y.mean(axis=0)


def plot_means(images, labels):
    means = np.zeros((10, 28, 28))
    for i in range(N):
        means[i] = empirical_mean(images[labels == i][:100]).reshape((28, 28))

    fig, axes = plt.subplots(N_row, N_col, figsize=(1.5 * N_col, 2 * N_row))
    for i in range(N):
        ax = axes[i // N_col, i % N_col]
        ax.imshow(means[i], cmap="gray_r")
        ax.set_title("Label: {}".format(i))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_variances(images, labels):
    variances = np.zeros((10, 28, 28))
    for i in range(N):
        variances[i] = empirical_variance(images[labels == i][:100]).reshape((28, 28))

    fig, axes = plt.subplots(N_row, N_col, figsize=(1.5 * N_col, 2 * N_row))
    for i in range(N):
        ax = axes[i // N_col, i % N_col]
        ax.imshow(variances[i], cmap="gray_r")
        ax.set_title("Label: {}".format(i))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def main():
    train_images = np.fromfile("data/train/images", dtype=np.uint8)
    images = np.reshape(train_images[16:], (-1, 784))

    train_labels = np.fromfile("data/train/labels", dtype=np.uint8)
    labels = train_labels[8:]

    plot_means(images, labels)
    plot_variances(images, labels)


if __name__ == "__main__":
    main()
