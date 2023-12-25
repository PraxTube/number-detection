import numpy as np
import matplotlib.pyplot as plt


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


def main():
    train_images = np.fromfile("data/train/images", dtype=np.uint8)
    images = np.reshape(train_images[16:], (-1, 784))

    train_labels = np.fromfile("data/train/labels", dtype=np.uint8)
    labels = train_labels[8:]

    mean = empirical_mean(images[labels == 0][:100]).reshape((28, 28))

    image = images[1].reshape((28, 28))
    _ = plt.figure
    plt.imshow(mean, cmap="gray")
    plt.show()
    _ = plt.figure
    plt.imshow(image, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
