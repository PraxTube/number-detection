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


def covariance_eigenvectors(images):
    X = images
    means = empirical_mean(X)
    Y_T = X - means
    Y = Y_T.T

    U, _, _ = np.linalg.svd(Y)
    return U.T


def project_to_subspace(A, b, X):
    t = A @ (X.T - b[:, np.newaxis])
    return t.T


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


def plot_pvs(images, labels):
    X = images[:1000]
    means = empirical_mean(X)
    Y_T = X - means
    Y = Y_T.T
    N = 50

    E = np.real(np.linalg.eig(Y @ Y_T)[0])[:N]
    _, raw_S, _ = np.linalg.svd(Y)
    S = np.power(raw_S, 2)[:N]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    ax = axes[0]
    ax.scatter(range(N), E)
    ax.set_title("Real Eigenvalues of S")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True)

    ax = axes[1]
    ax.scatter(range(N), S)
    ax.set_title("Squared Singular values")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_subspace(images, labels):
    X = images[:1000]
    b = empirical_mean(X)
    S = covariance_eigenvectors(X)
    A = S[:5]

    fig, axes = plt.subplots(1, 6, figsize=(14, 8))
    images = np.vstack((b, A))
    titles = (
        "Mean, b",
        "1. column vector of A",
        "2. column vector of A",
        "3. column vector of A",
        "4. column vector of A",
        "5. column vector of A",
    )

    for i in range(6):
        ax = axes[i]
        ax.imshow(images[i].reshape((28, 28)), cmap="gray_r")
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_four_images(images, labels):
    X = images[:1000]
    indices = np.array((69, 420, 500, 1500))
    b = empirical_mean(X)
    A = covariance_eigenvectors(X)[:5]

    t = project_to_subspace(A, b, images[indices])

    fig, axes = plt.subplots(2, len(indices), figsize=(14, 8))
    for i in range(len(indices)):
        ax = axes[0, i]
        ax.imshow(images[indices[i]].reshape((28, 28)), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, i]
        ax.imshow(t[i].reshape(1, -1), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_k_means(images, labels, test_images, test_labels):
    K = 2
    max_iterations = 100
    first_images = images[labels == 0][:500]
    second_images = images[labels == 1][:500]
    X = np.vstack((first_images, second_images))

    b = empirical_mean(X)
    A = covariance_eigenvectors(X)[:2]

    t = project_to_subspace(A, b, X)
    centers = np.array(((5, 1), (1, 5)))

    for _ in range(max_iterations):
        C = [[] for _ in range(K)]

        for i in range(len(X)):
            distances = np.linalg.norm(t[i] - centers, axis=1)
            closest_center = np.argmin(distances)
            C[closest_center].append(i)

        for k in range(K):
            if len(C[k]) > 0:
                cluster_points = t[C[k]]
                centers[k] = np.mean(cluster_points, axis=0)

    t_0 = project_to_subspace(A, b, test_images[test_labels == 0][:100])
    t_1 = project_to_subspace(A, b, test_images[test_labels == 1][:100])

    fig, axes = plt.subplots(1, 1, figsize=(14, 8))

    ax = axes
    ax.scatter(t_0[:, 0], t_0[:, 1])
    ax.scatter(t_1[:, 0], t_1[:, 1])
    ax.scatter(centers[:, 0], centers[:, 1])
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    train_images = np.fromfile("data/train/images", dtype=np.uint8)
    images = np.reshape(train_images[16:], (-1, 784))
    train_labels = np.fromfile("data/train/labels", dtype=np.uint8)
    labels = train_labels[8:]

    test_images = np.fromfile("data/test/images", dtype=np.uint8)
    t_images = np.reshape(test_images[16:], (-1, 784))
    test_labels = np.fromfile("data/test/labels", dtype=np.uint8)
    t_labels = test_labels[8:]

    plot_k_means(images, labels, t_images, t_labels)


if __name__ == "__main__":
    main()
