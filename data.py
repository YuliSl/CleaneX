import numpy as np
from tensorflow import keras
np.random.seed(0)


def pairwise_squared_euclidean(X1, X2):
    """Calculates pairwise squared euclidean distances between the points in `X1` and `X2`.
    Uses a simple algebraic trick to save memory during calculation.
    """
    D = (np.square(X1).sum(axis=1) - 2*X2.dot(X1.T)).T + np.square(X2).sum(axis=1)
    D[D < 0] = 0  # fix precision errors around 0 (to avoid later errors with sqrt)
    
    return D


# ~~~~~ Synthetic data ~~~~~ #

def multivariate_scaled_uniform(loc, scale, size):
    """Samples points uniformly from a hypercube of edge length 2*sqrt(3*`scale`) centered at `loc`.
    """
    d = len(loc)
    X = np.sqrt(3*scale)*(2*np.random.random((size, d)) - 1) + loc
    return X

def generate_synthetic_data(k, d, r, sigma2, x_dist, y_dist):
    """Generates synthetic `d`-dimensional data with `k` classes and `r` points per class.
    Class centroids are distributed according to `y_dist`:
        'normal': standard multivariate normal;
        'uniform': uniformly on the interval [-sqrt(3), sqrt(3)] on each axis.
    Points of each class are distributed according to `x_dist`:
        'normal': multivariate normal centered at the class centroid with a diagonal covariance
            matrix, whose constant entry is `sigma2`;
        'uniform': uniformly on the hypercube centered at the class centroid with edge length
            2*sqrt(3*`sigma2`).
    Returns a tuple with the data points and the centroids.
    """
    if y_dist == 'normal':
        centroids = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=k)
    elif y_dist == 'uniform':
        centroids = multivariate_scaled_uniform(np.zeros(d), 1, size=k)

    if x_dist == 'normal':
        X = np.vstack([np.random.multivariate_normal(y, sigma2*np.eye(d), size=r) for y in centroids])
    elif x_dist == 'uniform':
        X = np.vstack([multivariate_scaled_uniform(y, sigma2, size=r) for y in centroids])
        
    return X, centroids

def calculate_synthetic_scores(k, d, r, sigma2, x_dist, y_dist):
    """Generates synthetic data according to the given parameters (see `generate_synthetic_data`)
    and calculates the score matrix `S`, where rows correspond to data points and columns to classes.
    The score of point x w.r.t. class y is the euclidean distance between x and the centroid of y.
    Returns a tuple with the score matrix `S` and the true labels of the data points `Y`.
    """
    # Generate data:
    X, centroids = generate_synthetic_data(k, d, r, sigma2, x_dist, y_dist)
    Y = np.hstack([[y]*r for y in range(k)])  # generate list of labels
    
    # Calculate distances to centroids:
    S = np.sqrt(pairwise_squared_euclidean(X, centroids))
    
    return S, Y


# ~~~~~ CIFAR-100 ~~~~~ #

def generate_CIFAR_embeddings():
    """Uses a VGG16 model, pre-trained on ImageNet, to generate a 512-dimensional embedding of
    CIFAR-100 images.
    Returns the embeddings and labels of the train and test sets.
    """
    # Load CIFAR-100 data:
    (X_train_raw, Y_train), (X_test_raw, Y_test) = keras.datasets.cifar100.load_data(label_mode='fine')

    # Load VGG16 model, pre-trained on ImageNet:
    vgg_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                               input_shape=X_train_raw.shape[1:])
    
    # Generate embeddings:
    X_train = vgg_model.predict(X_train_raw)
    X_test = vgg_model.predict(X_test_raw)
    
    # Reshape:
    X_test, X_train = np.squeeze(X_test), np.squeeze(X_train)
    Y_train, Y_test = Y_train.flatten(), Y_test.flatten()
    
    return X_train, Y_train, X_test, Y_test

def calculate_CIFAR_scores():
    """Loads CIFAR-100 data, generates a 512-dimensional embedding of the images (see
    `generate_CIFAR_embeddings`), and calculates the score matrix `S`, where rows correspond to images
    in the test set and columns to classes.
    The score of an image x w.r.t. class y is the euclidean distance between the embeddings of x
    and the closest train image from class y (that is, 1-nearest-neighbor).
    Returns a tuple with the score matrix `S` and the true labels of the test images `Y_test`.
    """
    # Generate embeddings:
    X_train, Y_train, X_test, Y_test = generate_CIFAR_embeddings()
    
    # Calculate distances to nearest neighbor of each class:
    S = []
    for y in range(100):
        X_train_y = X_train[Y_train == y]
        S_y = pairwise_squared_euclidean(X_test, X_train_y)
        S.append(np.min(S_y, axis=1, keepdims=True))
    S = np.sqrt(np.hstack(S))
    
    return S, Y_test
