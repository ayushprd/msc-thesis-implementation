import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp
from scipy.spatial import distance
import numpy as np
from jax import random
from jax.nn import softplus, sigmoid
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from gbmap.embedding.gbnn_four import GBMAP4


def mse(y, y_pred):
    return jnp.mean((y - y_pred) ** 2)


@jit
def pairwise_distances(X):
    return jnp.sqrt(jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))


@jit
def log_normalise(x):
    """Find normalised log-probabilities in a numerically stable manner."""
    return x - logsumexp(x)


def drop_matrix_diagonal(m):
    """Takes in pXp square matrix and outputs pX(p-1) matrix where the diagonal elements
    have been dropped from each row."""

    # def f(i, x):
    #    return jnp.concatenate((x[:i], x[(i + 1) :]))

    # return jnp.array([f(i, m[i, :]) for i in range(m.shape[0])])

    n = m.shape[0]

    return m.ravel()[jnp.arange(n * n) % (n + 1) != 0].reshape((n, n - 1))


@jit
def kernel_linear(P, Q):
    """Linear kernel."""
    return P @ Q.T


@jit
def kernel_rbf(P, Q, sigma=1.0):
    """RBF kernel."""
    return jnp.exp(
        -jnp.sum((P[:, None, :] - Q[None, :, :]) ** 2, axis=2) / (2.0 * sigma**2)
    )


@jit
def kernel_polynomial(
    P,
    Q,
    degree=2.0,
    coef=1.0,
    const=1.0,
):
    """Polynomial kernel with degree d, coef c, and constant a

    K(P,Q) = (c * <P,Q> + a)^d

    Args:
        P (array like): feature array 1
        Q (array like): feature array 2
        degree (float, optional): Polynomial degree. Defaults to 2.0.
        coef (float, optional): Coefficient scaling the inner product. Defaults to 1.0.
        const (float, optional): Constant added to the scaled inner product. Defaults to 1.0.

    Returns:
        array like: kernel output
    """

    return (coef * (P @ Q.T) + const) ** degree


def shuffle_matrix(m, idx):
    return m.ravel()[idx].reshape(m.shape)


def efficient_squared_dist(X, Y=None):
    """Calculate squared Euclidean distances efficiently between the row vectors of matrice(s).

    The distances are calculated using the following formula:
        dist(x, y) = dot(x, x) - 2 * dot(x, y) + dot(y, y)

    Args:
        X (array): Array 1 of row vectors
        Y (array, optional): Array 2 of row vectors. Defaults to None.

    Returns:
        array: (n,m) distance matrix.
    """
    if Y is None:
        # did not specify Y, calculate distances for rows of X
        x2 = y2 = jnp.sum(X**2, axis=1)
        Y = X
    else:
        x2 = jnp.sum(X**2, axis=1)
        y2 = jnp.sum(Y**2, axis=1)

    xy = jnp.matmul(X, Y.T)
    # sum each row of xy with x2[i] and each column of xy with y2[j]
    # to do the above we need to reshape x2 -> (n, 1)
    x2 = x2.reshape(-1, 1)
    d2 = x2 - 2 * xy + y2
    return jnp.maximum(d2, 0.0)  # clip possible negative values


def gaussian_smoother(x0, x, y, s=1):
    """Non-parametric estimator for f(x) = y, calculates Gaussian weighted average of y in the neighborhood of x.

    Args:
        x0 (array like): New data points we want to infer y from, assumed to be two dimensional array.
        x (array like): Train data points, assumed to be two dimensional array.
        y (array like): Train target values.
        s (int, optional): radius for the Gaussia kernel (sigma). Defaults to 1.

    Returns:
        array like: predicted y hat for each data point.
    """

    # compute squared distances for x0, x pairs
    sqdist = jnp.sum((x0[:, None, :] - x[None, :, :]) ** 2, axis=2)
    # gaussian kernel
    K = jnp.exp(-sqdist / (2 * s**2))
    # normalize the rows to get a kernel weight for each x position
    K_norm = K / K.sum(axis=1)[:, None]
    # compute kernel weighed average
    yhat = (K_norm * y).sum(axis=1)
    return yhat


def nebiloss(yhat, y):
    """negative binomial log-likelihood loss L(yhat, y) (for classification)

    Args:
        yhat (array like): prediction values
        y (array like): target labels

    Returns:
        array like: negative binomial log-likelihood loss between yhat and y
    """

    return jnp.logaddexp(0.0, -2 * yhat * y)


def grad_nebiloss(yhat, y):
    """gradient of negative binomial log-likelihood loss \partial_yhat L(yhat, y) (for classification)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: gradient of the negative binomial log-likelihood loss between yhat and y
    """
    return -2 * y / (1 + jnp.exp(2 * yhat * y))


def logisticloss(yhat, y):
    """logistic loss (variant of nebiloss)

    Args:
        yhat (array like): prediction values
        y (array like): true labels

    Returns:
        array like: logistic loss between yhat and y
    """
    return jnp.logaddexp(0.0, -yhat * y)


def grad_logisticloss(yhat, y):
    """gradient of logistic loss \partial_yhat L(yhat, y) (for classification)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: gradient of the logistic loss between yhat and y
    """
    return -y / (1 + jnp.exp(yhat * y))


def adaloss(yhat, y):
    """ada loss L(yhat, y) (for classification)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: ada loss between yhat and y
    """
    return jnp.exp(-yhat * y)


def grad_adaloss(yhat, y):
    """gradient of ada loss \partial_yhat L(yhat, y) (for classification)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: gradient of ada loss between yhat and y
    """
    return -y * jnp.exp(-yhat * y)


def lcloss(yhat, y):
    """log-cosh loss for regression (should have a more robust implementation)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: log-cosh loss between yhat and y
    """
    return jnp.log(jnp.cosh(yhat - y))


def grad_lcloss(yhat, y):
    """gradient of log-cosh loss (for regression)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: gradient of log-cosh loss between yhat and y
    """
    return jnp.tanh(yhat - y)


def squared_loss(yhat, y):
    """squared loss for regression (should have a more robust implementation)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: squared loss between yhat and y
    """
    return 0.5 * (yhat - y) ** 2


def grad_squared_loss(yhat, y):
    """gradient of squared loss (for regression)

    Args:
        yhat (array like): predicted values
        y (array like): true labels

    Returns:
        array like: gradient of squared loss between yhat and y
    """
    return yhat - y


def set_distance(reference_points, sample_points, metric="euclidean", stat="min"):
    """Distance between each sample point and the set of reference points

    Args:
        reference_points (array-like): a set of reference points (aka. training data set)
        sample_points (array-like): a set of sample points (aka. test data set)

    Returns:
        array: array of distances between each test point and the training data set
    """
    C = distance.cdist(sample_points, reference_points, metric=metric)

    if stat == "min":
        dis = np.min(C, axis=1)
    elif stat == "mean":
        dis = np.mean(C, axis=1)
    else:
        raise ValueError("stat not implemented.")
    return dis


def sort_data_and_labels(x, y, column_index):
    """sort a data set (x, y) with respect to a column of x

    Args:
        x (array-like): feature matrix
        y (array-like): vector of labels
        column_index (non-negative integer): the column index of x

    Returns:
        array-like: sorted data set
    """
    combined_data = list(zip(x, y))
    sorted_combined_data = sorted(combined_data, key=lambda pair: pair[0][column_index])

    sorted_x, sorted_y = zip(*sorted_combined_data)

    return np.array(sorted_x), np.array(sorted_y)


def drift_split(x, y, test_size=0.2, drop=True, splitter=None, clasification=False):
    """split data set (x, y) that induces drift

    Args:
        x (array-like): features
        y (array-like): labels
        test_size (float, optional): test size. Defaults to 0.2.
        drop (bool, optional): drop the feature that was used to split the data set. Defaults to True.

    Returns:
        array-like: train-test split that induces drift
    """

    if splitter is None:
        if clasification:
            raise ValueError("not yet implemeted for classification.")
        test_error = []
        for i in range(x.shape[1]):
            # train-test split based on the i-th feature
            x_sort, y_sort = sort_data_and_labels(x, y, column_index=i)
            x_train, x_test, y_train, y_test = train_test_split(
                x_sort, y_sort, test_size=test_size, shuffle=False
            )

            # train a model and get test error
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            y_test_pred = lr.predict(x_test)
            test_error.append(mse(y_test, y_test_pred))

        test_error = np.array(test_error)
        best_idx = test_error.argmax()
        x_sort, y_sort = sort_data_and_labels(x, y, column_index=best_idx)
        x_train, x_test, y_train, y_test = train_test_split(
            x_sort, y_sort, test_size=test_size, shuffle=False
        )

        if drop:
            x_train = np.delete(x_train, best_idx, axis=1)
            x_test = np.delete(x_test, best_idx, axis=1)
    elif splitter == "rf":
        rf = (
            RandomForestClassifier(random_state=42)
            if clasification
            else RandomForestRegressor(random_state=42)
        )
        rf.fit(x, y)
        rf_feature_importances = rf.feature_importances_
        rf_max_idx = rf_feature_importances.argmax()
        x_sort, y_sort = sort_data_and_labels(x, y, column_index=rf_max_idx)
        x_train, x_test, y_train, y_test = train_test_split(
            x_sort, y_sort, test_size=test_size, shuffle=False
        )

        if drop:
            x_train = np.delete(x_train, rf_max_idx, axis=1)
            x_test = np.delete(x_test, rf_max_idx, axis=1)

    else:
        raise ValueError("splitter not implemeted.")

    return x_train, x_test, y_train, y_test


def add_irrelevant_features(
    x_train, x_test, n_noise_dim=0, noise_scale=1, seed=42, rotation=True
):
    """add irrelevant features to a data set

    Args:
        x_train (array-like): training features
        x_test (array-like): test features
        n_noise_dim (int, optional): number of irrelevant features. Defaults to 0.
        noise_scale (int, optional): scale of irrelevant features. Defaults to 1.
        seed (int, optional): random seed when generate irrelevant features. Defaults to 42.
        rotation (bool, optional): rotation the entire data set at the end. Defaults to True.

    Returns:
        array-like: new data set with added irrelevant features and the rotation matrix used (if applied)
    """
    np.random.seed(seed=seed)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    # include noise
    x_train = np.hstack(
        [
            x_train,
            noise_scale
            * np.random.normal(loc=0, scale=1, size=(train_size, n_noise_dim)),
        ]
    )
    x_test = np.hstack(
        [
            x_test,
            noise_scale
            * np.random.normal(loc=0, scale=1, size=(test_size, n_noise_dim)),
        ]
    )

    # rotation
    Q = None
    if rotation:
        random_matrix = np.random.rand(x_train.shape[1], x_train.shape[1])
        Q, R = np.linalg.qr(random_matrix)
        x_train = x_train @ Q
        x_test = x_test @ Q

    return x_train, x_test, Q


# Construct a random unit vector of dimensionality p
def random_unit_vector(p, key):
    while True:
        u = random.normal(key, (p,))
        s = jnp.sqrt(jnp.sum(u**2))
        if s > 0.0:
            return u / s
        key = random.split(key, 1)


def F1max(ind: np.ndarray, error: np.ndarray, thr_error: float):
    """Find max F1 score and corresponding indicator threshold.

    Args:
        ind: Drift indicator
        error: Error
        thr_error: Threshold: if error >= thr_error then drift.
    """

    def _F1(tp, fp, fn):
        return np.where(
            (tp == 0) & (fp == 0) & (fn == 0),
            0.0,
            tp / (tp + 0.5 * (fp + fn)),
        )

    # For given thr_ind, tp = sum((ind >= thr_ind) & (e >= thr_e)).
    # If we sort e by ind in descending order, we get tp(thr_ind) = cumsum(e >= thr_e).
    o = np.argsort(-ind)
    has_drift = error[o] >= thr_error
    tp = np.cumsum(has_drift)
    fp = np.cumsum(~has_drift)
    fn = np.sum(has_drift) - tp
    F1 = _F1(tp, fp, fn)
    i_max = np.argmax(F1)
    return F1[i_max], ind[o][i_max]


# Logistic loss
def loss_logistic(y, yp):
    return jnp.average(softplus(-y * yp))


# Logit
def logit(p):
    return jnp.log(p / (1.0 - p))


# Transfer real y values to binary values
def to_binary(y, key):
    return jnp.select([random.uniform(key, y.shape) <= sigmoid(y)], [1.0], default=-1.0)


def knnpred(Ztest, Z, y, k=5, average=jnp.average):
    # Compute all L1 distances from test points to the training data
    distances = jnp.sum(jnp.abs(Ztest[:, None, :] - Z[None, :, :]), axis=2)
    # Find indices of k nearest neighbours
    kni = jnp.argsort(distances, axis=1)[:, :k]
    # return averaged target values
    return vmap(lambda i: average(y[i]))(kni)


# average function used, e.g., by knnpred for classification problems where y is a vector of target labels -1 and +1.
def caverage(y, pseudocount=1.0):
    return logit(
        (jnp.sum(0.5 + 0.5 * y) + 0.5 * pseudocount) / (y.shape[0] + pseudocount)
    )


def accuracy(y, yp):
    return jnp.average(y * yp >= 0.0)


# Quadratic loss
def loss_quadratic(y, yp):
    return jnp.average((y - yp) ** 2)


# Split dataset into two roughly equally sized halves by a given feature (a and b).
# Then split these halves into two in random (1 and 2), resulting to four blocks in total (a1, a2, b1, b2).
def split4(key, X, feature=0):
    key_a, key_b = random.split(key)
    i = jnp.argsort(X[:, feature])
    ia = i[: (X.shape[0] // 2)]
    ib = i[(X.shape[0] // 2) :]
    ia1 = random.choice(key_a, ia, (ia.shape[0] // 2,), replace=False)
    ia2 = jnp.setdiff1d(ia, ia1)
    ib1 = random.choice(key_b, ib, (ib.shape[0] // 2,), replace=False)
    ib2 = jnp.setdiff1d(ib, ib1)
    return ia1, ia2, ib1, ib2


# See if splitting by a feature induced concept drift
def test_feature(key, X, y, feature=0, m=10, loss=loss_quadratic):
    ia1, ia2, ib1, ib2 = split4(key, X, feature)
    X = jnp.delete(X, (feature), axis=1)
    if loss.__name__ == "loss_logistic":
        is_classifier = True
    else:
        is_classifier = False
    gbmap = GBMAP4(
        n_boosts=m, random_state=49, optim_maxiter=200, is_classifier=is_classifier
    )
    gbmap.fit(X[ia1, :], y[ia1])
    yhat = gbmap.predict(X, get_score=True)
    return (
        loss(y[ia1], yhat[ia1]).item(),
        loss(y[ia2], yhat[ia2]).item(),
        loss(y[ib1], yhat[ib1]).item(),
        loss(y[ib2], yhat[ib2]).item(),
    )


def find_variables(key, X, y, jrange=None, loss=loss_quadratic):
    if jrange is None:
        jrange = range(X.shape[1] - 1)
    drift_list = []
    for j in jrange:
        _, la2, lb1, lb2 = test_feature(key, X, y, feature=j, loss=loss)
        loss_a = la2
        loss_b = (lb1 + lb2) / 2.0
        drift = loss_b - loss_a
        drift_list.append(drift)
    drift_list = jnp.array(drift_list)
    return drift_list.argmax()


# Compute various statistics on classification results
# Not use !
# def findbestf1(y, v=None):
#     if v is not None:
#         y = y[jnp.argsort(v)]

#     # Make sure y is in reverse order
#     y = jnp.flip(y)

#     ny = jnp.logical_not(y)

#     tp = jnp.concatenate((jnp.zeros(1), jnp.cumsum(y)))
#     fp = jnp.concatenate((jnp.zeros(1), jnp.cumsum(ny)))

#     tpr = tp / jnp.sum(y)
#     fpr = fp / jnp.sum(ny)

#     auc = jnp.sum((fpr[1:] - fpr[:-1]) * tpr[1:])

#     fn = jnp.sum(y) - tp
#     tn = jnp.sum(ny) - fp

#     f1 = 2.0 * tp / (2.0 * tp + fp + fn)

#     return tpr, fpr, f1, tp, fp, fn, tn, auc