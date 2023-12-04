import numpy as np


def get_lids(dist, k=100):
    """
    Estimating local intrinsic dimensionality via
    the MLE method (Amsaleg et al., 2015)

    Parameters
    ----------
    dist : array-like, shape(n_queries, n_neighbors)
        Where n_queries refers to the number of points
        (queries); and n_neighbors refers to the number
        of closest neighbors of each point.

    k : int, default=100
        Determines how many nearest neighbors
        must be evaluated to estimate the LID.
        It can be less or equal to the number of
        neighbors in the `dist` matrix.

    Returns
    -------
    lids : array-like, shape(N,)
        The local intrinsic dimensionality of
        each element from the dist matrix.
    """

    dist = dist[:, 1 : k + 1]
    lids = np.log(dist[:, 0 : k - 1] / dist[:, k - 1 : k])
    lids = lids.sum(axis=1) / (k)
    lids = -(1.0 / lids)
    return lids


def get_pcs(X, variance=0.9):
    """
    Returns the first `n` PCs that explain
    a desired variance.

    Parameters
    ----------
    X : array-like, shape(n_samples, n_features)
        The dataset.

    variance : float, default=0.9
        The desired explained variance.

    Returns
    -------
    n_pcs : int
        Number of PCs that explain the `variance`.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    sc = StandardScaler()
    pca = PCA()
    pca.fit(sc.fit_transform(X))
    variance = sorted(pca.explained_variance_ratio_, reverse=True)
    total = 0
    for n_pcs, var in enumerate(variance):
        total += var
        if total >= 0.9:
            return n_pcs + 1

    return n_pcs


def get_rc(dist, k=100):
    """
    Returns the Relative Constrast (He et al., 2012)

    Parameters
    ----------
    dist : array-like, shape(n_queries, n_neighbors)
        Where n_queries refers to the number of points
        (queries); and n_neighbors refers to the number
        of closest neighbors of each point.

    k : int, default=100
        Determines how many nearest neighbors
        must be evaluated to estimate the RC.
        It can be less or equal to the number of
        neighbors in the `dist` matrix.

    Returns
    -------
    rc : array-like, shape(N,)
        The RC of each data point in the dist matrix.
    """
    dist = dist[:, 1 : k + 1]
    rc = np.mean(dist, axis=1) / dist[:, k - 1]
    return rc


def get_rv(X):
    """
    Returns the Relative Variance (François et al., 2007)

    Parameters
    ----------
    X : array-like, shape(n_samples, n_features)
        The dataset.

    Returns
    -------
    rv : float
        The RV of the dataset `X`.
    """
    from scipy.linalg import norm

    norm_values = norm(X, axis=1)
    rv = np.std(norm_values) / np.mean(norm_values)
    return rv