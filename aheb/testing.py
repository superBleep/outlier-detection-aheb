import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import itertools


def gen_data(features: dict) -> tuple[NDArray, NDArray]:
    """
        Generate artificial contaminated datasets, based on a set of features.

        Parameters
        ----------
        features : dict
            Dictionary containing the necesarry features to build the dataset.

        Returns
        -------
        X : ndarray
            The constructed dataset.
        Y : ndarray
            Ground truth for the dataset.
    """
    n = features['n']
    n_cont = np.int32(n * 0.5)
    n_missing = np.int32(n * 0.1)

    ref = np.random.randint(-100, 100)

    # Generate an arithmetic progression with n elements
    match features['type']:
        case 'asc':
            X = np.arange(ref, ref + n, dtype=np.float32)
        case 'const':
            X = np.repeat([100.0], n)
        case 'desc':
            X = np.arange(ref + n - 1, ref - 1, -1, dtype=np.float32)
        
    cont_idx = np.random.choice(n, n_cont, replace=False)

    # Make sure the first element is an outlier
    if features['ref_outlier'] and 0 not in cont_idx:
        cont_idx[0] = 0

    # Make sure the first element is NOT an outlier
    if not features['ref_outlier'] and 0 in cont_idx:
        cont_idx = cont_idx[cont_idx != 0]
        n_cont -= 1

    # Generate outliers
    if features['gauss_out']:
        outliers = np.random.normal(np.nanmean(X), scale=5, size=n_cont)
    else:
        outliers = X[cont_idx] * np.random.choice([1e-2, 1e2], n_cont).astype('float')

    # Replace selected data with the generated outliers
    Y = np.repeat([0], n)
    for i, idx in enumerate(cont_idx):
        X[idx] = outliers[i]
        Y[idx] = 1

    if features['init_miss']:
        nan_idx = np.random.choice(np.arange(n)[1:], n_missing, replace=False)

        X[nan_idx], Y[nan_idx] = np.nan, -1

    return X, Y


def gen_features() -> list[dict]:
    """
        Generate different combinations of environments to test the algorithms.

        Returns
        -------
        env_combs : list[dict]
            List of dictionaries containing environment features.
    """
    env_features = {
        'type': ['asc', 'const', 'desc'],
        'n': np.arange(10, 1001),
        'gauss_out': [True, False],
        'ref_outlier': [True, False],
        'init_miss': [True, False]
    }

    keys, values = list(env_features.keys()), list(env_features.values())
    env_combs = [dict(zip(keys, comb)) for comb in itertools.product(*values)]

    return env_combs


def gen_graph(X: NDArray, Y_pred: NDArray, k1: float, k2: float) -> None:
    """
        Plot a graph for an outlier detection over a dataset.

        Parameters
        ----------
        X : ndarray
            Original dataset.
        Y_pred: ndarray
            Predicted outlier labels.
        k1: float
            Constant for the outlier detection criteria, using MMS.
        k2: float
            Constant for the outlier detection criteria, using EMMS.
    """
    plt.figure(1)

    plt.scatter(np.where(Y_pred == 0)[0], X[Y_pred == 0], c='green', marker='s', edgecolor='black', label='Nonoutliers')
    plt.scatter(np.where(Y_pred == 1)[0], X[Y_pred == 1], c='red', marker='^', edgecolor='black', label='Detected by MMS')
    plt.scatter(np.where(Y_pred == 2)[0], X[Y_pred == 2], c='yellow', marker='.', edgecolor='black', s=100, label='Detected by EMMS')

    for i in range(len(X)):
        alpha = 0.5 if i in np.where(Y_pred == 0)[0] else 1

        plt.text(i, X[i], '{:.2f}'.format(X[i]), alpha=alpha, ha='center', va='bottom')

    plt.title(f'Outlier predictions ($k_1={k1}$, $k_2={k2}$)')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()


    plt.show()

