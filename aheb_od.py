import itertools
import numpy as np
from numpy.typing import NDArray


def gen_data(features: dict) -> tuple[NDArray, NDArray]:
    """
        Generate artificial contaminated datasets, based on set of features.

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
    n_missing = np.int32(n * 0.3)
    ref = np.random.randint(-100, 100)

    # Generate an arithmetic progression with n elements
    match features['type']:
        case 'asc':
            X = np.arange(ref, ref + n, dtype=np.float32)
        case 'const':
            X = np.repeat([100.0], n)
        case 'desc':
            X = np.arange(ref + n - 1, ref - 1, -1, dtype=np.float32)
        
    # First n_missing datapoints are NaN
    if features['init_miss']:
        X[:n_missing] = [np.nan] * n_missing  

        n_cont = np.int32((n - n_missing) * 0.5)  # Only contaminate the present values
        cont_idx = np.random.choice(np.arange(start=n_missing, stop=n), n_cont, replace=False)

        # Make sure the first present element is an outlier
        if features['ref_outlier'] and n_missing not in cont_idx:
            cont_idx[0] = n_missing
    else:
        cont_idx = np.random.choice(n, n_cont, replace=False)

        # Make sure the first element is an outlier
        if features['ref_outlier'] and 0 not in cont_idx:
            cont_idx[0] = 0

    # Generate outliers
    if features['gauss_out']:
        outliers = np.random.normal(np.nanmean(X), scale=5, size=n_cont)
    else:
        outliers = X[cont_idx] * np.random.choice([1e-2, 1e2], n_cont)

    # Replace selected data with the generated outliers
    Y = np.repeat([0], n)
    for i, idx in enumerate(cont_idx):
        X[idx] = outliers[i]
        Y[idx] = 1

    return X, Y


def remove(X: NDArray, r: int) -> NDArray:
    """
        Remove the element at the specified index from a series.
        All the elements after the removed one will be angularly shifted to the left,
        so that the original relation of the series can be mantained.

        Parameters
        ----------
        X : ndarray
            Original series.
        r: int
            Index of the element to be removed.

        Returns
        -------
        X_new : ndarray
            Updated series, with the element removed.
    """
    n = len(X)

    for i in range(n):
        if not np.isnan(X[i]):
            ref = X[i]
            break

    X_new = np.empty(n - 1, dtype=X.dtype)

    X_new[:r] = X[:r]

    for i in range(1, n - r):
        next = ref + ((X[r + i] - ref) / (r + i)) * (r + i - 1)
        X_new[r + i - 1] = next

    return X_new


def remove_nan(X: NDArray) -> NDArray:
    """
        Remove all the missing elements from a series.

        Parameters
        ----------
        X : ndarray
            Original series.

        Returns
        -------
        X_new : ndarray
            Updated series, with all the missing elements removed.

        See also
        -------
        remove
    """
    n = len(X)
    X_new = np.copy(X)

    # Remove function rearranges elements based on the ones
    # to the right of the deleted one. Thus, the loop must
    # go from right to left
    for i in range(n - 1, -1, -1):
        if np.isnan(X[i]):
            X_new = remove(X_new, i)

    return X_new


def MMS(X: NDArray) -> tuple[float, float]:
    """
        Check for significant outliers in a dataset, using the MMS formula.

        Parameters
        ----------
        X : ndarray
            Original dataset.

        Returns
        -------
        MMS_max : float
            Value which identifies maximum outliers.
        MMS_min: float
            Value which identifies minimum outliers.
    """
    a_max, a_min = np.max(X), np.min(X)
    S_n = np.sum(X)
    n = len(X)

    MMS_max = (a_max - a_min) / (S_n - a_min * n)
    MMS_min = (a_max - a_min) / (a_max * n - S_n)

    return MMS_max, MMS_min


def EMMS(X: NDArray) -> tuple[float, float]:
    """
        Check for nonsignificant outliers in a dataset, using the EMMS formula.

        Parameters
        ----------
        X : ndarray
            Original dataset.

        Returns
        -------
        EMMS_max : float
            Value which identifies maximum outliers.
        EMMS_min: float
            Value which identifies minimum outliers.
    """
    def a_T(a: float) -> float:
        return a - X[0]

    def a_TT(a: float, n: int) -> float:
        G_aT = np.sum(a_T(X)) / n
        Gx = np.sum(np.arange(n)) / n

        return np.abs(a_T(a) - np.where(X == a)[0][0] * (G_aT / Gx))

    n = len(X)
    a_max_TT = a_TT(np.max(X), n)
    S_n_TT = np.sum(a_TT(X, n))

    EMMS_max = a_max_TT / S_n_TT
    EMMS_min = a_max_TT / (a_max_TT * n - S_n_TT)

    print(a_T(X))

    return EMMS_max, EMMS_min


def detect_unknown(X: NDArray, k1: float, k2: float) -> tuple[NDArray, NDArray]:
    X_clean = remove_nan(X)
    X_new = np.copy(X_clean)
    n = len(X)
    Y_pred = np.repeat([0], n)

    R_w = (2 / n) * (1 + k1)
    MMS_max, MMS_min = MMS(X_new)

    while MMS_max > R_w or MMS_min > R_w:
        if MMS_max > R_w:
            out = np.max(X_new)  # Max. element is the outlier
        else:
            out = np.min(X_new)  # Max. element is the outlier

        # Outlier indexes (current & original datasets)
        r = np.where(X_new == out)[0][0]
        i = np.where(X_clean == out)[0][0]

        Y_pred[i] = 1  # Mark outlier
        X_new = remove(X_new, r)  # Remove outlier
        n -= 1

        MMS_max, MMS_min = MMS(X_new)

    R_w = (2 / n) * (1 + k2)
    EMMS_max, EMMS_min = EMMS(X_new)

    while EMMS_max > R_w or EMMS_min > R_w:
        if EMMS_max > R_w:
            out = np.max(X_new)
        else:
            out = np.min(X_new)

        r = np.where(X_new == out)[0][0]
        i = np.where(X_clean == out)[0][0]

        Y_pred[i] = 1
        X_new = remove(X_new, r)
        n -= 1

        EMMS_max, EMMS_min = EMMS(X_new)

    return X_new, Y_pred


if __name__ == '__main__':
    env_features = {
        'type': ['asc', 'const', 'desc'],
        'n': np.arange(10, 1001),
        'gauss_out': [True, False],
        'ref_outlier': [True, False],
        'init_miss': [True, False]
    }

    keys, values = list(env_features.keys()), list(env_features.values())
    env_combs = [dict(zip(keys, comb)) for comb in itertools.product(*values)]

    X, Y = np.array([100, 101, 102, 103.6, 104]), np.array([0, 0, 0, 1, 0])
    X_new, Y_pred = detect_unknown(X, 0.5, 0.01)

    print(X_new, Y_pred)
