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
    n_missing = np.int32(n * 0.1)

    np.random.seed(40)
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
        np.delete(cont_idx, np.where(cont_idx == 0)[0][0])

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

    if features['init_miss']:
        nan_idx = np.random.choice(np.arange(n)[1:], n_missing, replace=False)

        X[nan_idx], Y[nan_idx] = np.nan, -1

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


def MMS(X: NDArray) -> tuple[float, float, float]:
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
        out: float
            Identified outlier.
    """
    a_max, a_min = np.max(X), np.min(X)
    S_n = np.sum(X)
    n = len(X)

    MMS_max = (a_max - a_min) / (S_n - a_min * n)
    MMS_min = (a_max - a_min) / (a_max * n - S_n)
    
    if MMS_max > MMS_min:
        out = np.max(X)
    elif MMS_max < MMS_min:
        out = np.min(X)
    else:
        out = None

    return MMS_max, MMS_min, out


def EMMS(X: NDArray) -> tuple[float, float, float]:
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
        out: float
            Identified outlier.
    """
    a_0 = X[0]

    def a_T(a: float) -> float:
        return a - a_0

    n = len(X)
    G_aT = np.sum(a_T(X)) / n
    Gx = np.sum(np.arange(n)) / n

    def a_TT(a: float) -> float:
        return np.abs(a_T(a) - np.where(X == a)[0][0] * (G_aT / Gx))

    a_TT_X = np.array(list(map(a_TT, X)))
    a_TT_max = np.max(a_TT_X)

    S_n_TT = np.sum(np.array(list(map(a_TT, X))))

    if a_TT_max == 0:
        return None, None, None
    else:
        EMMS_max = a_TT_max / S_n_TT
        EMMS_min = a_TT_max / (a_TT_max * n - S_n_TT)
        out = X[np.where(a_TT_X == a_TT_max)[0][0]]

        return EMMS_max, EMMS_min, out


def detect_unknown(X: NDArray, k1: float, k2: float) -> tuple[NDArray, NDArray]:
    """
        First method of detecting outliers, as defined in the paper.
        The method is applied without any knowledge of the nature of the elements (outliers / nonoutliers).

        Parameters
        ----------
        X : ndarray
            Original dataset.
        k1: float
            Constant for the outlier detection criteria, using MMS.
        k2: float
            Constant for the outlier detection criteria, using EMSS.
        
        Returns
        -------
        X_clean : ndarray
            Dataset with outliers removed.
        Y_pred: float
            Predicted values for the anomalies in the original dataset.

        See also
        -------
        MMS, EMMS
    """
    X_clean = remove_nan(X)
    X_new = np.copy(X_clean)

    n = len(X)
    Y_pred = np.repeat([0], n)

    R_w = (2 / n) * (1 + k1)
    MMS_max, MMS_min, out = MMS(X_new)

    while out != None and (MMS_max > R_w or MMS_min > R_w):
        r = np.where(X_new == out)[0][0]
        
        if np.where(X_clean == out)[0].size > 0:
            i = np.where(X_clean == out)[0][0]
            Y_pred[i] = 1  # Mark outlier

        X_new = remove(X_new, r)  # Remove outlier
        n = len(X_new)

        if n == 1:
            return X_new, Y_pred 
        else:
            MMS_max, MMS_min, out = MMS(X_new)

    R_w = (2 / n) * (1 + k2)
    EMMS_max, EMMS_min, out = EMMS(X_new)

    while out != None and (EMMS_max > R_w or EMMS_min > R_w):
        r = np.where(X_new == out)[0][0]

        if np.where(X_clean == out)[0].size > 0:
            i = np.where(X_clean == out)[0][0]
            Y_pred[i] = 1  # Mark outlier

        X_new = remove(X_new, r)
        n = len(X_new)

        if n == 1:
            return X_new, Y_pred 
        else:
            EMMS_max, EMMS_min, out = EMMS(X_new)

    return X_new, Y_pred


def detect_known(X: NDArray, Y: NDArray, k1: float, k2: float) -> tuple[NDArray, NDArray]:
    non_idx = np.where(Y == 0)[0]  # Indexes of non-outliers in X
    X_win = X[non_idx[0]:(non_idx[-1]+1)]  # Window where the ends are non-outliers

    X_new = remove_nan(X_win)
    X_clean = np.copy(X_new)

    n = len(X_win)
    Y_pred = np.repeat([0], n)

    R_w = (2 / n) * (1 + k1)
    MMS_max, MMS_min, out = MMS(X_new)

    while out != None and (MMS_max > R_w or MMS_min > R_w):
        out_idx = np.where(X_new == out)[0][0]

        # Check if outlier is first or last term
        if out_idx == 0 or out_idx == (n - 1):
            break
        else:
            if np.where(X_clean == out)[0].size > 0:
                i = np.where(X_clean == out)[0][0]
                Y_pred[i] = 1  # Mark outlier

            X_new = remove(X_new, out_idx)  # Remove outlier
            n = len(X_new)

            if n == 1:
                return X_new, Y_pred 
            else:
                MMS_max, MMS_min, out = MMS(X_new)

    R_w = (2 / n) * (1 + k2)
    EMMS_max, EMMS_min, out = EMMS(X_new)

    while out != None and (EMMS_max > R_w or EMMS_min > R_w):
        out_idx = np.where(X_new == out)[0][0]

        if out_idx == 0 or out_idx == (n - 1):
            break
        else:
            if np.where(X_clean == out)[0].size > 0:
                i = np.where(X_clean == out)[0][0]
                Y_pred[i] = 1  # Mark outlier

            X_new = remove(X_new, out_idx)  # Remove outlier
            n = len(X_new)

            if n == 1:
                return X_new, Y_pred 
            else:
                EMMS_max, MMS_min, out = MMS(X_new)

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
    n = len(env_combs)

    t_out, t_nonout, pred_out, pred_nonout = 0, 0, 0, 0
    for idx, comb in enumerate(env_combs[:1000]):
        print(f'\rComputing dataset {idx} out of {n}', end='')
        X, Y = gen_data(comb)
        _, Y_pred = detect_unknown(X, 0.5, 0.01)

        t_out += len(np.where(Y == 1)[0])
        t_nonout = len(Y) - t_out

        pred_out += len(np.where(Y_pred == 1)[0])
        pred_nonout = len(Y_pred) - pred_out
    print()

    print(pred_out / t_out)
    print(pred_nonout / t_nonout)
