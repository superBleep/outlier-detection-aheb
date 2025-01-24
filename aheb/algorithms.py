import numpy as np
from numpy.typing import NDArray
from base import MMS, EMMS
from deletion import remove, remove_nan


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
            Y_pred[i] = 2  # Mark outlier

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
                Y_pred[i] = 2  # Mark outlier

            X_new = remove(X_new, out_idx)  # Remove outlier
            n = len(X_new)

            if n == 1:
                return X_new, Y_pred 
            else:
                EMMS_max, MMS_min, out = MMS(X_new)

    return X_new, Y_pred