import numpy as np
from numpy.typing import NDArray
from base import MMS, EMMS
from deletion import to_const


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
            Constant for the outlier detection criteria, using EMMS.
        
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
    X_orig = np.copy(X)
    X_new = to_const(X)  # Convert the dataset to a presumed constant series

    n = len(X)
    Y_pred = np.repeat([0], n)

    R_w = (2 / n) * (1 + k1)
    MMS_max, MMS_min, out = MMS(X_new)

    while out != None and (MMS_max > R_w or MMS_min > R_w):
        # Continue only if the outlier is significant
        # (minimum / maximum of the original series)
        orig = X[np.where(np.isclose(X_new, out))][0]

        if orig != np.nanmax(X_orig) and orig != np.nanmin(X_orig):
            break

        r = np.where(np.isclose(X_new, out))[0][0]
        X_new[r] = np.nan
        X_orig[r] = np.nan
        X_new = to_const(X_new)  # Reconvert the working series

        Y_pred[r] = 1  # Mark the element as outlier

        MMS_max, MMS_min, out = MMS(X_new)

    R_w = (2 / n) * (1 + k2)
    EMMS_max, EMMS_min, out = EMMS(X_new)

    while out != None and (EMMS_max > R_w or EMMS_min > R_w):
        # Continue only if there are outliers present
        # (i.e. the transformation of the original series is not constant)
        orig = to_const(X_orig)
        if np.unique(orig[~np.isnan(orig)]).size == 1:
            break

        r = np.where(np.isclose(X_new, out))[0][0]
        X_new[r] = np.nan
        X_orig[r] = np.nan
        X_new = to_const(X_new)

        Y_pred[r] = 2

        EMMS_max, EMMS_min, out = EMMS(X_new)

    return X_orig, Y_pred
