import numpy as np
from numpy.typing import NDArray
from math import isnan


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
    a_max, a_min = np.nanmax(X), np.nanmin(X)
    S_n = np.nansum(X)
    n = np.count_nonzero(~np.isnan(X))

    MMS_max = (a_max - a_min) / (S_n - a_min * n)
    MMS_min = (a_max - a_min) / (a_max * n - S_n)

    if MMS_max > MMS_min:
        out = np.nanmax(X)
    elif MMS_max < MMS_min:
        out = np.nanmin(X)
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
    a_0 = X[~np.isnan(X)][0]  # First non nan element

    def a_T(a: float) -> float:
        return a - a_0

    idx = np.where(np.isnan(X), np.nan, np.arange(len(X)))
    n = np.count_nonzero(~np.isnan(X))
    G_aT = np.nansum(a_T(X))
    Gx = np.nansum(idx)
    
    def a_TT(a: float) -> float:
        i = np.where(X == a)[0][0] if not isnan(a) else np.where(np.isnan(X))[0][0]
        return np.abs(a_T(a) - i * (G_aT / Gx))

    a_TT_X = np.array(list(map(a_TT, X)))

    a_TT_max = np.nanmax(a_TT_X)
    a_TT_min = np.nanmin(a_TT_X)

    S_n_TT = np.nansum(np.array(list(map(a_TT, X))))

    if a_TT_max == 0 or a_TT_max * n == S_n_TT:
        return None, None, None
    else:
        EMMS_max = a_TT_max / S_n_TT
        EMMS_min = a_TT_max / (a_TT_max * n - S_n_TT)

        if EMMS_max > EMMS_min:
            out = X[np.where(a_TT_X == a_TT_max)[0][0]]
        elif EMMS_max < EMMS_min:
            out = X[np.where(a_TT_X == a_TT_min)[0][0]]
        else:
            out = None

        return EMMS_max, EMMS_min, out