import numpy as np
from numpy.typing import NDArray


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