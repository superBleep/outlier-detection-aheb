import numpy as np
from numpy.typing import NDArray


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


def to_const(X: NDArray) -> NDArray:
    """
        Transform the provided series to a new one with a constant value (NaN values included). If the resulting series is not constant, that means outliers are present in the original series.

        Parameters
        ----------
        X : ndarray
            Original series.

        Returns
        -------
        X_TT: ndarray
            Computed constant series.
    """
    idx = np.where(np.isnan(X), np.nan, np.arange(len(X)))

    X_T = X - X[0]  # Shift elements of the original series
    m = np.nansum(X_T) / np.nansum(idx)  # Gradient takes into accout the shifting
    X_prime = idx * m
    X_TT = X_prime - X_T

    return X_TT


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
