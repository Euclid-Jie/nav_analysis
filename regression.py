from typing import List
import numpy as np
import logging
import os
from numpy.lib.stride_tricks import sliding_window_view
from utils import clean

# 此代码参考自：YYTZ.ltd
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format=" %(levelname)-8s %(name)s: %(message)s",
)


def _regress_multi_dimensional_base(y: np.ndarray, xs: np.ndarray):
    """Perform linear regression on input Y and factors row by row.
    Args:
        y: np.ndarray, 2D array of Y variables (T*N).
        xs: np.ndarray, 3D array of X variables (T*N*K).

    Returns:
        tuple of np.ndarray:
            betas: a T*K ndarray of regression coefficients
            errors: a T*N ndarray of regression errors
    """
    # calculate condition number of X at each T, cond is 1D array with size T
    # np.einsum("ijk,ijl->ikl", xs, xs) is T*K*K where the K*K matrix = Xt'Xt
    cond = np.linalg.cond(np.einsum("ijk,ijl->ikl", xs, xs), 2)
    flag_large_cond = (cond > 50).astype(int).sum()
    if flag_large_cond > 0:
        logger.warning(
            f"Conditioon number for OLS X variables is large : {flag_large_cond} times over {y.shape[0]} periods"
        )
    # calculate beta at each T, size of beta is T*K
    # np.linalg.pinv(xs) = X'(X'X)^(-1) with size of T*K*N
    betas = np.einsum("ijk,ik->ij", np.linalg.pinv(xs), y)
    # calculate error at each T, size of error is T*N
    errors = y - np.einsum("ijk,ik->ij", xs, betas)
    return betas, errors


def regress_multi_dimensional(
    Y: np.ndarray, Xs: List[np.ndarray], intercept: bool = False
):
    """Perform linear regression on input Y and factors row by row.
    Args:
        Y: np.ndarray, 2D array of Y variables (T*N).
        Xs: list of np.ndarray , list of 2D array of X variables which all have same shape as Y ([T*N] *K)
        intercept: boolean, add 1s to Xs as intercept when true

    Returns:
        tuple of np.ndarray:
            betas: a T*K ndarray of regression coefficients
            errors: a T*N ndarray of regression errors
    """
    assert Y.ndim == 2, "Y can only be 2D array"
    assert len(Xs) > 0, "At least on X"
    assert Y.shape == Xs[0].shape, "shape of Y and Xs must align"
    if intercept:
        # add 1s to Xs, so the shape is [T*N] *(K+1)
        xs = np.stack(Xs + [np.ones_like(Y, dtype="float64")], axis=-1)
    # convert Xs to 3D array with size T*N*K
    else:
        xs = np.stack(Xs, axis=-1)
    # replace nan with 0s
    y = clean(Y)
    xs = clean(xs)

    betas, errors = _regress_multi_dimensional_base(y, xs)

    RSSf = np.sum(errors**2, axis=1)

    # Calculating Standard Error of coefficients
    _, N, K = xs.shape

    denom = np.sum(
        (xs - np.mean(xs, axis=(1, 2))[:, None][:, :, None]) ** 2, axis=(1, 2)
    )
    SE = np.sqrt(RSSf / (N - K) / denom)
    T_values = betas / SE[:, None]

    # Calculating the F statistic
    TSS = np.sum((y - np.mean(y, axis=1)[:, None]) ** 2, axis=1)  # Total sum of squares
    F = (((TSS - RSSf) / (K - 1)) / (RSSf / (N - K)))[:, None]

    # Calculate the R2
    R2 = 1 - RSSf / TSS

    return betas, errors, T_values, F, R2


def regress_multi_dimensional_by_columns(
    Y: np.ndarray, Xs: List[np.ndarray], intercept: bool = False
):
    """
    Perform linear regression on input Y and factors column by column.
    Args:
        Y: np.ndarray, 2D array of Y variables (T*N).
        Xs: list of np.ndarray , list of 2D array of X variables which all have same shape as Y ([T*N] *K)
        intercept: boolean, add 1s to Xs as intercept when true

    Returns:
        tuple of np.ndarray:
            betas: a K*N or (K+1)*N ndarray of regression coefficients
            errors: a T*N ndarray of regression errors
            T Values: a K*N or (K+1)*N ndarray of T Values
            F Value: a 1*N ndarray of F Values
    """
    assert Y.ndim == 2, "Y can only be 2D array"
    assert len(Xs) > 0, "At least on X"
    assert Y.shape == Xs[0].shape, "shape of Y and Xs must align"

    Y = Y.T
    Xs = [X.T for X in Xs]

    if intercept:
        # add 1s to Xs, so the shape is [T*N] *(K+1)
        xs = np.stack(Xs + [np.ones_like(Y, dtype="float64")], axis=-1)
    # convert Xs to 3D array with size T*N*K
    else:
        xs = np.stack(Xs, axis=-1)
    # replace nan with 0s
    y = clean(Y)
    xs = clean(xs)

    betas, errors = _regress_multi_dimensional_base(y, xs)

    RSSf = np.sum(errors**2, axis=1)

    # Calculating Standard Error of coefficients
    _, N, K = xs.shape

    denom = np.sum(
        (xs - np.mean(xs, axis=(1, 2))[:, None][:, :, None]) ** 2, axis=(1, 2)
    )
    SE = np.sqrt(RSSf / (N - K) / denom)
    T_values = betas / SE[:, None]

    # Calculating the F statistic
    TSS = np.sum((y - np.mean(y, axis=1)[:, None]) ** 2, axis=1)  # Total sum of squares
    F = (((TSS - RSSf) / (K - 1)) / (RSSf / (N - K)))[:, None]

    # Calculate the R2
    R2 = 1 - RSSf / TSS

    return betas.T, errors.T, T_values.T, F.T, R2


def rolling_regression(
    Y: np.ndarray, Xs: List[np.ndarray], window: int, intercept: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear regression on input Y and factors row by row.
    Args:
        Y: np.ndarray, 1D array of Y variables (T).
        Xs: list of np.ndarray , list of 1D array of X variables which all have same shape as Y (T *K)
        window: int, rolling window size
        intercept: boolean, add 1s to Xs as intercept when true
    """
    assert Y.ndim == 1, "Y can only be 1D array"
    assert len(Xs) > 0, "At least on X"
    assert Y.shape == Xs[0].shape, "shape of Y and Xs must align"

    Y_roll = sliding_window_view(Y, window)
    Xs_roll = [sliding_window_view(X, window) for X in Xs]

    return regress_multi_dimensional(Y_roll, Xs_roll, intercept)
