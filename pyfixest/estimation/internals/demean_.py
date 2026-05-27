from collections.abc import Callable

import numba as nb
import numpy as np

from pyfixest.estimation.internals.literals import DemeanerBackendOptions


def demean_model(
    Y: np.ndarray,
    X: np.ndarray,
    # TODO: make sure we really need to pass names here -- is there another way to get the cache hit?
    y_names: list[str],
    x_names: list[str],
    fe: np.ndarray | None,
    weights: np.ndarray | None,
    lookup_demeaned_data: dict[frozenset[int], dict[str, np.ndarray]],
    na_index: frozenset[int],
    fixef_tol: float,
    fixef_maxiter: int,
    demean_func: Callable,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Demean a regression model.

    Demeans a single regression model via the alternating projections algorithm
    (see `demean` function). Prior to demeaning, the function checks if some of
    the variables have already been demeaned and uses values from the cache
    `lookup_demeaned_data` if possible. If the model has no fixed effects, the
    function does not demean the data.

    Parameters
    ----------
    Y : np.ndarray
        2D array of the dependent variable, shape (N, 1).
    X : np.ndarray
        2D array of the covariates, shape (N, k).
    y_names : list[str]
        Column names for Y, used as cache keys.
    x_names : list[str]
        Column names for X, used as cache keys.
    fe : np.ndarray or None
        2D integer array of fixed effects, shape (N, n_fe). None if no fixed effects.
    weights : numpy.ndarray or None
        A numpy array of weights. None if no weights.
    lookup_demeaned_data : dict[frozenset[int], dict[str, np.ndarray]]
        Cache mapping na_index to {col_name: demeaned_column}. Checked to avoid
        redundant demeaning across multiple models sharing the same fixed effects.
    na_index : frozenset[int]
        Indices of dropped rows, used as a hashable cache key.
    fixef_tol: float
        The tolerance for the demeaning algorithm.
    fixef_maxiter: int
        The maximum number of iterations for the demeaning algorithm.
    demean_func : Callable
        The demeaning backend function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Demeaned (Yd, Xd) arrays.
    """
    if fe is None:
        # no demeaning to do, return early
        return Y, X

    yx_names = y_names + x_names
    YX = np.column_stack([Y, X]).astype(np.float64, copy=False)

    if weights is not None and weights.ndim > 1:
        weights = weights.flatten()

    # TODO: make this a default dict so that we can get rid of the first branch
    cache = lookup_demeaned_data.get(na_index)

    if cache is None:
        # no cache, so demean everything

        YX_demeaned, success = demean_func(
            x=YX,
            flist=fe.astype(np.uintp),
            weights=weights,
            tol=fixef_tol,
            maxiter=fixef_maxiter,
        )
        if not success:
            raise ValueError(f"Demeaning failed after {fixef_maxiter} iterations.")

        cache = {name: YX_demeaned[:, i] for i, name in enumerate(yx_names)}
        lookup_demeaned_data[na_index] = cache

    elif missing := [n for n in yx_names if n not in cache]:
        # there is a cache, but it's missing some entries

        missing_idx = [yx_names.index(n) for n in missing]
        var_diff = YX[:, missing_idx]
        if var_diff.ndim == 1:
            var_diff = var_diff.reshape(-1, 1)

        demeaned_new, success = demean_func(
            x=var_diff,
            flist=fe.astype(np.uintp),
            weights=weights,
            tol=fixef_tol,
            maxiter=fixef_maxiter,
        )
        if not success:
            raise ValueError(f"Demeaning failed after {fixef_maxiter} iterations.")

        for i, name in enumerate(missing):
            cache[name] = demeaned_new[:, i]

    # cache is now populated -- pull from it
    YX_demeaned = np.column_stack([cache[n] for n in yx_names])

    return YX_demeaned[:, :1], YX_demeaned[:, 1:]


@nb.njit
def _sad_converged(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    for i in range(a.size):
        if np.abs(a[i] - b[i]) >= tol:
            return False
    return True


@nb.njit(locals=dict(id=nb.uint32))
def _subtract_weighted_group_mean(
    x: np.ndarray,
    sample_weights: np.ndarray,
    group_ids: np.ndarray,
    group_weights: np.ndarray,
    _group_weighted_sums: np.ndarray,
) -> None:
    _group_weighted_sums[:] = 0

    for i in range(x.size):
        id = group_ids[i]
        _group_weighted_sums[id] += sample_weights[i] * x[i]

    for i in range(x.size):
        id = group_ids[i]
        x[i] -= _group_weighted_sums[id] / group_weights[id]


@nb.njit
def _calc_group_weights(
    sample_weights: np.ndarray, group_ids: np.ndarray, n_groups: np.ndarray
):
    n_samples, n_factors = group_ids.shape
    dtype = sample_weights.dtype
    group_weights = np.zeros((n_factors, n_groups), dtype=dtype).T

    for j in range(n_factors):
        for i in range(n_samples):
            id = group_ids[i, j]
            group_weights[id, j] += sample_weights[i]

    return group_weights


@nb.njit(parallel=True)
def demean(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[np.ndarray, bool]:
    """
    Demean an array.

    Workhorse for demeaning an input array `x` based on the specified fixed
    effects and weights via the alternating projections algorithm.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape (n_samples, n_features). Needs to be of type float.
    flist : numpy.ndarray
        Array of shape (n_samples, n_factors) specifying the fixed effects.
        Needs to already be converted to integers.
    weights : numpy.ndarray
        Array of shape (n_samples,) specifying the weights.
    tol : float, optional
        Tolerance criterion for convergence. Defaults to 1e-08.
    maxiter : int, optional
        Maximum number of iterations. Defaults to 100_000.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        A tuple containing the demeaned array of shape (n_samples, n_features)
        and a boolean indicating whether the algorithm converged successfully.

    Examples
    --------
    ```{python}
    import numpy as np
    import pyfixest as pf
    from pyfixest.utils.dgps import get_blw
    from pyfixest.estimation.internals.demean_ import demean
    from formulaic import model_matrix

    fml = "y ~ treat | state + year"

    data = get_blw()
    data.head()

    Y, rhs = model_matrix(fml, data)
    X = rhs[0].drop(columns="Intercept")
    fe = rhs[1].drop(columns="Intercept")
    YX = np.concatenate([Y, X], axis=1)

    # to numpy
    Y = Y.to_numpy()
    X = X.to_numpy()
    YX = np.concatenate([Y, X], axis=1)
    fe = fe.to_numpy().astype(int)  # demean requires fixed effects as ints!

    YX_demeaned, success = demean(YX, fe, weights = np.ones(YX.shape[0]))
    Y_demeaned = YX_demeaned[:, 0]
    X_demeaned = YX_demeaned[:, 1:]

    print(np.linalg.lstsq(X_demeaned, Y_demeaned, rcond=None)[0])
    print(pf.feols(fml, data).coef())
    ```
    """
    n_samples, n_features = x.shape
    n_factors = flist.shape[1]

    if x.flags.f_contiguous:
        res = np.empty((n_features, n_samples), dtype=x.dtype).T
    else:
        res = np.empty((n_samples, n_features), dtype=x.dtype)

    n_threads = nb.get_num_threads()

    n_groups = flist.max() + 1
    group_weights = _calc_group_weights(weights, flist, n_groups)
    _group_weighted_sums = np.empty((n_threads, n_groups), dtype=x.dtype)

    x_curr = np.empty((n_threads, n_samples), dtype=x.dtype)
    x_prev = np.empty((n_threads, n_samples), dtype=x.dtype)

    not_converged = 0
    for k in nb.prange(n_features):
        tid = nb.get_thread_id()

        xk_curr = x_curr[tid, :]
        xk_prev = x_prev[tid, :]
        for i in range(n_samples):
            xk_curr[i] = x[i, k]
            xk_prev[i] = x[i, k] - 1.0

        for _ in range(maxiter):
            for j in range(n_factors):
                _subtract_weighted_group_mean(
                    xk_curr,
                    weights,
                    flist[:, j],
                    group_weights[:, j],
                    _group_weighted_sums[tid, :],
                )
            if _sad_converged(xk_curr, xk_prev, tol):
                break

            xk_prev[:] = xk_curr[:]
        else:
            not_converged += 1

        res[:, k] = xk_curr[:]

    success = not not_converged
    return (res, success)


def _set_demeaner_backend(
    demeaner_backend: DemeanerBackendOptions,
) -> Callable:
    """Set the demeaning backend.

    Currently, we allow for a numba backend, rust backend, jax backend, and cupy backend.
    JAX and CuPy are expected to be faster on GPU for larger problems, but not necessarily
    faster than the numba and rust algos.

    Parameters
    ----------
    demeaner_backend : Literal["numba", "jax", "rust", "cupy", "cupy32", "cupy64"]
        The demeaning backend to use.

    Returns
    -------
    Callable
        The demeaning function.

    Raises
    ------
    ValueError
        If the demeaning backend is not supported.
    """
    if demeaner_backend == "rust":
        from pyfixest.core.demean import demean as demean_rs

        return demean_rs
    elif demeaner_backend == "rust-cg":
        from pyfixest.core.demean import demean_within

        return demean_within
    elif demeaner_backend == "numba":
        return demean
    elif demeaner_backend == "jax":
        from pyfixest.estimation.jax.demean_jax_ import demean_jax

        return demean_jax
    elif demeaner_backend in ["cupy", "cupy64"]:
        from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy64

        return demean_cupy64
    elif demeaner_backend == "cupy32":
        from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy32

        return demean_cupy32
    else:
        raise ValueError(f"Invalid demeaner backend: {demeaner_backend}")
