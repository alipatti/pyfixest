import numpy as np
import pandas as pd
import pyhdfe
import pytest

from pyfixest.core import demean as demean_rs
from pyfixest.core.demean import demean_within
from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy32, demean_cupy64
from pyfixest.estimation.internals.demean_ import (
    _set_demeaner_backend,
    demean,
    demean_model,
)
from pyfixest.estimation.jax.demean_jax_ import demean_jax


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_cupy32", "demean_cupy64"],
)
def test_demean(benchmark, demean_func):
    rng = np.random.default_rng(929291)

    N = 1_000
    M = 10
    x = rng.normal(0, 1, M * N).reshape((N, M))
    f1 = rng.choice(list(range(M)), N).reshape((N, 1))
    f2 = rng.choice(list(range(M)), N).reshape((N, 1))

    flist = np.concatenate((f1, f2), axis=1).astype(np.uint)

    # without weights
    weights = np.ones(N)
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_pyfixest, _ = demean_func(x, flist, weights, tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)

    # with weights
    weights = rng.uniform(0, 1, N).reshape((N, 1))
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x, weights)
    res_pyfixest, _ = benchmark(demean_func, x, flist, weights.flatten(), tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)


def test_set_demeaner_backend():
    # Test numba backend
    demean_func = _set_demeaner_backend("numba")
    assert demean_func == demean

    # Test jax backend
    demean_func = _set_demeaner_backend("jax")
    assert demean_func == demean_jax

    demean_func = _set_demeaner_backend("rust")
    assert demean_func == demean_rs

    demean_func = _set_demeaner_backend("cupy32")
    assert demean_func == demean_cupy32

    demean_func = _set_demeaner_backend("cupy64")
    assert demean_func == demean_cupy64

    demean_func = _set_demeaner_backend("rust-cg")
    assert demean_func == demean_within

    # Test invalid backend raises ValueError
    with pytest.raises(ValueError, match="Invalid demeaner backend: invalid"):
        _set_demeaner_backend("invalid")


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_cupy32", "demean_cupy64"],
)
def test_demean_model_no_fixed_effects(benchmark, demean_func):
    """Test demean_model when there are no fixed effects."""
    N = 1000
    Y = np.random.randn(N, 1)
    X = np.random.randn(N, 2)
    weights = np.ones(N)
    lookup_dict: dict = {}

    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        y_names=["y"],
        x_names=["x1", "x2"],
        fe=None,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    assert np.allclose(Y, Yd)
    assert np.allclose(X, Xd)


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_cupy32", "demean_cupy64"],
)
def test_demean_model_with_fixed_effects(benchmark, demean_func):
    """Test demean_model with fixed effects."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = rng.normal(0, 1, (N, 1))
    X = rng.normal(0, 1, (N, 2))
    fe = np.column_stack([rng.integers(0, 10, N), rng.integers(0, 5, N)])
    weights = np.ones(N)
    lookup_dict: dict = {}

    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        y_names=["y"],
        x_names=["x1", "x2"],
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    assert not np.allclose(Y, Yd)
    assert not np.allclose(X, Xd)

    # verify results are cached
    assert frozenset() in lookup_dict
    cache = lookup_dict[frozenset()]
    assert np.allclose(cache["y"].reshape(-1, 1), Yd)
    assert np.allclose(cache["x1"], Xd[:, 0])
    assert np.allclose(cache["x2"], Xd[:, 1])


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_cupy32", "demean_cupy64"],
)
def test_demean_model_with_weights(benchmark, demean_func):
    """Test demean_model with weights."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = rng.normal(0, 1, (N, 1))
    X = rng.normal(0, 1, (N, 2))
    fe = rng.integers(0, 10, (N, 1))
    weights = rng.uniform(0.5, 1.5, N)
    lookup_dict: dict = {}

    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        y_names=["y"],
        x_names=["x1", "x2"],
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    Yd_unweighted, Xd_unweighted = demean_model(
        Y=Y,
        X=X,
        y_names=["y"],
        x_names=["x1", "x2"],
        fe=fe,
        weights=np.ones(N),
        lookup_demeaned_data={},
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    assert not np.allclose(Yd, Yd_unweighted)
    assert not np.allclose(Xd, Xd_unweighted)


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_cupy32", "demean_cupy64"],
)
def test_demean_model_caching(benchmark, demean_func):
    """Test the caching behavior of demean_model."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = rng.normal(0, 1, (N, 1))
    X = rng.normal(0, 1, (N, 2))
    fe = rng.integers(0, 10, (N, 1))
    weights = np.ones(N)
    lookup_dict: dict = {}

    # first run — computes and caches
    Yd1, Xd1 = demean_model(
        Y=Y,
        X=X,
        y_names=["y"],
        x_names=["x1", "x2"],
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # second run — should hit cache
    Yd2, Xd2 = benchmark(
        demean_model,
        Y=Y,
        X=X,
        y_names=["y"],
        x_names=["x1", "x2"],
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    assert np.allclose(Yd1, Yd2)
    assert np.allclose(Xd1, Xd2)

    # add new variable — should use partial cache for x1/x2 and demean only x3
    X_new = np.column_stack([X, rng.normal(0, 1, N)])

    _, Xd3 = demean_model(
        Y=Y,
        X=X_new,
        y_names=["y"],
        x_names=["x1", "x2", "x3"],
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # original columns should match previous results
    assert np.allclose(Xd3[:, :2], Xd2)
    assert Xd3.shape[1] == 3


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_cupy32", "demean_cupy64"],
)
def test_demean_model_maxiter_convergence_failure(demean_func):
    """Test that demean_model fails when maxiter is too small."""
    N = 100
    rng = np.random.default_rng(42)

    Y = rng.normal(0, 1, (N, 1))
    X = rng.normal(0, 1, (N, 1))
    fe = np.column_stack([rng.choice(N // 10, N), rng.choice(N // 10, N)])
    weights = np.ones(N)

    with pytest.raises(ValueError, match="Demeaning failed after 1 iterations"):
        demean_model(
            Y=Y,
            X=X,
            y_names=["y"],
            x_names=["x1"],
            fe=fe,
            weights=weights,
            lookup_demeaned_data={},
            na_index=frozenset(),
            fixef_tol=1e-6,
            fixef_maxiter=1,
            demean_func=demean_func,
        )


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_cupy32", "demean_cupy64"],
)
def test_demean_model_custom_maxiter_success(demean_func):
    """Test that demean_model succeeds with reasonable maxiter."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = rng.normal(0, 1, (N, 1))
    X = rng.normal(0, 1, (N, 1))
    fe = rng.integers(0, 10, (N, 1))
    weights = np.ones(N)

    Yd, Xd = demean_model(
        Y=Y,
        X=X,
        y_names=["y"],
        x_names=["x1"],
        fe=fe,
        weights=weights,
        lookup_demeaned_data={},
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=5000,
        demean_func=demean_func,
    )

    assert isinstance(Yd, np.ndarray)
    assert isinstance(Xd, np.ndarray)
    assert Yd.shape == Y.shape
    assert Xd.shape == X.shape


def test_demean_maxiter_parameter():
    """Test that the demean function respects maxiter parameter."""
    N = 100
    rng = np.random.default_rng(42)

    x = rng.normal(0, 1, N * 2).reshape((N, 2))
    flist = np.arange(N).reshape((N, 1)).astype(np.uint)
    weights = np.ones(N)

    _, success = demean(x, flist, weights, tol=1e-10, maxiter=1)
    assert not success

    _, success = demean(x, flist, weights, tol=1e-10, maxiter=100_000)
    # may or may not converge, but should not crash


def test_feols_integration_maxiter():
    """Integration test: Test fixef_maxiter flows from feols to demean."""
    import pyfixest as pf

    N = 1000
    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        {
            "y": rng.normal(0, 1, N),
            "x": rng.normal(0, 1, N),
            "fe": rng.integers(0, 50, N),
        }
    )

    with pytest.raises(ValueError, match="Demeaning failed after 1 iterations"):
        pf.feols("y ~ x | fe", data=data, fixef_maxiter=1)

    model = pf.feols("y ~ x | fe", data=data)
    assert model is not None


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean_rs, demean_within, demean_cupy32, demean_cupy64],
    ids=["demean_rs", "demean_within", "demean_cupy32", "demean_cupy64"],
)
def test_demean_complex_fixed_effects(benchmark, demean_func):
    """Benchmark demean functions with complex multi-level fixed effects."""
    X, flist, weights = generate_complex_fixed_effects_data()

    X_demeaned, success = benchmark.pedantic(
        demean_func,
        args=(X, flist, weights),
        kwargs={"tol": 1e-10},
        iterations=1,
        rounds=1,
        warmup_rounds=0,
    )

    assert success, "Benchmarked demeaning should succeed"
    assert X_demeaned.shape == X.shape


def generate_complex_fixed_effects_data():
    """
    Complex fixed effects example ported from fixest R-implementation:
    https://github.com/lrberge/fixest/blob/ac1be27fda5fc381c0128b861eaf5bda88af846c/_BENCHMARK/Data%20generation.R#L125 .

    """
    rng = np.random.default_rng(42)
    n = 10**5  # Large dataset for benchmarking
    nb_indiv = n // 20
    nb_firm = max(1, round(n / 160))
    nb_year = max(1, round(n**0.3))
    id_indiv = rng.choice(nb_indiv, n, replace=True)
    id_firm_base = rng.integers(0, 21, n) + np.maximum(1, id_indiv // 8 - 10)
    id_firm = np.minimum(id_firm_base, nb_firm - 1)
    id_year = rng.choice(nb_year, n, replace=True)
    x1 = (
        5 * np.cos(id_indiv)
        + 5 * np.sin(id_firm)
        + 5 * np.sin(id_year)
        + rng.uniform(0, 1, n)
    )
    x2 = np.cos(id_indiv) + np.sin(id_firm) + np.sin(id_year) + rng.normal(0, 1, n)
    y = (
        3 * x1
        + 5 * x2
        + np.cos(id_indiv)
        + np.cos(id_firm) ** 2
        + np.sin(id_year)
        + rng.normal(0, 1, n)
    )
    X = np.column_stack([x1, x2, y])
    flist = np.column_stack([id_indiv, id_firm, id_year]).astype(np.uint64)
    weights = rng.uniform(0.5, 2.0, n)
    return X, flist, weights
