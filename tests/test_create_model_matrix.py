import pandas as pd
import polars as pl
import pytest

from pyfixest.estimation.formula.model_matrix import ModelMatrix, create_model_matrix
from pyfixest.estimation.formula.parse import Formula


def _make_formula(f: str) -> Formula:
    (formula,) = Formula.parse(f)
    return formula


DF = pl.DataFrame | pd.DataFrame


@pytest.fixture(params=["pandas", "polars"])
def simple_df(request: pytest.FixtureRequest) -> DF:
    data = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "x": [1.0, 2.0, 3.0, 4.0],
        "z": [4.0, 3.0, 2.0, 1.0],
        "fe": [1, 1, 2, 2],
    }

    if request.param == "pandas":
        return pd.DataFrame(data)

    if request.param == "polars":
        return pl.DataFrame(data)

    raise NotImplementedError


def test_returns_model_matrix(simple_df: DF) -> None:
    """create_model_matrix returns a ModelMatrix for a basic OLS formula."""
    result = create_model_matrix(formula=_make_formula("y ~ x"), data=simple_df)
    assert isinstance(result, ModelMatrix)


def test_dependent_and_independent_shape(simple_df: DF) -> None:
    """Dependent has one column and independent has intercept + x."""
    result = create_model_matrix(formula=_make_formula("y ~ x"), data=simple_df)
    assert result.dependent.shape == (4, 1)
    assert result.independent.shape == (4, 2)  # intercept + x


def test_fixed_effects_extracted(simple_df: DF) -> None:
    """Fixed effects are extracted when specified with | in the formula."""
    result = create_model_matrix(formula=_make_formula("y ~ x | fe"), data=simple_df)
    assert result.fixed_effects is not None
    assert result.fixed_effects.shape == (4, 1)


def test_iv_endogenous_and_instruments(simple_df: DF) -> None:
    """Endogenous and instruments are populated for an IV formula."""
    result = create_model_matrix(
        formula=_make_formula("y ~ 1 | fe | x ~ z"),
        data=simple_df,
    )
    assert result.endogenous is not None
    assert result.endogenous.shape == (4, 1)
    assert result.instruments is not None
    assert result.instruments.shape == (4, 1)


def test_drop_rows_on_na() -> None:
    """Rows with NA values are dropped and recorded in na_index."""
    df = pd.DataFrame({"y": [1.0, None, 3.0], "x": [1.0, 2.0, 3.0]})
    result = create_model_matrix(formula=_make_formula("y ~ x"), data=df)
    assert len(result.na_index) == 1
    assert result.dependent.shape[0] == 2
