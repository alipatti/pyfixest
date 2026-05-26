import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Any, Final, Generic, cast

import formulaic
import narwhals as nw
import numpy as np
import pandas as pd
from formulaic.parser import DefaultFormulaParser
from formulaic.utils.structured import Structured
from narwhals.typing import IntoDataFrame, IntoDataFrameT

from pyfixest.core.detect_singletons import detect_singletons
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG, FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.parse import Formula
from pyfixest.utils.utils import capture_context


@dataclass(frozen=True, kw_only=True)
class _ModelMatrixKey:
    main: str = "second_stage"
    fixed_effects: str = "fe"
    instrumental_variable: str = "first_stage"
    weights: str = "weights"


MM = Structured[formulaic.ModelMatrix[IntoDataFrameT]]


class ModelMatrix(Generic[IntoDataFrameT]):
    """
    A wrapper around formulaic.ModelMatrix for the specification of PyFixest models.

    This class organizes and processes model matrices for econometric estimation,
    extracting dependent and independent variables, fixed effects, instrumental
    variables, and weights. It handles missing data, singleton observations,
    and ensures proper formatting for estimation procedures.

    Attributes
    ----------
    dependent : pd.DataFrame
        The dependent variable(s) (left-hand side of the main equation).
    independent : pd.DataFrame
        The independent variable(s) (right-hand side of the main equation).
    fixed_effects : pd.DataFrame or None
        Fixed effects variables, encoded as integers.
    endogenous : pd.DataFrame or None
        Endogenous variables in instrumental variable specifications.
    instruments : pd.DataFrame or None
        Instrumental variables for IV estimation.
    weights : pd.DataFrame or None
        Observation weights for weighted estimation.
    model_spec : formulaic.ModelSpec
        The underlying formulaic model specification.
    na_index : frozenset[int]
        Indices of rows that were dropped.
    """

    _data: IntoDataFrameT

    def __init__(
        self,
        model_matrix: MM[IntoDataFrameT],
        drop_rows: set[int],
        drop_singletons: bool = True,
        drop_intercept: bool = False,
    ) -> None:
        self._drop_intercept = drop_intercept

        assert model_matrix.model_spec, "model matrix must have `.model_spec`"
        self._model_spec = model_matrix.model_spec

        self._collect_columns(model_matrix)
        self._collect_data(model_matrix)
        self._process(dropped_rows=drop_rows, drop_singletons=drop_singletons)

    @staticmethod
    def _get_columns(mm: MM[IntoDataFrameT], *keys: str) -> list[str] | None:
        """Extract column names by traversing nested keys, or None if missing."""
        try:
            result = mm
            for k in keys:
                result = result[k]  # type: ignore

            assert nw.dependencies.is_into_dataframe(result)
            result = cast(IntoDataFrameT, result)

            return nw.from_native(result).columns
        except KeyError:
            return None

    def _collect_columns(self, model_matrix: MM[IntoDataFrameT]) -> None:
        self._dependent = self._get_columns(model_matrix, _ModelMatrixKey.main, "lhs")
        self._independent = self._get_columns(model_matrix, _ModelMatrixKey.main, "rhs")
        self._fixed_effects = self._get_columns(
            model_matrix, _ModelMatrixKey.fixed_effects
        )
        self._endogenous = self._get_columns(
            model_matrix, _ModelMatrixKey.instrumental_variable, "lhs"
        )
        self._instruments = self._get_columns(
            model_matrix, _ModelMatrixKey.instrumental_variable, "rhs"
        )
        self._weights = self._get_columns(model_matrix, _ModelMatrixKey.weights)

    def _collect_data(self, model_matrix: MM[IntoDataFrameT]) -> None:

        def _combine(df1: nw.DataFrame, df2: nw.DataFrame) -> nw.DataFrame:
            new_columns = set(df2.columns) - set(df1.columns)

            if not new_columns:
                return df1

            # NOTE: old pandas code was this
            # >>> data = pd.concat(datas, ignore_index=False, axis=1)
            # do we need to do something special for ignore_index=False?
            # tests pass, so i assume no...

            return nw.concat(
                [df1, df2.select(new_columns)],
                how="horizontal",
            )

        self._data = reduce(
            _combine,
            (nw.from_native(df) for df in model_matrix._flatten()),  # type: ignore
        ).to_native()

    def _process(self, dropped_rows: set[int], drop_singletons: bool = False) -> None:
        if self.dependent.shape[1] != 1:
            # If the dependent variable is not numeric, formulaic's contrast encoding kicks in
            # creating multiple columns for the dependent variable
            # TODO: Make this check more explicit?
            raise TypeError("The dependent variable must be numeric.")
        if self.endogenous is not None and self.endogenous.shape[1] != 1:
            raise TypeError("The endogenous variable must be numeric.")
        # Drop rows with non-finite values
        is_infinite = pd.Series(
            ~np.isfinite(self._data).all(axis=1), index=self._data.index
        )
        if is_infinite.any():
            infinite_indices = is_infinite[is_infinite].index.tolist()
            dropped_rows |= set(infinite_indices)
            self._data.drop(infinite_indices, inplace=True)
            warnings.warn(
                f"{is_infinite.sum()} rows with infinite values dropped from the model.",
            )
        if self._fixed_effects is not None:
            # Ensure fixed effects are `int32`
            self._data[self._fixed_effects] = self._data[self._fixed_effects].astype(
                "int32"
            )
        if self.fixed_effects is not None or self._drop_intercept:
            if self._independent is not None:
                self._independent = [
                    col for col in self._independent if col != "Intercept"
                ]
            if self._instruments is not None:
                self._instruments = [
                    col for col in self._instruments if col != "Intercept"
                ]
        # Drop singletons if specified
        if drop_singletons and self.fixed_effects is not None:
            is_singleton = pd.Series(
                detect_singletons(self.fixed_effects.to_numpy()),
                index=self._data.index,
            )
            if is_singleton.any():
                singleton_indices = self._data[is_singleton].index.tolist()
                dropped_rows |= set(singleton_indices)
                self._data.drop(singleton_indices, inplace=True)
                warnings.warn(
                    f"{is_singleton.sum()} singleton fixed effect(s) dropped from the model."
                )
        self._na_index = frozenset(dropped_rows)

    @property
    def dependent(self) -> IntoDataFrameT:
        """
        Get the dependent variable(s) from the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the dependent variable(s) (left-hand side
            of the main equation).
        """
        cols = self._dependent or []
        return nw.from_native(self._data).select(cols).to_native()

    @property
    def independent(self) -> IntoDataFrameT:
        """
        Get the independent variable(s) from the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the independent variable(s) (right-hand side
            of the main equation). Intercept columns are excluded when fixed
            effects are present.
        """
        cols = self._independent or []
        return nw.from_native(self._data).select(cols).to_native()

    @property
    def fixed_effects(self) -> IntoDataFrameT | None:
        """
        Get the fixed effects variables from the model.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the fixed effects variables encoded as integers,
            or None if no fixed effects are specified in the model.
        """
        if self._fixed_effects is None:
            return None

        return nw.from_native(self._data).select(self._fixed_effects).to_native()

    @property
    def endogenous(self) -> IntoDataFrameT | None:
        """
        Get the endogenous variable(s) for instrumental variable estimation.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the endogenous variable(s) (left-hand side
            of the first-stage equation in IV estimation), or None if not
            using instrumental variables.
        """
        if self._endogenous is None:
            return None

        return nw.from_native(self._data).select(self._endogenous).to_native()

    @property
    def instruments(self) -> IntoDataFrameT | None:
        """
        Get the instrumental variable(s) for IV estimation.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the instrumental variable(s) (right-hand side
            of the first-stage equation in IV estimation), or None if not
            using instrumental variables. Intercept columns are excluded when
            fixed effects are present.
        """
        if self._instruments is None:
            return None

        return nw.from_native(self._data).select(self._instruments).to_native()

    @property
    def weights(self) -> IntoDataFrameT | None:
        """
        Get the observation weights for weighted estimation.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the observation weights (must be non-negative
            numeric values), or None if no weights are specified.
        """
        if self._weights is None:
            return None

        return nw.from_native(self._data).select(self._weights).to_native()

    @property
    def model_spec(self) -> formulaic.ModelSpec:
        """
        Get the underlying formulaic model specification.

        Returns
        -------
        formulaic.ModelSpec
            The formulaic ModelSpec object containing metadata about the
            model structure and transformations.
        """
        return self._model_spec

    @property
    def na_index(self) -> frozenset[int]:
        """Integer positions of rows dropped in model matrix creation."""
        return self._na_index


def create_model_matrix(
    formula: Formula,
    data: IntoDataFrameT,
    weights: str | None = None,
    drop_singletons: bool = False,
    drop_intercept: bool = False,
    ensure_full_rank: bool = True,
    context: int | Mapping[str, Any] = 0,
) -> ModelMatrix[IntoDataFrameT]:
    """
    Create a ModelMatrix from a formula and data.

    This function constructs model matrices for econometric estimation by parsing
    formulas and extracting the necessary components (dependent/independent variables,
    fixed effects, instruments, weights) from the provided data.

    Parameters
    ----------
    formula : Formula
        A Formula object specifying the model structure, including dependent and
        independent variables, fixed effects, and instrumental variables.
    data : pd.DataFrame
        The input data containing all variables referenced in the formula.
        The index will be reset during processing.
    weights : str or None, default=None
        Column name in data to use as observation weights. Weights must be
        non-negative numeric values. If None, no weighting is applied.
    drop_singletons : bool, default=False
        If True, observations that are singletons in any fixed effect category
        are dropped from the model.
    drop_intercept : bool, default=False
        If True, the intercept column is removed from the independent variables
        and instruments matrices. The intercept is always removed when fixed
        effects are present, regardless of this parameter.
    ensure_full_rank : bool, default=True
        If True, formulaic will ensure the design matrix is full rank by
        dropping collinear columns.
    context : int or Mapping[str, Any], default=0
        Additional context variables for formulaic during model matrix creation.
        Can be an integer (stack frame depth) or a dictionary of variables to
        make available in the formula environment (e.g., custom transformations).

    Returns
    -------
    ModelMatrix
        A ModelMatrix object containing the processed dependent and independent
        variables, fixed effects, instruments, weights, and metadata about
        dropped observations.

    """
    data = nw.from_native(data).with_row_index("__row_index__").to_native()
    n_observations: Final[int] = nw.from_native(data).shape[0]

    formula_formulaic = _get_formulaic_formula(
        formula=formula, data=data, weights=weights
    )

    model_matrix = cast(
        formulaic.ModelMatrix[IntoDataFrameT],
        formula_formulaic.get_model_matrix(
            data=data,
            ensure_full_rank=ensure_full_rank,
            na_action="drop",
            # output="pandas",
            context=FORMULAIC_TRANSFORMS | {**capture_context(context)},
        ),
    )

    # TODO: need to figure out how to do this
    drop_rows: set[int] = set(range(n_observations)).difference(
        model_matrix[_ModelMatrixKey.main]["lhs"].index
    )
    return ModelMatrix(
        model_matrix,
        drop_rows=drop_rows,
        drop_singletons=drop_singletons,
        drop_intercept=drop_intercept,
    )


def _get_formulaic_formula(
    formula: Formula,
    data: IntoDataFrame,
    weights: str | None = None,
) -> formulaic.Formula:
    # Collate kwargs to be passed to formulaic.Formula
    formula_kwargs: dict[str, str] = {_ModelMatrixKey.main: formula.second_stage}
    if formula.is_fixed_effects:
        formula_kwargs.update(
            {_ModelMatrixKey.fixed_effects: f"{formula.fixed_effects_wrapped} - 1"}
        )
    if formula.is_instrumental_variable:
        formula_kwargs.update(
            {_ModelMatrixKey.instrumental_variable: formula.first_stage}
        )

    if weights is not None:
        w = nw.from_native(data).get_column(weights)

        if not w.dtype.is_numeric():
            try:
                w = w.cast(nw.Float64)
            except nw.exceptions.InvalidOperationError:
                raise ValueError(f"The weights column '{weights}' must be numeric.")

        if (w < 0).any():
            raise ValueError(
                f"The weights column '{weights}' must not contain negative values."
            )

        data = nw.from_native(data).with_columns(w.alias(weights)).to_native()

        formula_kwargs.update({_ModelMatrixKey.weights: f"{weights}-1"})

    formula_formulaic = formulaic.Formula(
        formula_kwargs,  # type: ignore
        _parser=DefaultFormulaParser(
            feature_flags=FORMULAIC_FEATURE_FLAG,
            # When FEs are present, include_intercept=True so that spans_intercept=True
            # terms (like i()) receive reduced_rank=True from formulaic, causing them to
            # drop the first level (matching R/fixest). The intercept column is removed
            # afterwards in ModelMatrix._process(). Without this, i() would receive
            # reduced_rank=False and generate all levels; the post-hoc collinearity check
            # would then drop the last level instead of the first, mismatching R.
            include_intercept=formula.is_fixed_effects,
        ),
    )  # type: ignore
    return formula_formulaic
