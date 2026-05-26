import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Any, Generic, cast

import formulaic
import narwhals as nw
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
    dependent : IntoDataFrameT
        The dependent variable(s) (left-hand side of the main equation).
    independent : IntoDataFrameT
        The independent variable(s) (right-hand side of the main equation).
    fixed_effects : IntoDataFrameT or None
        Fixed effects variables, encoded as integers.
    endogenous : IntoDataFrameT or None
        Endogenous variables in instrumental variable specifications.
    instruments : IntoDataFrameT or None
        Instrumental variables for IV estimation.
    weights : IntoDataFrameT or None
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
        data: IntoDataFrameT,
        drop_singletons: bool = True,
        drop_intercept: bool = False,
    ) -> None:
        self._drop_intercept = drop_intercept

        assert model_matrix.model_spec, "model matrix must have `.model_spec`"
        self._model_spec = model_matrix.model_spec

        self._collect_columns(model_matrix)
        self._collect_data(model_matrix)
        self._process(data=data, drop_singletons=drop_singletons)

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

    def _process(self, data: IntoDataFrameT, drop_singletons: bool = False) -> None:
        """Validate, clean, and finalize the collected model matrix data.

        Adds a sequential row index, then drops rows with null values in any
        formula-referenced column, rows with infinite values in any float column,
        and (optionally) singleton fixed effect observations. The indices of all
        dropped rows are stored in na_index, which is used as a hashable cache
        key for demeaned data in multi-model estimation.

        The null check uses the original pre-formula data rather than the model
        matrix because formulaic with na_action="ignore" can absorb NaN into
        zeros when encoding categoricals (e.g. C(f1)), so NaN would not
        propagate to the model matrix columns.

        Parameters
        ----------
        data : IntoDataFrameT
            The original input data, used to detect null values in
            formula-referenced columns before any encoding is applied.
        drop_singletons : bool, default False
            If True, drop observations that are singletons in any fixed effect
            group and warn about the count removed.

        Raises
        ------
        TypeError
            If the dependent variable has more than one column (a non-numeric
            dependent triggers formulaic contrast encoding producing multiple
            columns), or if the endogenous variable has more than one column.
        """
        # TODO: revisit this implementation (currently quite cludgy and probably slow)

        if nw.from_native(self.dependent).shape[1] != 1:
            # If the dependent variable is not numeric, formulaic's contrast encoding kicks in
            # creating multiple columns for the dependent variable
            # TODO: Make this check more explicit?
            raise TypeError("The dependent variable must be numeric.")

        if (
            self.endogenous is not None
            and nw.from_native(self.endogenous).shape[1] != 1
        ):
            raise TypeError("The endogenous variable must be numeric.")

        df = nw.from_native(self._data).with_row_index("__row_index__")
        n = df.shape[0]
        backend = nw.get_native_namespace(df)

        # drop rows with null in any formula-referenced column, checked against the
        # original data because C(col) encodes NaN as zeros and hides it from the matrix
        formula_vars = {
            v
            for item in self._model_spec._flatten()  # type: ignore[attr-defined]
            if hasattr(item, "variables")
            for v in item.variables
        }
        raw_df = nw.from_native(data)
        null_cols = [c for c in formula_vars if c in raw_df.columns]
        if null_cols:
            is_null = raw_df.select(
                nw.any_horizontal(
                    *(nw.col(c).is_null() for c in null_cols), ignore_nulls=True
                ).alias("__is_null__")
            ).get_column("__is_null__")
            n_null = int(is_null.sum())
            if n_null > 0:
                df = df.filter(~is_null)
                warnings.warn(f"{n_null} rows with NA values dropped from the model.")

        # drop rows with infinite values in any float column (NAs already removed)
        float_cols = [
            c for c, dt in df.schema.items() if dt in (nw.Float32, nw.Float64)
        ]
        if float_cols:
            is_inf = df.select(
                nw.any_horizontal(
                    *(~nw.col(c).is_finite() for c in float_cols), ignore_nulls=True
                ).alias("__is_inf__")
            ).get_column("__is_inf__")
            n_inf = int(is_inf.sum())
            if n_inf > 0:
                df = df.filter(~is_inf)
                warnings.warn(
                    f"{n_inf} rows with infinite values dropped from the model."
                )

        # remove intercept
        if self._fixed_effects is not None or self._drop_intercept:
            if self._independent is not None:
                self._independent = [c for c in self._independent if c != "Intercept"]
            if self._instruments is not None:
                self._instruments = [c for c in self._instruments if c != "Intercept"]

        if self._fixed_effects is not None:
            df = df.with_columns(nw.col(*self._fixed_effects).cast(nw.Int32))

        # drop singletons
        if drop_singletons and self._fixed_effects is not None:
            singleton_mask = detect_singletons(
                df.select(self._fixed_effects).to_numpy()
            )
            is_singleton = nw.Series.from_iterable(
                "__is_singleton__",
                singleton_mask.tolist(),
                backend=backend,
            )
            n_singleton = int(is_singleton.sum())
            if n_singleton > 0:
                df = df.filter(~is_singleton)
                warnings.warn(
                    f"{n_singleton} singleton fixed effect(s) dropped from the model."
                )

        self._data = df.drop("__row_index__").to_native()
        self._na_index = frozenset(range(n)) - frozenset(df["__row_index__"])

    @property
    def dependent(self) -> IntoDataFrameT:
        """
        Get the dependent variable(s) from the model.

        Returns
        -------
        IntoDataFrameT
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
        IntoDataFrameT
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
        IntoDataFrameT or None
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
        IntoDataFrameT or None
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
        IntoDataFrameT or None
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
        IntoDataFrameT or None
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
    data : IntoDataFrameT
        The input data containing all variables referenced in the formula.
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
    formula_formulaic = _get_formulaic_formula(
        formula=formula, data=data, weights=weights
    )

    model_matrix = cast(
        Structured[formulaic.ModelMatrix[IntoDataFrameT]],
        formula_formulaic.get_model_matrix(
            data=data,
            ensure_full_rank=ensure_full_rank,
            # we have custom dropping rules (e.g. infinite values, singleton FES),
            # so we implement our own row-dropping logic later.
            # TODO: at some point, it would be nice to upstream custom row-dropping logic
            na_action="ignore",
            context=FORMULAIC_TRANSFORMS | {**capture_context(context)},
        ),
    )

    return ModelMatrix(
        model_matrix,
        data=data,
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
