import functools
import itertools
from typing import Final

import narwhals.stable.v1 as nw
from formulaic.parser import DefaultOperatorResolver
from formulaic.parser.types import Operator, OrderedSet
from formulaic.utils.stateful_transforms import stateful_transform
from narwhals.stable.v1.typing import IntoSeriesT


@stateful_transform
def encode_fixed_effects(*args: IntoSeriesT, _state: dict) -> IntoSeriesT:
    """Encode fixed effect interactions for model matrix construction."""
    _encoding: Final[str] = "__fixed_effect_encoding__"

    # concat series into a single df
    data = nw.concat(
        [nw.from_native(s, series_only=True).to_frame() for s in args],
        how="horizontal",
    )

    if _encoding not in _state:
        # get mapping from FEs -> integers
        # sort so that is stable across runs
        # (not strictly necessary but is nice and has negligible perf. overhead)
        _state[_encoding] = (
            data.unique().sort(data.columns).with_row_index(name=_encoding)
        )

    return (
        data.join(_state[_encoding], on=data.columns, how="left")
        .get_column(_encoding)
        .to_native()
    )


class _FixedEffectsOperatorResolver(DefaultOperatorResolver):
    def __init__(self):
        super().__init__()

    @property
    def operators(self) -> list[Operator]:
        operators = [
            operator for operator in super().operators if operator.symbol != "^"
        ]

        operators.append(
            Operator(
                symbol="^",
                arity=2,
                precedence=500,
                associativity="left",
                to_terms=lambda *term_sets: OrderedSet(
                    functools.reduce(lambda x, y: x * y, term)
                    for term in itertools.product(*term_sets)
                ),
            )
        )
        return operators
