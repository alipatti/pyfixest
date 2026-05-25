import functools
import itertools
from typing import Final

import narwhals.stable.v1 as nw
import pandas as pd
from formulaic.parser import DefaultOperatorResolver
from formulaic.parser.types import Operator, OrderedSet
from formulaic.utils.stateful_transforms import stateful_transform
from narwhals.typing import IntoSeries


@stateful_transform
def encode_fixed_effects(*args, _state=None, _metadata=None, _spec=None):
    """Encode fixed effect interactions for model matrix construction."""
    data = pd.concat(args, axis=1)
    _encoding: Final[str] = "__fixed_effect_encoding__"
    if _encoding not in _state:
        data[_encoding] = data.groupby(data.columns.tolist()).ngroup()
        _state[_encoding] = data.drop_duplicates()
        return data[_encoding]

    return data.merge(_state[_encoding], on=data.columns.tolist(), how="left")[
        _encoding
    ]


# TODO: finish this
# TODO: type these args
def encode_fixed_effects_narwhals(
    *args: IntoSeries, _state=None, _metadata=None, _spec=None
):
    _encoding: Final[str] = "__fixed_effect_encoding__"

    data = nw.concat(
        [nw.from_native(s, series_only=True).to_frame() for s in args],
        how="horizontal",
    )

    raise NotImplementedError


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
