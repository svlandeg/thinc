from typing import Tuple, Callable, List, Optional, TypeVar

from ..types import Array
from ..model import Model


InputType = TypeVar("InputType", bound=List[Array])
OutputType = TypeVar("OutputType", bound=List[Array])


def with_square_sequences(layer: Model) -> Model:
    return Model(
        f"with_square_sequences-{layer.name}", forward, init=init, layers=[layer]
    )


def forward(
    model: Model, seqs_in: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    padded_in, _, unpad = model.ops.square_sequences(seqs_in)
    (padded_out, _), backprop_layer = model.layers[0](padded_in, is_train)
    seqs_out = unpad(padded_out)

    def backprop(d_seqs_out: OutputType) -> InputType:
        d_padded_out, sizes_at_t, unpad = model.ops.square_sequences(d_seqs_out)
        d_padded_in = backprop_layer((d_padded_out, None))
        return unpad(d_padded_in)

    return seqs_out, backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    model.layers[0].initialize(
        X=model.ops.square_sequences(X)[0] if X is not None else None,
        Y=model.ops.square_sequences(Y)[0] if Y is not None else None,
    )