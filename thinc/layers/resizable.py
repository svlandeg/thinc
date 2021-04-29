from typing import Callable, Optional, TypeVar

from ..model import Model
from ..config import registry


InT = TypeVar("InT")
OutT = TypeVar("OutT")

NEG_VALUE = -3000


@registry.layers("resizable.v1")
def resizable(layer_creation: Callable) -> Model[InT, OutT]:
    """TODO."""
    layer = layer_creation()
    return Model(
        f"resizable({layer.name})",
        forward,
        init=init,
        layers=[layer],
        attrs={"layer_creation": layer_creation, "resize_output": resize},
        dims={name: layer.maybe_get_dim(name) for name in layer.dim_names}
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool):
    layer = model.layers[0]
    Y, callback = layer(X, is_train=is_train)

    def backprop(dY: OutT) -> InT:
        return callback(dY)

    return Y, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    layer = model.layers[0]
    layer.initialize(X, Y)
    return model


def resize(model, new_nO, resizable_layer):
    old_layer = resizable_layer.layers[0]
    if old_layer.has_dim("nO") is None:
        # the output layer had not been initialized/trained yet
        old_layer.set_dim("nO", new_nO)
        return model
    elif new_nO == old_layer.get_dim("nO"):
        # the output dimension didn't change
        return model

    # initialize the new layer
    layer_creation = resizable_layer.attrs["layer_creation"]
    new_layer = layer_creation()
    new_layer.set_dim("nO", new_nO)
    if old_layer.has_dim("nI"):
        new_layer.set_dim("nI", old_layer.get_dim("nI"))
    new_layer.initialize()

    if old_layer.has_param("W"):
        larger_W = new_layer.get_param("W")
        larger_b = new_layer.get_param("b")
        smaller_W = old_layer.get_param("W")
        smaller_b = old_layer.get_param("b")
        larger_W[: len(smaller_W)] = smaller_W
        larger_b[: len(smaller_b)] = smaller_b
        # TODO: RELU instead
        if "activation" in model.attrs and model.attrs["activation"] in [
            "softmax",
            "logistic",
        ]:
            # ensure little influence on the softmax/logistic activation
            larger_b[len(smaller_b):] = NEG_VALUE
        new_layer.set_param("W", larger_W)
        new_layer.set_param("b", larger_b)

    model.layers[0] = new_layer
    return model
