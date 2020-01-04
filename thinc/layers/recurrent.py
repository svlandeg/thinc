from typing import Tuple
from ..model import Model
from ..types import Array


def recurrent(step_model: Model) -> Model:
    model = Model(
        step_model.name,
        forward,
        init=init,
        params={"initial_cells": None, "initial_hiddens": None},
        dims={"nO": step_model.get_dim("nO") if step_model.has_dim("nO") else None},
        layers=[step_model]
    )
    if model.has_dim("nO"):
        model.initialize()
    return model


def init(model, X=None, Y=None):
    Xt = X[0][0] if X is not None else None
    Yt = Y[0][0] if Y is not None else None
    if Xt is not None or Yt is not None:
        model.layers[0].initialize(X=Xt, Y=Yt)
    nO = model.get_dim("nO")
    model.set_param("initial_cells", model.ops.allocate((nO,)))
    model.set_param("initial_hiddens", model.ops.allocate((nO,)))


def forward(model: Model, X_size_at_t: Tuple[Array, Array], is_train: bool):
    # Expect padded batches, sorted by decreasing length. The size_at_t array
    # records the number of batch items that are still active at timestep t.
    X, size_at_t = X_size_at_t
    step_model = model.layers[0]
    nO = step_model.get_dim("nO")
    Y = model.ops.allocate((X.shape[0], X.shape[1], nO))
    backprops = [None] * X.shape[0]
    (cell, hidden) = _get_initial_state(model, X.shape[1], nO)
    for t in range(X.shape[0]):
        # At each timestep t, we finish some of the sequences. The sequences
        # are arranged longest to shortest, so we can drop the finished ones
        # off the end.
        n = size_at_t[t]
        inputs = ((cell[:n], hidden[:n]), X[t, :n])
        ((cell, hidden), Y[t, :n]), backprops[t] = step_model(inputs, is_train)

    def backprop(dY_size_at_t):
        dY, size_at_t = dY_size_at_t
        d_cell = step_model.ops.allocate((dY.shape[1], nO)),
        d_hidden = step_model.ops.allocate((dY.shape[1], nO)),
        dX = step_model.ops.allocate((dY.shape[0], dY.shape[1], nI))
        for t in range(dX.shape[0] - 1, -1, -1):
            # Is this right? It feels strange to pass in the too-long buffer.
            # Can't we just pass in d_cell_t??
            (d_cell_t, d_hidden_t), dXt = backprops[t](((d_cell, d_hidden), dY[t]))
            d_cell[: d_state_t[0].shape[0]] = d_state_t[0]
            d_hidden[: d_state_t[1].shape[0]] = d_state_t[1]
            dX[t, : dXt.shape[0]] = dXt
        step_model.inc_grad("initial_cells", d_cell.sum(axis=0))
        step_model.inc_grad("initial_hiddens", d_hidden.sum(axis=0))
        return (dX, size_at_t)

    return (Y, size_at_t), backprop


def _get_initial_state(model, n, nO):
    initial_cells = model.ops.allocate((n, nO))
    initial_hiddens = model.ops.allocate((n, nO))
    initial_cells += model.get_param("initial_cells")
    initial_hiddens += model.get_param("initial_hiddens")
    return (initial_cells, initial_hiddens)
