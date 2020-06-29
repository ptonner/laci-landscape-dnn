from torch import nn
from typing import Dict, List, Optional
import attr

_SUPPORTED = {
    "linear": nn.Linear,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "conv1d": nn.Conv1d,
    "conv": nn.Conv1d,
    "maxpool": nn.MaxPool1d,
    "batchnorm": nn.BatchNorm1d,
    "dropout": nn.Dropout,
    "flatten": nn.Flatten,
}


@attr.s
class Layer:

    kind: str = attr.ib()
    args: Optional[List] = attr.ib(factory=list)
    kwargs: Optional[Dict] = attr.ib(factory=dict)

    @staticmethod
    def _find(kind):
        if kind.lower() in _SUPPORTED:
            return _SUPPORTED[kind.lower()]

        raise ValueError("Unknown layer {}!".format(kind))

    def build(self):
        kind, args, kwargs = attr.astuple(self)
        cls = self._find(kind)
        return cls(*args, **kwargs)


@attr.s
class LayerList:

    layers: List[Layer] = attr.ib()

    def build(self):
        return nn.ModuleList([l.build() for l in self.layers])


# def build_layer(layer: Layer):
#     """Build a single layer.
#     """
#     kind, args, kwargs = attr.astuple(layer)
#     return lookup[kind](*args, **kwargs)
#
#
# def build_layers(layers: List[Layer]):
#     """Build a module list from the supplied layers.
#     """
#
#     lay = []
#     for kind, args, kwargs in map(attr.astuple, layers):
#         lay.append(lookup[kind](*args, **kwargs))
#     return nn.ModuleList(lay)
