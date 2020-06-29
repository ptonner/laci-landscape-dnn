from phlanders.model import Model, LayerList

import attr
import torch
from torch.nn import functional as F


@attr.s(eq=False)
class Regression(Model):

    layers: LayerList = attr.ib()

    def __attrs_post_init__(self,):
        super(Regression, self).__attrs_post_init__()

        self._layers = self.layers.build()

    def forward(self, sequence, *args, **kwargs):
        x = sequence.float()
        for l in self._layers:
            x = l(x)

        return x

    def loss(self, sequence, y, weights=None, *args, **kwargs):

        yhat = self(sequence)
        y = y.reshape_as(yhat)

        if weights is None:
            return F.mse_loss(yhat, y.float())
        else:
            if weights.ndim == 1:
                weights = weights[:, None]
            return torch.mean(weights * (yhat - y.float()) ** 2)
