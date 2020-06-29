from phlanders.model import Model

import attr
import torch
from torch.nn import functional as F


@attr.s(eq=False)
class Gaussian(Model):
    """A model of Gaussian prediction of outcomes, constructed from the
    prediction of a base model.

    """

    base: Model = attr.ib()
    tol: float = attr.ib(default=1e-6)

    def forward(self, *args, **kwargs):
        z = self.base(*args, **kwargs)
        K = z.shape[1]

        if not K % 2 == 0:
            raise ValueError("Must have even number of outputs!")

        # split output into mean and variance
        mu = z[:, : K // 2]
        var = self.tol + F.softplus(z[:, K // 2 :] + 1)

        return mu, var

    def loss(self, sequence, y, weights=None, *args, **kwargs):

        mu, var = self(sequence)
        y = y.reshape_as(mu)

        # Gaussian negative log-likelihood
        nll = torch.log(var) / 2 + (mu - y.float()) ** 2 / 2 / var

        if weights is None:
            return torch.mean(nll)
        else:
            if weights.ndim == 1:
                weights = weights[:, None]
            return torch.mean(weights * nll)
