import attr
from torch import nn
import numpy as np
from phlanders.register import _Register
from phlanders.serial import Serializable


class _Model(_Register):
    """The model registry; all models are accessed through here for
    reconstruction.

    """


@attr.s(eq=False)
class Model(nn.Module, Serializable, metaclass=_Model):
    def loss(self, *args, **kwargs):
        """The loss function for this model.
        """
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        """The forward method of this model.
        """
        raise NotImplementedError()

    def __attrs_post_init__(self):
        super(Model, self).__init__()

    def reset(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    @property
    def num_parameters(self):
        return sum([np.product(p.shape) for p in self.parameters()])
