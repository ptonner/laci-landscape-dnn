from phlanders.register import _Register
from phlanders.serial import Serializable
from phlanders.dataset import Dataset
from phlanders.model import Model
from phlanders.experiment import Sampler

import attr
import os
import torch
from typing import Dict, Optional


class _Experiment(_Register):
    """The experiment metaclass.
    """


@attr.s
class Experiment(Serializable, metaclass=_Experiment):

    model: Model = attr.ib()
    dataset: Dataset = attr.ib()
    epochs: int = attr.ib()
    batch_size: int = attr.ib(default=128)
    verbose: int = attr.ib(default=1)
    optimizer: Dict = attr.ib(factory=dict)
    lr_gamma: float = attr.ib(default=1.0)
    lr_step: int = attr.ib(default=0)
    batchsize_norm: bool = attr.ib(default=False)
    sampler: Optional[Sampler] = attr.ib(default=None)
    weighter: Optional[Sampler] = attr.ib(default=None)

    def __attrs_post_init__(self):
        self._device = None

    def train(self, device, *args, **kwargs):
        self._device = device
        self.model.to(device)
        self.dataset.initialize(device)
        self._train(*args, **kwargs)

    def _train(self, *args, **kwargs):
        raise NotImplementedError()

    def finish(self, *args, **kwargs):
        pass

    def save_weights(self, model, path):
        dname = os.path.dirname(path)
        os.makedirs(dname, exist_ok=True)

        torch.save(model.state_dict(), path)

    def load_weights(self, path, **kwargs):
        self.model.load_state_dict(torch.load(path, **kwargs))
        return self.model

    @property
    def P(self):
        """The number of model/dataset pairs in this experiment.
        """
        raise NotImplementedError()

    def _check_pair(self, p):
        if p < 0 or p >= self.P:
            raise ValueError(
                "Invalid pair ({}) for experiment with size {}!".format(p, self.P)
            )

    def pair_parameters(self, p, *args, **kwargs):
        """Return the file name of the pth model/dataset pair.
        """
        self._check_pair(p)
        return "pair-{}.pt"

    def pair_data(self, p, validation=False, *args, **kwargs):
        raise NotImplementedError()
