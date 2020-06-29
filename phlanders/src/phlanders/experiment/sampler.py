from phlanders.register import _Register
from phlanders.dataset.util import weight
from typing import Optional
import attr

from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset


def _get_component(data, component, index):

    if isinstance(data, Dataset):
        d = data[: len(data)]
        d = d[index]
    elif isinstance(data, list):
        d = data[index]
    elif isinstance(data, dict):
        d = data[component]
    else:
        raise ValueError("Unknown data type {}!".format(type(data)))

    d = d.detach().cpu().numpy()

    return d


class _Sampler(_Register):
    """The Sampler metaclass.
    """


@attr.s
class Sampler(metaclass=_Sampler):
    """A data sampler method.
    """

    def weights(self, data, *args, **kwargs):
        raise NotImplementedError()

    def _build(self, data, batch_size, *args, **kwargs):
        raise NotImplementedError()

    def build(self, data, *args, **kwargs):
        w = self.weights(data, *args, **kwargs)
        if w.ndim > 1:
            w = w.sum(axis=1)

        return WeightedRandomSampler(w, len(data))


@attr.s
class Abundance(Sampler):

    log: bool = attr.ib()
    bins: int = attr.ib()
    component: str = attr.ib()
    index: int = attr.ib()
    scale: Optional[float] = attr.ib(default=1.0)

    def weights(self, data, *args, **kwargs):

        d = _get_component(data, self.component, self.index)
        w = weight(d, log=self.log, bins=self.bins, invert=True)

        w = self.scale * w
        return w


@attr.s
class Component(Sampler):

    component: str = attr.ib()
    index: int = attr.ib()
    invert: bool = attr.ib()

    def weights(self, data, *args, **kwargs):
        w = _get_component(data, self.component, self.index)
        if self.invert:
            return 1 / w
        return w
