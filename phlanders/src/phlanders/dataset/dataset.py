import attr
import hashlib
import pandas as pd
from typing import Dict, Optional
import numpy as np
import torch

from phlanders.serial import Serializable
from phlanders.register import _Register
from phlanders.dataset import Feature

import logging

logger = logging.getLogger(__name__)


class _Dataset(_Register):
    """The register metaclass for datasets.
    """


@attr.s
class Dataset(Serializable, metaclass=_Dataset):

    collate = None

    def __attrs_post_init__(self):
        self._initialized = False

    def initialize(self, device="cpu", *args, **kwargs):
        """Prepare the dataset for learning, e.g. build tensors.
        """
        if not self._initialized:
            self._initialize(device, *args, **kwargs)
        self._initialized = True

    def _initialize(self, device="cpu", *args, **kwargs):
        """If necessary, initialize the data.
        """

    def validate(self, *args, **kwargs):
        """Optional validation method for the dataset.
        """
        pass

    def to_tensor_dataset(self,):
        raise NotImplementedError()


@attr.s
class PandasDataset(torch.utils.data.Dataset):
    """Base class for pandas based datasets.

    Must have an attribute df pointing to the pandas dataframe.
    """

    features: Dict[str, Feature] = attr.ib()
    pretransferDevice: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        super(PandasDataset, self).__attrs_post_init__()

        self._feature_order = self.features.keys()
        logger.debug("Dataset feature order: {}".format(self._feature_order))

    def _initialize(self, device, *args, **kwargs):

        extract = {
            k: self._extract_feature(self.features[k], self.df)
            for k in self._feature_order
        }

        if not self.pretransferDevice:
            device = "cpu"

        self._datapoints = {}
        for k in self._feature_order:
            ext = extract[k]
            if isinstance(ext, list):
                self._datapoints[k] = [torch.Tensor(dp).to(device) for dp in ext]
            elif isinstance(ext, np.ndarray):
                self._datapoints[k] = torch.from_numpy(ext).to(device)
            else:
                raise TypeError(
                    "Cannot handle extracted feature of type {}".format(type(ext))
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # fix for variable length slices
        if isinstance(idx, (slice, np.ndarray)):
            ret = []

            for k in self._feature_order:

                tmp = self._datapoints[k]
                if isinstance(tmp, torch.Tensor):
                    ret.append(tmp[idx])
                else:
                    if isinstance(idx, slice):
                        start = idx.start
                        stop = idx.stop
                        step = idx.step
                        bs = (stop - start) // step
                        rng = range(start, stop, step)
                        batch = [self._datapoints[k][i] for i in rng]
                    else:
                        bs = len(idx)
                        batch = [self._datapoints[k][i] for i in idx]

                    mx = max([b.shape[0] for b in batch])
                    rest = batch[0].shape[1:]
                    tmp = torch.zeros((bs, mx) + rest)
                    slc = (slice(None),) * (batch[0].ndim - 1)

                    for j, b in enumerate(batch):
                        k = b.shape[0]
                        tmp[(j,) + (slice(0, k),) + slc] = b

                    ret.append(tmp)

            return ret

        return [self._datapoints[k][idx] for k in self._feature_order]

    @staticmethod
    def _extract_feature(feat, df):
        if isinstance(feat, list):
            ret = np.column_stack([ff.extract(df) for ff in feat])
        else:
            ret = feat.extract(df)

        return ret

    def to_tensor_dataset(self):

        featureOrder = self.features.keys()
        logger.debug("Dataset feature order: {}".format(featureOrder))

        return torch.utils.data.TensorDataset(
            *[
                torch.from_numpy(self._extract_feature(self.features[f], self.df))
                for f in featureOrder
            ]
        )

    def collate(self, batch):
        """A collation function to be used by torch DataLoader's
        """

        ret = {}

        for i, k in enumerate(self._feature_order):
            f = self.features[k]
            if f._fixed_length:
                # ret.append(torch.cat(
                #     [b[i][None] for b in batch])
                # )
                ret[k] = torch.cat([b[i][None] for b in batch])
            else:
                # assuming that the first dimension is the dynamic
                # one, will have to change if this is ever not true
                bs = len(batch)
                mx = max([b[i].shape[0] for b in batch])
                rest = batch[0][i].shape[1:]
                tmp = torch.zeros((bs, mx) + rest)
                slc = (slice(None),) * (batch[0][i].ndim - 2)

                for j, b in enumerate(batch):
                    b = b[i]
                    kk = b.shape[0]
                    tmp[(j,) + (slice(0, kk),) + slc] = b

                ret[k] = tmp

        return ret


@attr.s
class DataframeDataset(PandasDataset, Dataset):
    """Create pandas dataset directly from dataframe.
    """

    df: pd.DataFrame = attr.ib(default=None)


@attr.s
class CsvDataset(PandasDataset, Dataset):
    """Load a pandas dataframe dataset from a csv.
    """

    path: str = attr.ib(default=None)
    pd_kwargs: Dict = attr.ib(factory=dict)
    expected_md5: Optional[str] = attr.ib(default=None)

    def __attrs_post_init__(self):
        super(CsvDataset, self).__attrs_post_init__()

        self.df = pd.read_csv(self.path, **self.pd_kwargs)
        # self.initialize()

    @property
    def md5(self,):
        with open(self.path, "rb") as f:
            m = hashlib.md5(f.read())
            return m.hexdigest()

    def validate(self):
        if self.expected_md5 is not None:
            if not self.md5 == self.expected_md5:
                raise ValueError(
                    """Expected md5 {} did not match computed {}. If you are sure this is
                                 the data you want to use, update or
                                 remove the expected_md5 field.""".format(
                        self.expected_md5, self.md5
                    )
                )
