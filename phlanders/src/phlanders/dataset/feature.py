import attr
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from itertools import permutations

from phlanders.serial import Serializable
from phlanders.register import _Register


def _reshape(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values

    if isinstance(x, np.ndarray):
        x = np.stack(x.tolist())
        if x.shape[0] == 1:
            x = x[0]
    else:
        if len(x) == 1:
            x = x[0]

    return x


class _Feature(_Register):
    """The register metaclass of dataset features.
    """


@attr.s
class Feature(Serializable, metaclass=_Feature):
    """
    """

    def extract(self, df):
        return _reshape(self._extract(df))

    def _extract(self, df):
        """The extraction method for a given feature from a dataframe.
        """
        raise NotImplementedError()

    def __mul__(self, other):
        return Product(self, other)

    @property
    def _fixed_length(self):
        """Determine whether a feature has fixed length extracted values.

        Currently only False for SequencePath features with a certain
        configuration.

        """
        return True


@attr.s
class Column(Feature):
    """A basic CSV feature, just use the column as is.
    """

    name: str = attr.ib(validator=attr.validators.instance_of(str))

    def _extract(self, df):
        return df[self.name]


@attr.s
class Product(Feature):

    feature1: Feature = attr.ib()
    feature2: Feature = attr.ib()

    def _extract(self, df):

        return self.feature1._extract(df) * self.feature2._extract(df)


@attr.s
class Stacked(Feature):
    """Combine features into a multidimensional array.
    """

    features: List[Feature] = attr.ib()

    def _extract(self, df):
        return np.column_stack([f.extract(df) for f in self.features])


@attr.s
class Compound(Feature):
    """A feature built on another, basic building block.
    """

    feature: Feature = attr.ib()

    def _extract(self, df):
        return self._extract_compound(self.feature._extract(df))

    def _extract_compound(self, df):
        """Additional transform on the child feature.
        """
        raise NotImplementedError()


@attr.s
class ShiftFeature(Compound):
    shift: float = attr.ib()

    def _extract_compound(self, df):
        return df + self.shift


@attr.s
class RangeNormFeature(Compound):

    lo: int = attr.ib(default=0)
    hi: int = attr.ib(default=1)

    @classmethod
    def from_column(cls, df, col):
        return cls(Column(col), df[col].min(), df[col].max())

    def _extract_compound(self, df):
        return (df - self.lo) / (self.hi - self.lo)


@attr.s
class GaussNormFeature(Compound):

    mean: float = attr.ib()
    var: float = attr.ib()

    @classmethod
    def from_column(cls, df, col):
        return cls(Column(col), df[col].values.mean(), df[col].values.var())

    @classmethod
    def from_feature(cls, df, feature):
        return cls(feature, feature.extract(df).mean(), feature.extract(df).var())

    def _extract_compound(self, df):

        return (df - self.mean) / np.sqrt(self.var)


@attr.s
class LogFeature(Compound):

    base: float = attr.ib()

    def _extract_compound(self, df):

        return np.log(df) / np.log(self.base)


@attr.s
class BinaryFeature(Compound):

    threshold: float = attr.ib()

    def _extract_compound(self, df):

        return (df > self.threshold).astype(int)


@attr.s
class SequenceFeature(Column):
    """A sequence feature in a fitness dataset.
    """

    alphabet: str = attr.ib()
    flatten: bool = attr.ib(default=False)
    slices: List[Tuple[int, int, int]] = attr.ib(factory=list)
    variants: Optional[List[Tuple[str, List[str]]]] = attr.ib(default=None)

    @staticmethod
    def to_categorical(z, M):
        L = len(z)
        ret = np.zeros((M, L), dtype="f")
        ind = np.arange(L)
        ret[z, ind] = 1
        return ret

    @property
    def M(self):
        """The alphabet size.
        """
        return len(self.alphabet)

    def _lookup(self, s):
        if s not in self.alphabet:
            raise ValueError("{} not found in {}!".format(s, self.alphabet))

        return self.alphabet.index(s)

    def onehot(self, s):
        ind = [self._lookup(ss) for ss in s]
        return SequenceFeature.to_categorical(ind, len(self.alphabet))

    def _extract(self, df):
        return df[self.name].apply(self.x)

    def x(self, s):
        if len(self.slices) > 0:
            s = "".join(s[slice(*sl)] for sl in self.slices)

        if self.variants is None:
            x = self.onehot(s)
            if self.flatten:
                return x.T.ravel()
        else:
            x = self.variant_code(s)

        return x

    def str(self, x):

        if self.variants is None:
            if self.flatten:
                x = x.reshape((x.shape[0] // self.M, self.M)).T

            ind, pos = np.where(x)
            ind = ind[np.argsort(pos)]
            return "".join([self.alphabet[i] for i in ind])

        x = x.tolist()
        res = []
        for wt, v in self.variants:
            tmp = x[: len(v)]
            x = x[len(v) :]

            ind, = np.where(tmp)
            if len(ind) > 0:
                res.append(v[ind[0]])
            else:
                res.append(wt)
        return "".join(res)

    def variant_code(self, s):
        if self.variants is None:
            raise ValueError("No variant information!")

        K = sum([len(v) for _, v in self.variants])
        code = np.zeros(K)
        offset = 0

        for (_, v), ss in zip(self.variants, s):
            if ss in v:
                code[offset + v.index(ss)] = 1
            offset += len(v)

        return code

    @classmethod
    def build_variants(cls, source, sequences):

        variants = []

        for i in range(len(source)):
            src = source[i]
            st = set()
            for seq in sequences:
                if seq[i] == src:
                    continue
                st.add(seq[i])
            variants.append((src, list(st)))

        return variants


@attr.s
class SequencePath(SequenceFeature):

    source = attr.ib(default=None)

    # deprecated
    fill = attr.ib(default=False)
    depth = attr.ib(default=-1)

    @property
    def _source_x(self,):
        return super(SequencePath, self).x(self.source)

    def x(self, s):
        if not len(s) == len(self.source):
            raise ValueError(
                "Length of input({}) does not equal source({})".format(
                    len(s), len(self.source)
                )
            )

        miss = [
            i
            for i, (ss, sr) in enumerate(zip(list(s), list(self.source)))
            if not ss == sr
        ]
        dist = len(miss)
        enc = super(SequencePath, self).x(s)

        ret = np.zeros((dist,) + enc.shape, dtype="f")

        for i, ms in enumerate(miss):
            tmp = self.source[:ms] + s[ms] + self.source[ms + 1 :]
            ret[i, :] = super(SequencePath, self).x(tmp) - self._source_x

        return ret

    def from_path(self, pth):

        if pth[0] == self.source:
            pth = [p for p in pth[1:]]

        miss = len(pth)
        L = len(self.source)
        enc = super(SequencePath, self).x(pth[0])
        ret = np.zeros((L,) + enc.shape, dtype="f")

        prev = self.source
        for i in range(miss):
            ret[i, :] = super(SequencePath, self).x(pth[i]) - super(
                SequencePath, self
            ).x(prev)
            prev = pth[i]

        return ret

    def str(self, x):

        # rebuild the parent style x
        x = x.sum(axis=0)
        x = x + self._source_x

        return super(SequencePath, self).str(x)

    def _extract(self, df):
        return [self.x(rw) for rw in df[self.name]]

    @property
    def _fixed_length(self):
        # return self.fill or (self.depth > 0)
        return False

    def paths(self, s):
        """All paths to sequence s.
        """

        miss = [
            i
            for i, (ss, sr) in enumerate(zip(list(s), list(self.source)))
            if not ss == sr
        ]

        for perm in permutations(miss):
            tmp = self.source
            pth = [tmp]
            for p in perm:
                tmp = tmp[:p] + s[p] + tmp[p + 1 :]
                pth.append(tmp)
            yield pth
