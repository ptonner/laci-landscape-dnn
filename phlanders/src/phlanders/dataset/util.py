import numpy as np


def abundance(z, bins=10, *args, **kwargs):
    """Calculate the approximate "abundance" of continuous array z values.

    The abundance here is an approximation of distinct values across a
    multi-dimensional space defined by z. For each observation of z,
    the approximate count of similar values is computed based on a
    computed histogram. This can be used for weighting observations by
    their approximate abundance.

    :param z: The feature values to calculate an abundance for.
    :type z: (N, D) array
    :param bins: The number of bins to use for calculating abundance.
    :type bins: int

    """
    # build a multi-dim histogram and edge values
    h, edges = np.histogramdd(z, bins=bins)

    if z.ndim == 1:
        z = z[:, None]

    # for each dimension independently, find the matching edge
    # position for each observation
    ind = []
    for i in range(z.shape[1]):
        dig = np.digitize(z[:, i], edges[i], right=False) - 1

        # cleanup rightmost edge, which will match the highest edge
        # value (and be an out of bounds index)
        dig = [min(d, h.shape[i] - 1) for d in dig]

        ind.append(dig)

    # using indices, find the count of each observation
    ret = []
    for tup in zip(*ind):
        ret.append(h[tup])

    return np.array(ret)


def weight(z, log=True, invert=True, *args, **kwargs):
    a = abundance(z, *args, **kwargs)

    if invert:
        a = 1 / a

    if log:
        a = np.log10(a)

    return a
