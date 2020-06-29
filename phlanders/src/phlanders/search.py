import numpy as np


def library(seeds, mutation_rate, alphabet, size=None):
    """Generate a randomized library from the provided seed(s)
    """
    alphabet = set(alphabet)

    if not isinstance(seeds, list):
        seeds = [seeds]

    if size is None:
        size = len(seeds)

    i = np.random.choice(np.arange(len(seeds)), size, replace=True)

    nw = []
    for ii in i:
        s = seeds[ii]
        muts = np.random.binomial(len(s), mutation_rate)
        pos = np.random.choice(np.arange(len(s)), muts)
        tmp = s
        for p in pos:
            a = np.random.choice(list(alphabet - set(tmp[p])))
            tmp = tmp[:p] + a + tmp[p+1:]

        nw.append(tmp)

    return nw


def search():

    # build initial library

    # for a given number of iterations

        # select based on design target

        # cross/randomize

        # filter out those too far away, and replace
