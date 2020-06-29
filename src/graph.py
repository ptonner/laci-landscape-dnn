import networkx as nx
from tqdm import tqdm


def neighbors(s, dictionary, alphabet):
    """Find the neighbors of sequence s from a given dictionary.

    :param s: The node to find neighbors for.
    :param dictionary: Possible neighbors, for optimal performance will be converted to a set.
    :param alphabet: Potential point mutation choices.
    :returns: iterator of valid neighbors
    :rtype: iterator

    """

    if not isinstance(dictionary, set):
        dictionary = set(dictionary)

    for j in range(len(s)):
        for d in alphabet:
            nw = "".join([d if i == j else c for i, c in enumerate(s)])
            if nw != s and nw in dictionary:
                yield nw


def hamming(seqs, alphabet, verbose=False):
    """Build a hamming graph of sequences from a given alphabet.

    :param seqs: Sequences to join into a graph
    :param alphabet: Valid mutation choices
    :param verbose: When true, provide progressbar updates
    :returns: A hamming graph of all sequences
    :rtype: nx.Graph

    """

    dictionary = set(seqs)
    G = nx.Graph()

    if verbose:
        pbar = tqdm(total=len(seqs))

    for i, s in enumerate(seqs):
        n = neighbors(s, dictionary, alphabet)
        for nn in n:
            G.add_edge(s, nn)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    return G


def findPathsofLength(G, u, n):
    """Find the paths from u of length n.

    Adapted from
    https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph

    """
    if n == 0:
        return [[u]]
    paths = [
        [u] + path
        for neighbor in G.neighbors(u)
        for path in findPathsofLength(G, neighbor, n - 1)
        if u not in path
    ]
    return paths


def connection(s1, s2):
    """A connection graph from s1 to s2

    :param s1: sequence 1
    :param s2: sequence 2
    :returns: A hamming graph of all paths from s1 to s2
    :rtype: nx.Graph

    """

    miss = [i for i, (ss1, ss2) in enumerate(zip(list(s1), list(s2))) if not ss1 == ss2]

    unfinished = [s1]

    G = nx.Graph()
    while len(unfinished) > 0:
        s = unfinished.pop(0)

        for i in miss:
            if s2[i] == s[i]:
                continue
            ss = s[:i] + s2[i] + s[i + 1 :]
            if ss not in G:
                unfinished.append(ss)

            G.add_edge(s, ss)

    return G
