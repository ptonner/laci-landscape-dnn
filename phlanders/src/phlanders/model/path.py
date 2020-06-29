import attr
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from phlanders.model import Model

import logging

logger = logging.getLogger(__name__)


@attr.s(eq=False)
class Path(Model):
    """An path model of fitness, predicting each step-wise fitness change.

    :param rnn: The recurrent neural network architecture to use (RNN, LSTM, GRU)
    :type rnn: str
    :param loci: The number of loci in the input space (e.g. 4xL for a length L nucleotide sequence.)
    :type loci: int
    :param hidden: The dimensionality of the hidden dimension.
    :type hidden: int
    """

    rnn: str = attr.ib(validator=attr.validators.in_(["rnn", "lstm", "gru"]))
    loci: int = attr.ib()
    hidden: int = attr.ib()
    k: int = attr.ib(default=1)
    num_layers: int = attr.ib(default=1)
    nonlinearity: str = attr.ib(default="tanh")
    bias: bool = attr.ib(default=True)
    dropout: float = attr.ib(default=0)
    bidirectional: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        super(Path, self).__attrs_post_init__()

        if self.rnn == "rnn":
            self._rnn = nn.RNN(
                input_size=self.loci,
                hidden_size=self.hidden,
                num_layers=self.num_layers,
                nonlinearity=self.nonlinearity,
                bias=self.bias,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        elif self.rnn == "lstm":
            self._rnn = nn.LSTM(
                input_size=self.loci,
                hidden_size=self.hidden,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        elif self.rnn == "gru":
            self._rnn = nn.GRU(
                input_size=self.loci,
                hidden_size=self.hidden,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

        # we want zero-biased linear layer because the null outputs
        # from the RNN will then remain zero after passing through
        # this layer
        self.linear = nn.Linear(self.hidden, self.k, bias=False)

    def forward(self, X, lengths=None, includeHistory=False):
        # X should be shape (bs, L, L*M) for batch size bs, sequence
        # length L and dimensionality L*M with alphabet length M

        if 1 < X.ndim < 3:
            logger.warning("Missing batch dimension")
            X = X[None, :, :]

        # reconstruct the lengths of each sequence path if necessary
        try:
            if lengths is None:
                if X.shape[1] == 0:
                    lengths = torch.ones(X.shape[0]).int()
                    X = torch.zeros((X.shape[0], 1, X.shape[2])).to(X.device)

                else:
                    tst = X != 0
                    tst = tst.any(2)
                    # _, ind_old = torch.max(tst, dim=1)
                    # _, ind = torch.max(tst.any(2), dim=1)
                    ind = torch.arange(tst.shape[1]).repeat(tst.shape[0], 1)
                    ind[~tst] = -1
                    ind, _ = torch.max(ind, dim=1)
                    lengths = ind + 1
                    # todo: this doesn't handle 0 length correctly
                    lengths = torch.max(lengths, torch.ones_like(lengths))
        except:
            print(X.shape)
            raise

        # randomize order if training
        if self.training:
            for i in range(X.shape[0]):
                ln = lengths[i].item()
                ind = np.arange(ln)
                order = np.random.choice(ind, ln, replace=False)
                X[i, :ln] = X[i, order]

        # build the packed data for the model
        try:
            pck = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        except:
            print(X.shape)
            raise

        pck = pck.to(X.device)

        # compute path
        out, hstate = self._rnn(pck)

        # reconstruct original shape
        output, _lengths = pad_packed_sequence(out, batch_first=True)
        # output = output.to(X.device)

        # check things make sense
        assert (lengths == _lengths.to(lengths.device)).all()

        # find the individual deltas for each path step and then
        # aggregate
        deltas = self.linear(output)
        delta = deltas.sum(axis=1)

        if includeHistory:
            return delta, deltas, hstate, output

        return delta, deltas, hstate

    def loss(self, sequence, y, weights=None, *args, **kwargs):

        yhat, _, _ = self(sequence)
        y = y.reshape_as(yhat)
        # return F.mse_loss(yhat, y.float())
        if weights is None:
            return F.mse_loss(yhat, y.float())
        else:
            if weights.ndim == 1:
                weights = weights[:, None]
            return torch.mean((weights * (yhat - y.float())) ** 2)
