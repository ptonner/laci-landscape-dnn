import attr
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from phlanders.model import Path

import logging

logger = logging.getLogger(__name__)


@attr.s(eq=False)
class bLSTM(nn.Module):

    loci: int = attr.ib()
    hidden: int = attr.ib()
    bias: bool = attr.ib(default=True)
    mu_init: float = attr.ib(default=1)
    rho_init: Tuple[float, float] = attr.ib(default=(-3, -1))
    prior_pi: float = attr.ib(default=0.5)
    prior_sigma_1: float = attr.ib(default=1)
    prior_sigma_2: float = attr.ib(default=0.001)
    stable_loss: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        super(bLSTM, self).__init__()

        self._update_parameters()

        self.mix1 = torch.distributions.Normal(0, self.prior_sigma_1)
        self.mix2 = torch.distributions.Normal(0, self.prior_sigma_2)

    def _update_parameters(self, device=None):

        self.weight_ih_mu = nn.Parameter(
            torch.Tensor(self.loci, self.hidden * 4)
            .uniform_(-self.mu_init, self.mu_init)
            .to(device)
        )
        self.weight_ih_rho = nn.Parameter(
            torch.Tensor(self.loci, self.hidden * 4).uniform_(*self.rho_init).to(device)
        )

        self.weight_hh_mu = nn.Parameter(
            torch.Tensor(self.hidden, self.hidden * 4)
            .uniform_(-self.mu_init, self.mu_init)
            .to(device)
        )
        self.weight_hh_rho = nn.Parameter(
            torch.Tensor(self.hidden, self.hidden * 4)
            .uniform_(*self.rho_init)
            .to(device)
        )

        self.bias_mu = nn.Parameter(
            torch.Tensor(self.hidden * 4)
            .uniform_(-self.mu_init, self.mu_init)
            .to(device)
        )
        self.bias_rho = nn.Parameter(
            torch.Tensor(self.hidden * 4).uniform_(*self.rho_init).to(device)
        )

    def reset_parameters(self):
        device = self.bias_mu.device
        self._update_parameters(device)

    @staticmethod
    def _sample_norm(mu, rho):
        device = mu.device
        eps = torch.randn_like(mu).to(device)
        sigma = torch.log(1 + torch.exp(rho)).to(device)
        p = mu + sigma * eps
        return p

    def sample(self, force=False):

        if self.training or force:
            return (
                self._sample_norm(self.weight_hh_mu, self.weight_hh_rho),
                self._sample_norm(self.weight_ih_mu, self.weight_ih_rho),
                self._sample_norm(self.bias_mu, self.bias_rho),
            )
        else:
            return (self.weight_hh_mu, self.weight_ih_mu, self.bias_mu)

    def loss(self, p, mu, rho):
        # compute the prior loss
        if self.stable_loss:
            lp1 = self.mix1.log_prob(p)
            lp2 = self.mix2.log_prob(p)
            mlp = torch.max(lp1, lp2)
            stable_lp = self.prior_pi * torch.exp(lp1 - mlp) + (
                1 - self.prior_pi
            ) * torch.exp(lp2 - mlp)
            prior = torch.log(stable_lp) + mlp
        else:
            p1 = torch.exp(self.mix1.log_prob(p))
            p2 = torch.exp(self.mix2.log_prob(p))
            prior = torch.log(self.prior_pi * p1 + (1 - self.prior_pi) * p2)

        # compute variational loss
        device = mu.device
        sigma = torch.log(1 + torch.exp(rho)).to(device)
        vari = (
            -np.log(np.sqrt(np.pi))
            - torch.log(sigma)
            - (((p - mu) ** 2) / (2 * sigma ** 2))
        )

        return vari.sum() - prior.sum()

    def forward(self, x, _params=None):

        B, L, K = x.size()

        assert K == self.loci

        h_t, c_t = (
            torch.zeros(self.hidden).to(x.device),
            torch.zeros(self.hidden).to(x.device),
        )

        # sample weights
        if _params is None:
            weight_hh, weight_ih, bias = self.sample()
        else:
            weight_hh, weight_ih, bias = _params

        # calculate loss
        loss = (
            self.loss(weight_hh, self.weight_hh_mu, self.weight_hh_rho)
            + self.loss(weight_ih, self.weight_ih_mu, self.weight_ih_rho)
            + self.loss(bias, self.bias_mu, self.bias_rho)
        )

        # compute hidden states
        H = self.hidden
        hidden_seq = []

        for t in range(L):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ weight_ih + h_t @ weight_hh + bias

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :H]),
                torch.sigmoid(gates[:, H : H * 2]),  # forget
                torch.tanh(gates[:, H * 2 : H * 3]),
                torch.sigmoid(gates[:, H * 3 :]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, loss, (h_t, c_t)


@attr.s(cmp=False)
class bPath(Path):

    num_batch: int = attr.ib(default=1)
    mu_init: float = attr.ib(default=1)
    rho_init: Tuple[float, float] = attr.ib(default=(-3, -1))
    prior_pi: float = attr.ib(default=0.5)
    prior_sigma_1: float = attr.ib(default=1)
    prior_sigma_2: float = attr.ib(default=0.001)
    stable_loss: bool = attr.ib(default=False)
    dynamic_weight: bool = attr.ib(default=False)
    error_weight: bool = attr.ib(default=False)
    error_disable: Tuple[int] = attr.ib(default=tuple())
    phenotype_weight: str = attr.ib(default="none")
    phenotype_disable: Tuple[int] = attr.ib(default=tuple())

    def __attrs_post_init__(self):
        super(bPath, self).__attrs_post_init__()
        self._rnn = bLSTM(
            self.loci,
            self.hidden,
            self.bias,
            self.mu_init,
            self.rho_init,
            self.prior_pi,
            self.prior_sigma_1,
            self.prior_sigma_2,
            self.stable_loss,
        )
        self._batch_count = 0

    def forward(self, X, lengths=None, includeHistory=False, _params=None):

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

        # randomize order if training
        if self.training:
            for i in range(X.shape[0]):
                ln = lengths[i].item()
                ind = np.arange(ln)
                order = np.random.choice(ind, ln, replace=False)
                X[i, :ln] = X[i, order]

        # build the packed data for the model
        # pck = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        # pck = pck.to(X.device)

        # compute path
        out, loss, hstate = self._rnn(X, _params=_params)

        # turn off garbage
        lengths_size = lengths.size()

        flat_lengths = lengths.view(-1, 1)

        max_length = out.shape[1]
        unit_range = torch.arange(max_length)
        flat_range = unit_range.repeat(flat_lengths.size()[0], 1)
        flat_indicator = flat_range < flat_lengths

        indicator = flat_indicator.view(lengths_size + (-1, 1))
        out = out * indicator.to(out.device)

        # for i in range(out.shape[0]):
        #     out[i, lengths[i] :, :] = 0

        # reconstruct original shape
        # output, _lengths = pad_packed_sequence(out, batch_first=True)
        # output = output.to(X.device)

        # check things make sense
        # assert (lengths == _lengths.to(lengths.device)).all()

        # find the individual deltas for each path step and then
        # aggregate
        deltas = self.linear(out)
        delta = deltas.sum(axis=1)

        if includeHistory:
            return delta, loss, deltas, hstate, out

        return delta  # , deltas, hstate

    def loss(self, sequence, y, error=None, *args, **kwargs):

        sequence = sequence.float()
        y = y.float()
        bs = y.shape[0]

        yhat, loss, _, _, _ = self(sequence, includeHistory=True)
        y = y.reshape_as(yhat)

        if self.dynamic_weight:
            # slightly off, but prevents overflow
            weight = np.power(2, float(-self._batch_count - 1))

            # weight = np.power(2, self.num_batch - self._batch_count - 1) / (
            #     np.power(2, self.num_batch) - 1
            # )
        else:
            weight = 1 / self.num_batch

        if self.training:
            self._batch_count += 1
            self._batch_count = self._batch_count % self.num_batch

        ########
        # phenotype
        ########

        phenWeight = torch.ones_like(yhat)
        N = phenWeight.sum()
        if self.phenotype_weight == "abs":
            phenWeight = y.float().abs()
        elif self.phenotype_weight == "square":
            phenWeight = y.float() ** 2

        for disable in self.phenotype_disable:
            phenWeight[:, disable] = phenWeight[:, disable].mean()

        # reweight to original scale
        z = phenWeight.sum() / N
        phenWeight = phenWeight / z

        mse = (yhat - y.float()) ** 2
        mse = phenWeight * mse

        for disable in self.error_disable:
            error[:, disable] = error[:, disable].mean()

        if self.error_weight:  # and not yerr is None:
            like = torch.sum(mse / 2 / (error ** 2))
            return (like + weight * loss) / bs
        else:
            # phenWeight * F.mse_loss(yhat, y.float(), reduction="sum")
            return (torch.sum(mse) + weight * loss) / bs

    def reset(self):
        super(bPath, self).reset()
        self._batch_count = 0
