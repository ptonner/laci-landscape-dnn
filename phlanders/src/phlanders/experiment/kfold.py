import attr
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold as _KFold
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from phlanders.experiment import Experiment

import torch
from torch.utils.data import Subset, DataLoader
from torch import optim
from typing import Optional

import logging

logger = logging.getLogger(__name__)


@attr.s
class KFold(Experiment):

    splits: int = attr.ib(default=5)
    cvSeed: Optional[int] = attr.ib(default=0)
    skip: Optional[int] = attr.ib(default=0)
    group: Optional[int] = attr.ib(default=1)

    # deprecated, do train/test split elsewhere
    testSize: float = attr.ib(default=0.2)
    ttSeed: Optional[int] = attr.ib(default=0)

    def __attrs_post_init__(self):
        super(KFold, self).__attrs_post_init__()

        self._splits = list(self.folds())
        self._cvs = [
            (Subset(self.dataset, tr), Subset(self.dataset, ts))
            for tr, ts in self._splits
        ]

    def indices(self):
        return np.arange(len(self.dataset) // self.group)

    def folds(self):
        ind = self.indices()
        cv = _KFold(self.splits, shuffle=True, random_state=self.cvSeed)

        # if needed, expand the folds back for groups
        if self.group > 1:
            for tr, ts in cv.split(ind):
                tr = np.repeat(tr * self.group, self.group) + np.tile(
                    np.arange(self.group), tr.shape[0]
                )
                ts = np.repeat(ts * self.group, self.group) + np.tile(
                    np.arange(self.group), ts.shape[0]
                )
                yield tr, ts
        else:
            for tr, ts in cv.split(ind):
                yield tr, ts

    @property
    def P(self):
        return self.splits - self.skip

    def pair_parameters(self, p, useBest, *args, **kwargs):
        self._check_pair(p)
        if useBest:
            return "split-{}-best.pt".format(p)
        return "split-{}.pt".format(p)

    def pair_data(self, p, validation=False, *args, **kwargs):

        self._check_pair(p)
        if validation:
            return self._cvs[p][1]
        return self._cvs[p][0]

    def _train(self, output, *args, **kwargs):
        lossLogger = defaultdict(list)
        lossLoggerTest = defaultdict(list)

        if self.skip > 0:
            splits = self._cvs[: -self.skip]
        else:
            splits = self._cvs

        for spl, (train, test) in enumerate(splits):
            logger.debug(
                "Train split {}, with dataset sizes {} and {}.".format(
                    spl, len(train), len(test)
                )
            )

            if self.sampler is None:
                train_loader = DataLoader(
                    train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=self.dataset.collate,
                )

                test_loader = DataLoader(
                    test, collate_fn=self.dataset.collate, batch_size=self.batch_size
                )
            else:
                train_loader = DataLoader(
                    train,
                    batch_size=self.batch_size,
                    collate_fn=self.dataset.collate,
                    sampler=self.sampler.build(data=train, batch_size=self.batch_size),
                )

                test_loader = DataLoader(
                    test,
                    collate_fn=self.dataset.collate,
                    batch_size=self.batch_size,
                    sampler=self.sampler.build(data=test, batch_size=self.batch_size),
                )

            test_best = np.inf

            self.model.reset()

            optimizer = optim.Adam(self.model.parameters(), **self.optimizer)
            scheduler = None
            if self.lr_step > 0:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, self.lr_step, self.lr_gamma
                )

            if self.verbose == 1:
                progressbar = tqdm(
                    total=self.epochs,
                    desc="Split:{}/{}".format(spl + 1, self.splits - self.skip),
                )

            for i in range(self.epochs):

                if self.verbose == 2:
                    progressbar = tqdm(
                        total=len(train_loader) * self.batch_size,
                        desc="Split: {}/{}, Epoch: {}".format(
                            spl + 1, self.splits - self.skip, i
                        ),
                    )

                total = 0.0
                self.model.train()
                for j, batch in enumerate(train_loader):

                    # batch = [v.to(self._device) for v in batch]
                    for k in batch.keys():
                        batch[k] = batch[k].to(self._device)

                    optimizer.zero_grad()
                    weights = None
                    if self.weighter is not None:
                        weights = self.weighter.weights(batch)
                        weights = torch.from_numpy(weights).to(self._device)
                    loss = self.model.loss(**batch, weights=weights)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                    total += loss

                    if self.verbose == 2:
                        progressbar.set_postfix(total=total / (j + 1))
                        progressbar.update(self.batch_size)

                if self.verbose == 2:
                    progressbar.close()

                ttotal = total / (j + 1)

                lossLogger[spl].append(total / (j + 1))

                # testing loss
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for j, batch in enumerate(test_loader):
                        # batch = [v.to(self._device) for v in batch]
                        for k in batch.keys():
                            batch[k] = batch[k].to(self._device)

                        weights = None
                        if self.weighter is not None:
                            weights = self.weighter.weights(batch)
                            weights = torch.from_numpy(weights).to(self._device)

                        loss = self.model.loss(**batch, weights=weights)
                        loss = loss.item()
                        total += loss

                lossLoggerTest[spl].append(total / (j + 1))

                if total < test_best:
                    test_best = total
                    self.save_weights(
                        self.model, os.path.join(output, "split-{}-best.pt".format(spl))
                    )

                if scheduler is not None:
                    scheduler.step()

                if self.verbose == 1:
                    progressbar.set_postfix(train=ttotal, test=(total) / (j + 1))
                    progressbar.update()

            self.save_weights(
                self.model, os.path.join(output, "split-{}.pt".format(spl))
            )

        self._plot_loss(output, train=lossLogger, validation=lossLoggerTest)
        self._save_loss(output, train=lossLogger, validation=lossLoggerTest)

    def _plot_loss(self, output, **kwargs):
        keys = kwargs.keys()
        fig, axes = plt.subplots(figsize=(4 * len(keys), 4), ncols=len(keys))

        mn, mx = np.inf, -np.inf
        for i, (k, log) in enumerate(kwargs.items()):
            ax = axes[i]
            ax.set_title(k)

            for spl in range(self.splits):
                if spl in log:
                    ax.plot(log[spl], label=str(spl))
                    mn = min(np.min(log[spl]), mn)
                    mx = max(np.max(log[spl]), mx)

        for i in range(len(keys)):
            axes[i].set_ylim(mn, mx)
            axes[i].semilogy()
            axes[i].set_xlabel("epoch")

            if i == 0:
                axes[i].legend()
                axes[i].set_ylabel("loss")
            else:
                axes[i].set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(output, "loss.pdf"), bbox_inches="tight")

    def _save_loss(self, output, **kwargs):
        for k, loss in kwargs.items():
            pd.DataFrame(
                {
                    "split-{}".format(spl): loss[spl]
                    for spl in range(self.splits - self.skip)
                }
            ).to_csv(os.path.join(output, "loss-{}.csv".format(k)))
