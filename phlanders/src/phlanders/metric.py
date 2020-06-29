from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
import numpy as np
import torch
import os


def scores(exp, results, device, **kwargs):

    s = []
    for i in range(exp.P):

        pth = exp.pair_parameters(i, **kwargs)
        pth = os.path.join(results, pth)

        m = exp.load_weights(
            pth,
            map_location=device)
        m.eval()
        tst = exp.pair_data(i, validation=True)

        seq, y = tst[:len(tst)][:2]
        # seq, y = trn[:len(tst)][:2]
        y = y.numpy()

        if y.ndim == 1:
            y = y[:, None]

        with torch.no_grad():
            yhat = m(seq)
            if isinstance(yhat, tuple):
                yhat = yhat[0]

            yhat = yhat.numpy()

        rw = []
        for j in range(y.shape[1]):
            rw.append(r2_score(y[:, j], yhat[:, j]))

        for j in range(y.shape[1]):
            rw.append(mean_squared_error(y[:, j], yhat[:, j]))

        s.append(rw)

    s = np.array(s)
    k = s.shape[1]//2
    cols = ["r2_{}".format(i) for i in range(k)] + \
        ["mse_{}".format(i) for i in range(k)]
    df = pd.DataFrame(s, columns=cols)
    return df


def predictions(exp, results, device, useBest=True, **kwargs):

    preds = []
    for i in range(exp.P):

        pth = exp.pair_parameters(i, useBest)
        pth = os.path.join(results, pth)

        m = exp.load_weights(
            pth,
            map_location=device)
        m.eval()
        tst = exp.pair_data(i, validation=True)

        seq, y = tst[:len(tst)][:2]
        y = y.numpy()

        if y.ndim == 1:
            y = y[:, None]

        with torch.no_grad():
            yhat = m(seq)
            if isinstance(yhat, tuple):
                yhat = yhat[0]

            yhat = yhat.numpy()

        pair = np.repeat(i, y.shape[0])[:, None]
        rw = np.concatenate((pair, y, yhat), axis=1)

        samplerWeight = None
        if exp.sampler is not None:
            samplerWeight = exp.sampler.weights(tst)
            if samplerWeight.ndim == 1:
                samplerWeight = samplerWeight[:, None]
            rw = np.concatenate((rw, samplerWeight), axis=1)

        weighterWeight = None
        if exp.weighter is not None:
            weighterWeight = exp.weighter.weights(tst)
            if weighterWeight.ndim == 1:
                weighterWeight = weighterWeight[:, None]
            rw = np.concatenate((rw, weighterWeight), axis=1)

        preds.append(rw)

    ret = np.concatenate(preds, axis=0)
    k = y.shape[1]
    cols = ['pair'] + ["y_{}".format(i) for i in range(k)] +\
        ["yhat_{}".format(i) for i in range(k)]

    if samplerWeight is not None:
        cols = cols + ["samplerWeight_{}".format(i) for i in
                       range(samplerWeight.shape[1])]

    if weighterWeight is not None:
        cols = cols + ["weighterWeight_{}".format(i) for i in
                       range(weighterWeight.shape[1])]

    df = pd.DataFrame(ret, columns=cols)
    return df
