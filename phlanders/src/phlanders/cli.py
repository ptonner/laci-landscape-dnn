from phlanders.experiment import Experiment
from phlanders.metric import scores, predictions

import uuid
import glob
import os
import torch
import argparse
from importlib import import_module

import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _args_processor(func):
    """Args decorator for universal actions.
    """

    def wrapper(args):
        import sys; sys.path.append(".")
        import_module(args._import)
        func(args)

    return wrapper


def _device_from_args(args):

    device = "cpu" if args.disable_cuda or not torch.cuda.is_available() else "cuda"

    return device


@_args_processor
def _train(args):

    for path in args.experiments:

        experiment = Experiment.load(path)
        experiment.dataset.validate()

        base = os.path.dirname(path)

        if args.output is None:
            id = str(uuid.uuid4())
            opath = os.path.join(base, "models", id)
        else:
            opath = os.path.join(base, args.output)

        device = "cpu" if args.disable_cuda or not torch.cuda.is_available() else "cuda"
        if args.disable_cudann:
            torch.backends.cudnn.enabled = False

        logger.info(
            "starting training {} model on {}, storing results in {}".format(
                type(experiment.model).__name__, str(device), opath
            )
        )

        experiment.train(device, output=opath)

        experiment.finish(output=opath)


@_args_processor
def _metrics(args):

    for configuration in args.experiments:
        experiment = Experiment.load(configuration)
        experiment.dataset.validate()
        experiment.dataset.initialize()

        results = args.results

        device = _device_from_args(args)

        kw = {}
        for a in args.true:
            kw[a] = True

        if results is None:
            dname = os.path.dirname(configuration)
            if args.output is None:
                results = glob.glob(os.path.join(dname, "models", "*"))
            else:
                results = [os.path.join(dname, args.output)]
        else:
            results = [results]

        for res in results:
            s = scores(experiment, res, device, **kw)
            s.to_csv(os.path.join(res, "metrics.csv"), index=False)


@_args_processor
def _predictions(args):

    for configuration in args.experiments:
        experiment = Experiment.load(configuration)
        experiment.dataset.validate()
        experiment.dataset.initialize()

        results = args.results

        device = _device_from_args(args)

        kw = {}
        for a in args.true:
            kw[a] = True

        if results is None:
            dname = os.path.dirname(configuration)
            if args.output is None:
                results = glob.glob(os.path.join(dname, "models", "*"))
            else:
                results = [os.path.join(dname, args.output)]
        else:
            results = [results]

        for res in results:
            p = predictions(experiment, res, device, **kw)
            p.to_csv(os.path.join(res, "predictions.csv"), index=False)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        "phlanders", description="Fitness landscape deep regression."
    )

    parser.add_argument(
        "--import",
        help="Additional module to import, useful for custom models",
        dest="_import",
    )

    subparsers = parser.add_subparsers()

    train = subparsers.add_parser("train", help="train a model")
    train.set_defaults(func=_train)
    train.add_argument("experiments", help="The experiments to train.", nargs="+")
    train.add_argument("--disable-cuda", action="store_true")
    train.add_argument("--disable-cudann", action="store_true")
    train.add_argument("--output", help="output directory")

    metrics = subparsers.add_parser("metrics", help="calculate a results of training")
    metrics.set_defaults(func=_metrics)
    metrics.add_argument(
        "--true", action="append", help="arguments to true during metric calculation"
    )
    metrics.add_argument(
        "experiments", help="The experiment to compute metrics for.", nargs="+"
    )
    metrics.add_argument(
        "--results", help="The results (e.g. output) of an experiment."
    )
    metrics.add_argument("--disable-cuda", action="store_true")
    metrics.add_argument("--output", help="output directory")

    predictions = subparsers.add_parser("predictions", help="calculate predictions")
    predictions.set_defaults(func=_predictions)
    predictions.add_argument(
        "--true",
        action="append",
        help="arguments to true during prediction calculation",
    )
    predictions.add_argument(
        "experiments", help="The experiments to compute predictions for.", nargs="+"
    )
    predictions.add_argument(
        "--results", help="The results (e.g. output) of an experiment."
    )
    # predictions.add_argument(
    #     "results", help="The results (e.g. output) of an experiment.")
    predictions.add_argument("--disable-cuda", action="store_true")
    predictions.add_argument("--output", help="output directory")

    args = parser.parse_args()
    args.func(args)
