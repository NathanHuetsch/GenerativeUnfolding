import os
import argparse
import torch
import time
from datetime import timedelta

from .documenter import Documenter
from .train import Model, ImportanceSampler, TransferFunction, Efficiency
from .plots import Plots, ClassifierPlots
from ..processes.base import Process
from ..processes.thj_lo.process import LoThjProcess

import pickle


def init_train_args(subparsers):
    """
    Initializes the argparsers for the src train and src plot commands

    Args:
        subparsers: Object returned by ArgumentParser.add_subparsers
    """
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("paramcard")
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.set_defaults(func=run_training)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("run_name")
    plot_parser.add_argument("--model", type=str, default="final")
    plot_parser.add_argument("--verbose", action="store_true")
    plot_parser.set_defaults(func=run_plots)


def run_training(args: argparse.Namespace):
    doc, params = Documenter.from_param_file(args.paramcard)
    model, process = init_run(doc, params, args.verbose)
    print("Running training")
    model.train()
    eval_model(doc, params, model, process)
    print("The end")


def run_plots(args: argparse.Namespace):
    doc, params = Documenter.from_saved_run(args.run_name)
    model, process = init_run(doc, params, args.verbose)
    model.load(args.model)
    eval_model(doc, params, model, process)
    print("The end")


def init_run(doc: Documenter, params: dict, verbose: bool) -> tuple[Model, Process]:
    use_cuda = torch.cuda.is_available()
    print("Using device " + ("GPU" if use_cuda else "CPU"))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print("Loading data")
    process = LoThjProcess(params["process_params"], device)
    data = (
        process.get_data("train"),
        process.get_data("val"),
        process.get_data("test"),
    )
    print(f"  Train events: {len(data[0].x_hard)}")
    print(f"  Val events: {len(data[1].x_hard)}")
    print(f"  Test events: {len(data[2].x_hard)}")

    print("Building model")
    model_path = doc.get_file("model", False)
    os.makedirs(model_path, exist_ok=True)
    if params["type"] == "transfer_function":
        model = TransferFunction(params, verbose, device, model_path, process)
    elif params["type"] == "importance_sampler":
        model = ImportanceSampler(params, verbose, device, model_path, process)
    elif params["type"] == "efficiency":
        model = Efficiency(params, verbose, device, model_path, process)
    else:
        raise ValueError(f"Unknown model type '{params['type']}'")
    model.init_data_loaders(data)
    model.init_optimizer()
    return model, process


def evaluation(
    doc: Documenter,
    params: dict,
    model: Model,
    process: Process,
    data: str = "test",
    model_name: str = "final",
    name: str = ""):
    print(f"Running evaluation with {model_name} model on {data} data")
    if data == "test":
        loader = model.test_loader
    elif data == "train":
        loader = model.train_loader

    model.load(model_name)

    if model.is_classifier:
        eval_classifier(doc, params, model, process)
        return

    print("  Generating single samples")
    t0 = time.time()
    x_gen_single = model.predict(loader=loader)
    t1 = time.time()
    time_diff = timedelta(seconds=round(t1 - t0))
    print(f"  Generation completed after {time_diff}")

    print("  Generating distributions")
    x_gen_dist = model.predict_distribution(loader=loader)

    if params.get("compute_test_loss", False):
        print("  Computing test loss")
        test_ll = model.dataset_loss(loader=loader)["loss"]
        print(f"    Result: {test_ll:.4f}")

    # TODO: Fix transfusion log likelihood.
    #    print("  Computing test log likelihood")
    #    t0 = time.time()
    #    test_ll, test_ll_err = model.test_log_likelihood()
    #    test_ll, test_ll_err = model.predict_loglikelihood(loader=loader).mean()
    #    t1 = time.time()
    #    time_diff = timedelta(seconds=round(t1 - t0))
    #    if test_ll_err is None:

    print("  Computing observables")
    test_data = process.get_data(data)
    if params["type"] == "transfer_function":
        x_test = test_data.x_reco
        observables = process.reco_observables()
    elif params["type"] == "importance_sampler":
        x_test = test_data.x_hard
        observables = process.hard_observables()
    plots = Plots(
        observables,
        model.losses,
        x_test,
        x_gen_single,
        x_gen_dist,
        test_data.event_type,
        model.model.bayesian
    )
    print("  Plotting loss")
    plots.plot_losses(doc.add_file("losses"+name+".pdf"))
    print("  Plotting observables")
    if params.get("save_hist_data", False):
        pickle_file = doc.add_file("observables"+name+".pkl")
    else:
        pickle_file = None
    plots.plot_observables(doc.add_file("observables"+name+".pdf"), pickle_file)
    print("  Plotting calibration")
    plots.plot_calibration(doc.add_file("calibration"+name+".pdf"))
    print("  Plotting pulls")
    plots.plot_pulls(doc.add_file("pulls"+name+".pdf"))
    print("  Plotting single events")
    plots.plot_single_events(doc.add_file("single_events"+name+".pdf"))


def save_likelihoods(
        doc: Documenter,
        model: Model,
        process: Process,
):
    """
    Evaluates the loglikelihoods and saves them
    Data is to be used for training of a distillation model
    """

    testset_loglikelihoods = model.predict_loglikelihood(loader=model.test_loader)
    valset_loglikelihoods = model.predict_loglikelihood(loader=model.val_loader)
    trainset_loglikelihoods = model.predict_loglikelihood(loader=model.train_loader)

    save_data = {
        "train_logp": trainset_loglikelihoods.cpu().numpy(),
        "val_logp": valset_loglikelihoods.cpu().numpy(),
        "test_logp": testset_loglikelihoods.cpu().numpy()
    }

    file = doc.add_file("likelihoods.pkl")
    with open(file, "wb") as f:
        pickle.dump(save_data, f)


def eval_model(doc: Documenter, params: dict, model: Model, process: Process):
    if model.is_classifier:
        eval_classifier(doc, params, model, process)
        return
    evaluation(doc, params, model, process)
    if params.get("evaluate_train", False):
        evaluation(doc, params, model, process, data="train", name="_train")
    if params.get("evaluate_best", False):
        evaluation(doc, params, model, process, model_name="best", name="_best")
        if params.get("evaluate_train", False):
            evaluation(doc, params, model, process, model_name="best", data="train", name="_best_train")

    if params.get("save_likelihoods", False):
        print("saving likelihoods", flush=True)
        save_likelihoods(doc, model, process)

def eval_classifier(doc: Documenter, params: dict, model: Model, process: Process):
    print("  Predicting classes")
    classes_test, classes_predict = model.predict_classes()
    test_ll = -torch.mean(classes_test * classes_predict.log()).item() * classes_test.shape[1]
    print(f"    Test log likelihood: {test_ll:.4f}")
    test_data = process.get_data("test")
    observables = process.hard_observables()
    plots = ClassifierPlots(
        observables,
        model.losses,
        test_data.x_hard,
        classes_test,
        classes_predict,
        test_data.event_type
    )
    print("  Plotting loss")
    plots.plot_losses(doc.add_file("losses.pdf"))
    print("  Plotting classes")
    plots.plot_classes(doc.add_file("classification.pdf"))
