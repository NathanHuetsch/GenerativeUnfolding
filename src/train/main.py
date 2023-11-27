import os
import argparse
import torch
import time
from datetime import timedelta

from .documenter import Documenter
from .train import Model, GenerativeUnfolding, Omnifold
from .plots import Plots, OmnifoldPlots
from ..processes.base import Process
from ..processes.zjets.process import ZJetsGenerative, ZJetsOmnifold


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
    print("------- Running training -------")
    model.train()
    print("------- Running evaluation -------")
    eval_model(doc, params, model, process)
    print("------- The end -------")


def run_plots(args: argparse.Namespace):
    doc, params = Documenter.from_saved_run(args.run_name)
    model, process = init_run(doc, params, args.verbose)
    model.load(args.model)
    print("------- Running evaluation -------")
    eval_model(doc, params, model, process)
    print("------- The end -------")


def init_run(doc: Documenter, params: dict, verbose: bool) -> tuple[Model, Process]:
    use_cuda = torch.cuda.is_available()
    print("Using device " + ("GPU" if use_cuda else "CPU"))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("------- Loading data -------")
    dataset = params.get("process", "ZJetsGenerative")
    print(f"    Process: {dataset}")
    process = eval(dataset)(params["process_params"], device)
    data = (
        process.get_data("train"),
        process.get_data("val"),
        process.get_data("test"),
    )
    dims_in = data[0].x_hard.shape[1]
    dims_c = data[0].x_reco.shape[1]
    params["dims_in"] = dims_in
    params["dims_c"] = dims_c
    print(f"    Train events: {len(data[0].x_hard)}")
    print(f"    Val events: {len(data[1].x_hard)}")
    print(f"    Test events: {len(data[2].x_hard)}")
    print(f"    Hard dimension: {dims_in}")
    print(f"    Reco dimension: {dims_c}")

    model_path = doc.get_file("model", False)
    os.makedirs(model_path, exist_ok=True)
    print("------- Building model -------")
    method = params.get("method", "GenerativeUnfolding")
    print(f"    Using method: {method}")
    model = eval(method)(params, verbose, device, model_path, process)
    model.method = method
    model.init_data_loaders()
    model.init_optimizer()
    return model, process


def evaluation_generative(
    doc: Documenter,
    params: dict,
    model: Model,
    process: Process,
    data: str = "test",
    model_name: str = "final",
    name: str = ""):

    print(f"Checkpoint: {model_name},  Data: {data}")
    if data == "test":
        loader = model.test_loader
    elif data == "train":
        loader = model.train_loader
    elif data == "analysis":
        analysis_data = process.get_data("analysis")
        val_loader_kwargs = {"shuffle": False, "batch_size": 10*params["batch_size"], "drop_last": False}
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(model.hard_pp(analysis_data.x_hard).float(),
                                           model.reco_pp(analysis_data.x_reco).float()),
            **val_loader_kwargs,
        )

    model.load(model_name)

    print(f"    Generating single samples")
    t0 = time.time()
    x_gen_single = model.predict(loader=loader)
    t1 = time.time()
    time_diff = timedelta(seconds=round(t1 - t0))
    print(f"    Generation completed after {time_diff}")

    print(f"    Generating distributions")
    x_gen_dist = model.predict_distribution(loader=loader)

    if params.get("compute_test_loss", False):
        print(f"    Computing test loss")
        test_ll = model.dataset_loss(loader=loader)["loss"]
        print(f"    Result: {test_ll:.4f}")

    print(f"    Computing observables")
    data = process.get_data(data)
    observables = process.hard_observables()
    data_hard_pp = model.input_data_preprocessed[0]
    data_reco_pp = model.cond_data_preprocessed[0]
    plots = Plots(
        observables=observables,
        losses=model.losses,
        x_hard=data.x_hard,
        x_reco=data.x_reco,
        x_gen_single=x_gen_single,
        x_gen_dist=x_gen_dist,
        x_hard_pp=data_hard_pp,
        x_reco_pp=data_reco_pp,
        bayesian=model.model.bayesian,
        show_metrics=True
    )
    print(f"    Plotting loss")
    plots.plot_losses(doc.add_file("losses"+name+".pdf"))
    print(f"    Plotting observables")
    if params.get("save_hist_data", False):
        pickle_file = doc.add_file("observables"+name+".pkl")
    else:
        pickle_file = None
    plots.plot_observables(doc.add_file("observables"+name+".pdf"), pickle_file)

    plots.plot_preprocessed(doc.add_file("preprocessed" + name + ".pdf"))
    #print(f"    Plotting calibration")
    #plots.plot_calibration(doc.add_file("calibration"+name+".pdf"))
    #print(f"   Plotting pulls")
    #plots.plot_pulls(doc.add_file("pulls"+name+".pdf"))
    #print(f"   Plotting single events")
    #plots.plot_single_events(doc.add_file("single_events"+name+".pdf"))
    print(f"    Plotting migration")
    plots.plot_migration(doc.add_file("migration" + name + ".pdf"))
    plots.plot_migration2(doc.add_file("migration2" + name + ".pdf"))
    if params.get("plot_gt_migration", True):
        plots.plot_migration(doc.add_file("gt_migration" + name + ".pdf"), gt_hard=True)
        plots.plot_migration2(doc.add_file("gt_migration2" + name + ".pdf"), gt_hard=True)


def evaluation_omnifold(
    doc: Documenter,
    params: dict,
    model: Model,
    process: Process,
    data: str = "test",
    model_name: str = "final",
    name: str = ""):

    print(f"Checkpoint: {model_name},  Data: {data}")
    if data == "test":
        loader = model.test_loader
    elif data == "train":
        loader = model.train_loader

    model.load(model_name)

    print(f"    Predicting weights")
    t0 = time.time()
    predictions = model.predict_probs(loader=loader)
    t1 = time.time()
    time_diff = timedelta(seconds=round(t1 - t0))
    print(f"    Predictions completed after {time_diff}")

    if params.get("compute_test_loss", False):
        print(f"    Computing test loss")
        test_ll = model.dataset_loss(loader=loader)["loss"]
        print(f"    Result: {test_ll:.4f}")

    print(f"    Computing observables")
    data = process.get_data(data)
    observables = process.hard_observables()
    plots = OmnifoldPlots(
        observables=observables,
        losses=model.losses,
        x_hard=data.x_hard,
        x_reco=data.x_reco,
        labels=data.label,
        predictions=predictions,
        bayesian=model.model.bayesian,
        show_metrics=True
    )
    print(f"    Plotting loss")
    plots.plot_losses(doc.add_file("losses"+name+".pdf"))
    print(f"    Plotting classes")
    plots.plot_classes(doc.add_file("classification"+name+".pdf"))
    print(f"    Plotting reco")
    plots.plot_reco(doc.add_file("reco"+name+".pdf"))
    print(f"    Plotting hard")
    plots.plot_hard(doc.add_file("hard"+name+".pdf"))
    print(f"    Plotting Observables")
    if params.get("save_hist_data", False):
        pickle_file = doc.add_file("observables"+name+".pkl")
    else:
        pickle_file = None
    plots.plot_observables(doc.add_file("observables"+name+".pdf"), pickle_file)


def eval_model(doc: Documenter, params: dict, model: Model, process: Process):

    if params.get("method", "GenerativeUnfolding") == "Omnifold":
        evaluation = evaluation_omnifold
        evaluate_analysis = False
    else:
        evaluation = evaluation_generative
        evaluate_analysis = params.get("evaluate_analysis")

    evaluation(doc, params, model, process)
    if params.get("evaluate_train", False):
        evaluation(doc, params, model, process, data="train", name="_train")
    if params.get("evaluate_best", False):
        evaluation(doc, params, model, process, model_name="best", name="_best")
        if params.get("evaluate_train", False):
            evaluation(doc, params, model, process, model_name="best", data="train", name="_best_train")
    if evaluate_analysis:
        evaluation(doc, params, model, process, data="analysis", name="_analysis")
