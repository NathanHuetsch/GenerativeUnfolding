import os
import argparse
import torch
import time
from datetime import timedelta
import numpy as np

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
    loader = params["process_params"].get("loader", "theirs")
    print(f"    Process: {dataset}, loader {loader}")
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

    if params.get("train_classifier", False):
        print(f"    Training classifier for evaluation")
        rng = np.random.RandomState(0) # anything generated with this rng is reproducible
        if model.model.bayesian:
            permutation = torch.as_tensor(rng.permutation(len(data.x_hard) + len(x_gen_single[0])), device=model.device)
        else:
            permutation = torch.as_tensor(rng.permutation(len(data.x_hard) + len(x_gen_single)), device=model.device)
        #model.train_classifier(doc = doc, raw_data = data, raw_gen_single = x_gen_single, raw_gen_dist = x_gen_dist, permutation = permutation)
        eval_classifier_preds, test_gen_classifier, test_label_classifier = model.eval_classifier(doc = doc, raw_data = data, raw_gen_single = x_gen_single, raw_gen_dist = x_gen_dist, permutation = permutation, data = "test", model_name = "final", name = "")
        print(test_gen_classifier.shape)                             

    plots = Plots(
        observables=observables,
        losses=model.losses,
        cl_losses = model.cl_losses if params.get("train_classifier", False) else None,
        x_hard=data.x_hard,
        x_reco=data.x_reco,
        x_gen_single=x_gen_single,
        x_gen_dist=x_gen_dist,
        x_hard_pp=data_hard_pp,
        x_reco_pp=data_reco_pp,
        bayesian=model.model.bayesian,
        show_metrics=True,
        plot_metrics=params.get("plot_metrics", False),
        n_unfoldings = params.get("n_unfoldings", 1),
        eval_classifier_preds = eval_classifier_preds if params.get("train_classifier", False) else None,
        cl_gen_test = test_gen_classifier if params.get("train_classifier", False) else None,
        cl_label_test = test_label_classifier if params.get("train_classifier", False) else None,
        debug = False,
    )
    if name == "":
        print(f"    Plotting loss")
        plots.plot_losses(doc.add_file("losses"+name+".pdf"))
        if params.get("train_classifier", False):
            plots.plot_losses(doc.add_file("classifier_losses"+name+".pdf"), cl_loss=True)
    print(f"    Plotting observables")
    if params.get("save_hist_data", True):
        pickle_file = doc.add_file("observables"+name+".pkl")
    else:
        pickle_file = None
    plots.plot_observables(doc.add_file("observables"+name+".pdf"), pickle_file)
    #plots.plot_observables_full(doc.add_file("observables_full" + name + ".pdf"), None)
    plots.save_metrics(doc.add_file("metrics" + name + ".pkl"))
    if params.get("plot_metrics", True):
        plots.hist_metrics_unfoldings(doc.add_file("unfolding_metrics"+name+".pdf"))
        plots.plot_multiple_unfoldings(doc.add_file("unfolding_samples"+name+".pdf"))

        if model.model.bayesian:
            #plots.hist_metrics_bayesian(doc.add_file("bayesian_metrics" + name + ".pdf"))
            plots.plot_multiple_bayesians(doc.add_file("bayesian_samples" + name + ".pdf"))

    if params.get("plot_preprocessed", False) and name == "":
        print(f"    Plotting preprocessed data")
        plots.plot_preprocessed(doc.add_file("preprocessed" + name + ".pdf"))
    print(f"    Plotting calibration")
    plots.plot_calibration(doc.add_file("calibration"+name+".pdf"), pickle_file=doc.add_file("calibration"+name+".pkl"))
    #print(f"    Plotting pulls")
    #plots.plot_pulls(doc.add_file("pulls"+name+".pdf"))
    #print(f"    Plotting single events")
    #plots.plot_single_events(doc.add_file("single_events"+name+".pdf"), pickle_file=doc.add_file("single_events"+name+".pkl"))
    print(f"    Plotting migration")
    #plots.plot_migration(doc.add_file("migration" + name + ".pdf"))

    plots.plot_migration2(doc.add_file("migration2" + name + ".pdf"), pickle_file=doc.add_file("migration2" + name + ".pkl"))
    plots.plot_migration2(doc.add_file("gt_migration2" + name + ".pdf"), pickle_file=doc.add_file("gt_migration2" + name + ".pkl"), gt_hard=True)
    #if params.get("plot_gt_migration", True) and name == "":
        #plots.plot_migration(doc.add_file("gt_migration" + name + ".pdf"), gt_hard=True)

    if params.get("save_samples", False):
        print(f"    Saving samples")
        file = doc.add_file("samples" + name + ".pkl")
        np.save(file, x_gen_single)



def evaluate_comparison(
    doc: Documenter,
    params: dict,
    model: Model,
    process: Process,
    model_name: str = "final",
    name: str = ""):

    print(f"Checkpoint: {model_name},  Data: Comparison Set")
    data_reco = torch.tensor(np.load('data/SB_Pythia_reco.npy')).to(model.device)
    data_hard = torch.tensor(np.load('data/SB_Pythia_hard.npy')).to(model.device)
    data_SB = torch.tensor(np.load('data/SB_Pythia_unfold.npy')).to(model.device)

    loader_kwargs = {"shuffle": False, "batch_size": 10*params["batch_size"], "drop_last": False}
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(model.hard_pp(data_hard).float(),
                                       model.reco_pp(data_reco).float()),
        **loader_kwargs,
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
    observables = process.hard_observables()
    plots = Plots(
        observables=observables,
        losses=model.losses,
        x_hard=data_hard,
        x_reco=data_reco,
        x_gen_single=x_gen_single,
        x_gen_dist=x_gen_dist,
        x_compare=data_SB,
        bayesian=model.model.bayesian,
        show_metrics=True,
        n_unfoldings = params.get("n_unfoldings", 1)
    )
    print(f"    Plotting observables")
    if params.get("save_hist_data", True):
        pickle_file = doc.add_file("observables_comparison"+name+".pkl")
    else:
        pickle_file = None
    plots.plot_observables(doc.add_file("observables_comparison"+name+".pdf"), pickle_file)
    print(f"    Plotting migration")
    plots.plot_migration2(doc.add_file("SB_migration2" + name + ".pdf"), SB_hard=True, pickle_file=doc.add_file("SB_migration2" + name + ".pkl"))

def evaluate_analysis_withSB(
    doc: Documenter,
    params: dict,
    model: Model,
    process: Process,
    model_name: str = "final",
    name: str = ""):

    print(f"Checkpoint: {model_name},  Data: Analysis Set, Including SB")
    data_reco = torch.tensor(np.load('data/SB_Herwig_reco.npy')).to(model.device)
    data_hard = torch.tensor(np.load('data/SB_Herwig_hard.npy')).to(model.device)
    data_SB = torch.tensor(np.load('data/SB_Herwig_unfold.npy')).to(model.device)

    loader_kwargs = {"shuffle": False, "batch_size": 10*params["batch_size"], "drop_last": False}
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(model.hard_pp(data_hard).float(),
                                       model.reco_pp(data_reco).float()),
        **loader_kwargs,
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
    observables = process.hard_observables()
    plots = Plots(
        observables=observables,
        losses=model.losses,
        x_hard=data_hard,
        x_reco=data_reco,
        x_gen_single=x_gen_single,
        x_gen_dist=x_gen_dist,
        x_compare=data_SB,
        bayesian=model.model.bayesian,
        show_metrics=True,
        n_unfoldings = params.get("n_unfoldings", 1)
    )
    print(f"    Plotting observables")
    if params.get("save_hist_data", True):
        pickle_file = doc.add_file("observables_analysis_wSB"+name+".pkl")
    else:
        pickle_file = None
    plots.plot_observables(doc.add_file("observables_analysis_wSB"+name+".pdf"), pickle_file)

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
        train_data = process.get_data("train")
        label_data = train_data.label
        reco_data = model.reco_pp(train_data.x_reco)
        loader_kwargs = {"shuffle": False, "batch_size": 10 * params["batch_size"], "drop_last": False}
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(label_data.float(),
                                                                            reco_data.float()),
                                                                            **loader_kwargs,
                                                                            )

    model.load(model_name)

    print(f"    Predicting weights")
    t0 = time.time()
    predictions = model.predict_probs(loader=loader)
    t1 = time.time()
    time_diff = timedelta(seconds=round(t1 - t0))
    print(f"    Predictions completed after {time_diff}")

    if params.get("compute_test_loss", False):
        print(f"    Computing {data} loss")
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
        show_metrics=True,
        pythia_only = params["process_params"].get("pythia_only", False),
        debug = False,
        invert_reweighting=params["process_params"].get("invert_reweighting", False),
    )
    if name == "":
        print(f"    Plotting loss")
        plots.plot_losses(doc.add_file("losses.pdf"))
    print(f"    Plotting classes")
    plots.plot_classes(doc.add_file("classification"+name+".pdf"))
    print(f"    Plotting weights")
    plots.plot_weights(doc.add_file("weights"+name+".pdf"))
    print(f"    Plotting reco")
    if params.get("save_hist_data", True):
        plots.plot_reco(doc.add_file("reco"+name+".pdf"), doc.add_file("reco"+name+".pkl"))
    else:
        plots.plot_reco(doc.add_file("reco"+name+".pdf"))
    print(f"    Plotting hard")
    if params.get("save_hist_data", True):
        plots.plot_hard(doc.add_file("hard"+name+".pdf"), doc.add_file("hard"+name+".pkl"))
    else:
        plots.plot_hard(doc.add_file("hard"+name+".pdf"))
    print(f"    Plotting Observables")
    if params.get("save_hist_data", True):
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
    
    if params.get("evaluate_analysis_wSB", False):
        evaluate_analysis_withSB(doc, params, model, process)
    if params.get("evaluate_comparison", False):
        evaluate_comparison(doc, params, model, process)
