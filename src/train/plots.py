import warnings
from dataclasses import dataclass
from typing import Optional
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import torch
from .utils import GetEMD, get_triangle_distance, get_normalisation_weight

from ..processes.observables import Observable
import yaml
from scipy.stats import binned_statistic
import sklearn.metrics as metrics


@dataclass
class Line:
    y: np.ndarray
    y_err: Optional[np.ndarray] = None
    y_ref: Optional[np.ndarray] = None
    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: str = "solid"
    fill: bool = False
    vline: bool = False


class Plots:
    """
    Implements the plotting pipeline to evaluate the performance of conditional generative
    networks.
    """

    def __init__(
        self,
        observables: list[Observable],
        losses: dict,
        cl_losses: Optional[dict],
        x_hard: torch.Tensor,
        x_reco: torch.Tensor,
        x_gen_single: torch.Tensor,
        x_gen_dist: torch.Tensor,
        x_compare=None,
        x_hard_pp=None,
        x_reco_pp=None,
        bayesian: bool = False,
        show_metrics: bool = True,
        plot_metrics: bool = False,
        n_unfoldings: int = 1,
        bayesian_samples: int = 1,
        eval_classifier_preds: Optional[torch.Tensor] = None,
        cl_gen_test: Optional[torch.Tensor] = None,
        cl_label_test: Optional[torch.Tensor] = None,
        debug: bool = False,
    ):
        """
        Initializes the plotting pipeline with the data to be plotted.
        Args:
            doc: Documenter object
            observables: List of observables
            losses: Dictionary with loss terms and learning rate as a function of the epoch
            x_test: Data from test dataset
            x_gen_single: Generated data (single point for each test sample)
            x_gen_dist: Generated data (multiple points for subset of test data)
            event_type: Event types for the test dataset
            bayesian: Boolean that indicates if the model is bayesian
        """
        self.observables = observables
        self.losses = losses
        self.cl_losses = cl_losses
        self.compare = x_compare is not None

        self.x_hard = x_hard
        self.x_reco = x_reco
        self.x_hard_pp = x_hard_pp
        self.x_reco_pp = x_reco_pp

        self.obs_hard = []
        self.obs_reco = []
        self.obs_gen_single = []
        self.obs_gen_dist = []
        self.obs_compare = []
        self.bins = []
        for obs in observables:
            o_hard = obs.compute(x_hard)
            self.obs_hard.append(o_hard.cpu().numpy())

            o_reco = obs.compute(x_reco)
            self.obs_reco.append(o_reco.cpu().numpy())

            o_gen_single = obs.compute(x_gen_single)
            self.obs_gen_single.append(o_gen_single.cpu().numpy())

            o_gen_dist = obs.compute(x_gen_dist)
            self.obs_gen_dist.append(o_gen_dist.cpu().numpy())

            self.bins.append(obs.bins(o_hard).cpu().numpy())

            if self.compare:
                o_compare = obs.compute(x_compare).cpu().numpy()
            else:
                o_compare = None
            self.obs_compare.append(o_compare)

        #self.n_unfoldings = x_gen_single.shape[-3]
        self.bayesian = bayesian
        self.show_metrics = show_metrics
        self.colors = [f"C{i}" for i in range(10)]
        self.plot_metrics = plot_metrics
        self.n_unfoldings = n_unfoldings
        self.bayesian_samples = bayesian_samples
        self.eval_classifier_preds = eval_classifier_preds.cpu().numpy() if eval_classifier_preds is not None else None
        self.cl_gen_test = cl_gen_test.cpu().numpy() if cl_gen_test is not None else None
        self.cl_label_test = cl_label_test.cpu().bool().numpy() if cl_label_test is not None else None
        self.debug = debug
        if self.show_metrics:
            print(f"    Computing metrics")
            self.compute_metrics()

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)

    def plot_losses(self, file: str, cl_loss: bool = False):
        """
        Makes plots of the losses (total loss and if bayesian, BCE loss and KL loss
        separately) and learning rate as a function of the epoch.
        Args:
            file: Output file name
        """
        losses = self.losses if not cl_loss else self.cl_losses if self.cl_losses is not None else None
        with PdfPages(file) as pp:
            loss_counter = 0
            self.plot_single_loss(
                pp,
                "loss",
                (losses["tr_loss"], losses["val_loss"]),
                ("train", "val"),
            )
            loss_counter += 2

            bce_bool = False
            mse_bool = False
            kl_bool = False
            nll_bool = False
            lr_bool = False
            movAvg_bool = False

            for l in losses:
                if 'bce' in l:
                    bce_bool = True
                elif 'mse' in l:
                    mse_bool = True
                elif 'kl' in l:
                    kl_bool = True
                elif 'nll' in l:
                    nll_bool = True
                elif 'lr' in l:
                    lr_bool = True
                elif 'movAvg' in l:
                    movAvg_bool = True
                else:
                    continue

            if bce_bool:
                self.plot_single_loss(
                    pp,
                    "BCE loss",
                    (losses["tr_bce"], losses["val_bce"]),
                    ("train", "val"),
                )
                loss_counter += 2
            if mse_bool:
                self.plot_single_loss(
                    pp,
                    "MSE loss",
                    (losses["tr_mse"], losses["val_mse"]),
                    ("train", "val"),
                )
                loss_counter += 2
            if kl_bool:
                self.plot_single_loss(
                    pp,
                    "KL loss",
                    (losses["tr_kl"], losses["val_kl"]),
                    ("train", "val"),
                )
                loss_counter += 2
            if nll_bool:
                self.plot_single_loss(
                    pp,
                    "NLL loss",
                    (losses["tr_nll"], losses["val_nll"]),
                    ("train", "val"),
                )
                loss_counter += 2
            if lr_bool:
                self.plot_single_loss(
                    pp, "learning rate", (losses["lr"],), (None,), "log"
                )
                loss_counter += 1
            if movAvg_bool:
                self.plot_single_loss(
                    pp, "mov. Avg.", (losses["movAvg"], losses["val_movAvg"]), ("train", "val",), "log"
                )
                loss_counter += 2
        
        if loss_counter < len(losses):
            print(f"Not all ({len(losses)}) losses being plotted ({loss_counter})")
            print("Losses:", [l for l in losses])


    def plot_single_loss(
        self,
        pp: PdfPages,
        ylabel: str,
        curves: tuple[np.ndarray],
        labels: tuple[str],
        yscale: str = "linear",
    ):
        """
        Makes single loss plot.
        Args:
            pp: Multipage PDF object
            ylabel: Y axis label
            curves: List of numpy arrays with the loss curves to be plotted
            labels: Labels of the loss curves
            yscale: Y axis scale, "linear" or "log"
        """
        fig, ax = plt.subplots(figsize=(4, 3.5))
        for i, (curve, label) in enumerate(zip(curves, labels)):
            epochs = np.arange(6, len(curve))
            ax.plot(epochs + 1, curve[6:], label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        if any(label is not None for label in labels):
            ax.legend(loc="center right", frameon=False)
        plt.savefig(pp, format="pdf", bbox_inches="tight")
        plt.close()

    def plot_observables(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare, data_gen_classifier in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare, self.cl_gen_test.T
            ):
                print(obs)

                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_reco, y_reco_err = self.compute_hist_data(bins, data_reco, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data_gen[0], bayesian=False) # if it is not bayesian the MAP0 is just the first unfolding
                
                if self.eval_classifier_preds is not None:
                    #print(data_gen_classifier.shape)
                    #print(self.eval_classifier_preds.shape)
                    binned_classes_predict, _, _ = binned_statistic(data_gen_classifier.T, self.eval_classifier_preds[:, 0], bins = bins)
                    #print(binned_classes_predict.shape)
                    #y_predict = np.stack(binned_classes_predict)
                    weights = np.clip(self.eval_classifier_preds[~self.cl_label_test] / (1-self.eval_classifier_preds[~self.cl_label_test]), 0., 200.)
                    y_rew, y_rew_err = self.compute_hist_data(bins, data_gen_classifier[~self.cl_label_test[..., 0]], bayesian=False, weights=weights)#self.eval_classifier_preds[:, 0])
                    #print(y_predict.shape)
                lines = [
                    Line(
                        y=y_reco,
                        y_err=y_reco_err,
                        label="Reco",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label="Unfold",
                        color=self.colors[1],
                    ),
                ]

                data_n = data_gen.shape[-1]
                data_MAP = data_gen[:self.n_unfoldings].reshape(-1)

                if self.bayesian:
                    data_nonMAP = data_gen[self.n_unfoldings:].reshape(len(data_gen[self.n_unfoldings:]), -1)
                    y_gen_full, _ = self.compute_hist_data(bins, data_MAP, bayesian=False, weights=np.full_like(data_MAP, 1. / self.n_unfoldings))
                    _, y_gen_full_err = self.compute_hist_data(bins, data_nonMAP, bayesian=self.bayesian)
                    if self.debug:
                        print("\n data_nonMAP shape:", data_nonMAP.shape)
                        print("MAP error", _)
                        print("nonMAP error", y_gen_full_err)
                else:
                    y_gen_full, y_gen_full_err = self.compute_hist_data(bins, data_MAP, bayesian=False)
                lines.append(
                    Line(
                        y=y_gen_full,
                        y_err=y_gen_full_err,
                        y_ref=y_hard,
                        label=f"{self.n_unfoldings} MAP Unfoldings" if self.bayesian else f"{self.n_unfoldings} Unfoldings",
                        color=self.colors[4],
                    )
                )
                if self.eval_classifier_preds is not None:
                    #lines.append(
                        #Line(
                        #    y=y_predict,
                        #    label=r"$C(x_{\text{hard}})$",
                        #    color=self.colors[3],
                        #))
                    lines.append(
                        Line(
                            y=y_rew,
                            y_err=y_rew_err,
                            y_ref=y_hard,
                            label=r"$\omega_{C}(x)\times \text{Unfold}$",
                            color="black",
                            linestyle="dashed"
                        )
                    )
                if self.compare:
                    y_comp, y_comp_err = self.compute_hist_data(bins, data_compare, bayesian=False)
                    lines.append(
                        Line(
                            y=y_comp,
                            y_err=y_comp_err,
                            y_ref=y_hard,
                            label="SB",
                            color=self.colors[3]
                        )
                    )
                if self.show_metrics:
                    if not self.bayesian:
                        metrics = [obs.metrics["full_emd_mean"],
                                obs.metrics["full_emd_std"],
                                obs.metrics["full_triangle_mean"],
                                obs.metrics["full_triangle_std"],
                                obs.metrics["single_emd_arrs"],
                                obs.metrics["single_triangle_arrs"]]
                    else:
                        metrics = [obs.metrics["MAP_full_emd_mean"],
                                obs.metrics["MAP_full_emd_std"],
                                obs.metrics["MAP_full_triangle_mean"],
                                obs.metrics["MAP_full_triangle_std"],
                                obs.metrics["MAP_single_emd_arrs"],
                                obs.metrics["MAP_single_triangle_arrs"],
                                obs.metrics["non_MAP_single_emd_arrs"],
                                obs.metrics["non_MAP_single_triangle_arrs"]]
                    pass
                    
                self.hist_plot(pp, lines, bins, obs, metrics=metrics if self.show_metrics else None)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_calibration(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes calibration plots for all observables.

        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, data_true, data_gen in zip(
                self.observables, self.obs_hard, self.obs_gen_dist
            ):
                data_true = data_true[None, ...]
                if not self.bayesian:
                    data_gen = data_gen[None, ...]
                #TODO: the commented out changes break non-bayesian calibration plots. please check
                data_true = data_true[:, : data_gen.shape[1]]
                #data_true = data_true[:, : data_gen.shape[-1]]
                calibration_x = np.linspace(0, 1, 101)
                for i, (data_true_elem, data_gen_elem) in enumerate(
                    zip(data_true, data_gen)
                ):
                    #quantiles = np.mean(data_gen_elem < data_true_elem[None, :], axis=0)
                    quantiles = np.mean(data_gen_elem < data_true_elem[:, None], axis=1)
                    calibration_y = np.mean(
                        quantiles[:, None] < calibration_x[None, :], axis=0
                    )
                    plt.plot(
                        calibration_x,
                        calibration_y,
                        color=self.colors[0],
                        alpha=0.8 if self.bayesian else 1,
                    )

                plt.plot([0, 1], [0, 1], color="k", linestyle=":")
                plt.xlabel(f"quantile ${{{obs.tex_label}}}$")
                plt.ylabel("fraction of events")
                plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
                plt.close()
                if pickle_file is not None:
                    pickle_data.append({"data_true": data_true, "data_gen": data_gen, "obs": obs})
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_pulls(self, file: str):
        """
        Plots pulls for all observables.
        Args:
            file: Output file name
        """

        bins = np.linspace(-5, 5, 40)
        with PdfPages(file) as pp:
            for obs, data_true, data_gen in zip(
                self.observables, self.obs_hard, self.obs_gen_dist
            ):
                data_true = data_true[: data_gen.shape[-2]]
                gen_mean = np.mean(data_gen, axis=-1)
                gen_std = np.std(data_gen, axis=-1)
                pull = (gen_mean - data_true) / gen_std
                y, y_err = self.compute_hist_data(bins, pull)
                lines = [
                    Line(
                        y=y,
                        y_err=y_err,
                        label="Pull",
                        color=self.colors[0],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs, show_ratios=False)

    def plot_single_events(self, file: str, pickle_file: Optional[str] = None):
        """
        Plots single event distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for j, (obs, bins, data_true, condition, data_gen) in enumerate(zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_dist
            )):
                for i in range(10):
                    x_true = data_gen
                    #TODO: the commented out changes break non-bayesian single event plots. please check
                    #y_gen, y_gen_err = self.compute_hist_data(bins, data_gen[..., :, i])
                    y_gen, y_gen_err = self.compute_hist_data(bins, data_gen[..., i, :])
                    event_reco = condition[i]
                    std_j = self.x_reco[:, j].cpu().std(0)
                    mask = np.abs(self.x_reco[:, j].cpu() - event_reco) < 0.01 * std_j
                    close_events_reco = self.x_reco[mask, j].cpu()
                    close_events_hard = self.x_hard[mask, j].cpu()
                    y_gt, y_gt_err = self.compute_hist_data(bins, close_events_hard)


                    lines = [
                        Line(
                            y=data_true[i],
                            label="Truth",
                            color=self.colors[0],
                            vline=True,
                        ),
                        Line(
                            y=condition[i],
                            label="Cond.",
                            color=self.colors[2],
                            vline=True,
                            linestyle="dashed"
                        ),
                        Line(
                            y=y_gen,
                            y_ref=None,
                            label="Gen",
                            color=self.colors[1],
                        ),
                        #Line(
                        #    y=np.mean(data_gen[..., i, :]),
                        #    y_ref=None,
                        #    label="Gen mean",
                        #    color=self.colors[1],
                        #    vline = True,
                        #    linestyle="dotted"
                        #),
                        Line(
                            y=y_gt,
                            y_ref=None,
                            label="cond. GT ",
                            color=self.colors[3],
                            linestyle="solid"
                        ),
                    ]
                    self.hist_plot(
                        pp, lines, bins, obs, title=f"Event {i+1}, true: {data_true[i]:.2f}, mean distr. {np.mean(data_gen[..., i, :]):.2f}", show_ratios=False, legend_kwargs={"loc": "best"}
                    )
                    if pickle_file is not None:
                        pickle_data.append({"lines": lines, "bins": bins, "obs": obs, "event_number": i+1, "reco": condition[i], "hard": data_true[i], "gen": data_gen[..., i, :]})
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_preprocessed(self, file: str):
        """
        Plots preprocessed observable distributions
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, data_hard_pp, data_reco_pp in zip(
                    self.observables, self.x_hard_pp.T, self.x_reco_pp.T
            ):
                bins = 100
                y_hard, bins = np.histogram(data_hard_pp.cpu().numpy(), bins=bins, density=True)
                y_reco, _ = np.histogram(data_reco_pp.cpu().numpy(), bins=bins, density=True)
                #y_gen, y_gen_err = self.compute_hist_data(bins, data_gen, bayesian=self.bayesian)
                normal = np.random.normal(size=data_hard_pp.shape)
                y_normal, _ = np.histogram(normal, bins=bins, density=True)
                lines = [
                    Line(
                        y=y_reco,
                        y_err=None,
                        label="Reco PP",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=None,
                        label="Hard PP",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_normal,
                        y_err=None,
                        label="Normal",
                        color=self.colors[5],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs, metrics=None, show_ratios=False, no_scale=True)

    def compute_hist_data(self, bins: np.ndarray, data: np.ndarray, bayesian=False, weights=None):
        if bayesian:
            hists = np.stack(
                [np.histogram(d, bins=bins, density=False, weights=weights)[0] for d in data], axis=0
            )
            y = hists[0]
            y_err = np.std(hists, axis=0)
        else:
            y, _ = np.histogram(data, bins=bins, density=False, weights=weights)
            y_err = np.sqrt(y)
        return y, y_err

    def hist_plot(
        self,
        pdf: PdfPages,
        lines: list[Line],
        bins: np.ndarray,
        observable: Observable,
        show_ratios: bool = True,
        title: Optional[str] = None,
        legend_kwargs: Optional[dict] = None,
        no_scale: bool = False,
        yscale: Optional[str] = None,
        metrics = None,
    ):
        """
        Makes a single histogram plot, used for the observable histograms and clustering
        histograms.
        Args:
            pdf: Multipage PDF object
            lines: List of line objects describing the histograms
            bins: Numpy array with the bin boundaries
            show_ratios: If True, show a panel with ratios
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            n_panels = 1 + int(show_ratios) + int(metrics is not None)
            fig, axs = plt.subplots(
                n_panels,
                1,
                sharex=True,
                figsize=(6, 4.5),
                gridspec_kw={"height_ratios": (12, 3, 1)[:n_panels], "hspace": 0.00},
            )
            if n_panels == 1:
                axs = [axs]

            for line in lines:
                if line.vline:
                    axs[0].axvline(line.y, label=line.label, color=line.color, linestyle=line.linestyle)
                    continue
                integral = np.sum((bins[1:] - bins[:-1]) * line.y)
                scale = 1 / integral if integral != 0.0 else 1.0
                if line.y_ref is not None:
                    ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                    ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
                if no_scale:
                    scale = 1.
                    ref_scale = 1.
                
                if self.debug: print("Actual values plotted:", line.y * scale)
                self.hist_line(
                    axs[0],
                    bins,
                    line.y * scale,
                    line.y_err * scale if line.y_err is not None else None,
                    label=line.label,
                    color=line.color,
                    fill=line.fill,
                    linestyle=line.linestyle
                )

                if line.y_ref is not None:
                    ratio = (line.y * scale) / (line.y_ref * ref_scale)
                    ratio_isnan = np.isnan(ratio)
                    if line.y_err is not None:
                        if len(line.y_err.shape) == 2:
                            ratio_err = (line.y_err * scale) / (line.y_ref * ref_scale)
                            ratio_err[:, ratio_isnan] = 0.0
                        else:
                            ratio_err = np.sqrt((line.y_err / line.y) ** 2)
                            ratio_err[ratio_isnan] = 0.0
                    else:
                        ratio_err = None
                    ratio[ratio_isnan] = 1.0
                    self.hist_line(
                        axs[1], bins, ratio, ratio_err, label=None, color=line.color, linestyle=line.linestyle,
                    )

            axs[0].legend(frameon=False, **(legend_kwargs if legend_kwargs is not None else {}))
            axs[0].set_ylabel("normalized")
            axs[0].set_yscale(observable.yscale if yscale is None else yscale)
            if title is not None:
                self.corner_text(axs[0], title, "left", "top")

            if show_ratios:
                axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Truth}}$")
                axs[1].set_yticks([0.95, 1, 1.05])
                axs[1].set_ylim([0.9, 1.1])
                axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
                axs[1].axhline(y=1.05, c="black", ls="dotted", lw=0.5)
                axs[1].axhline(y=0.95, c="black", ls="dotted", lw=0.5)

            if metrics is not None:
                axs[-1].text(bins[0], 0.2, f"10*EMD: {metrics[0]:.4f} $\pm$ {metrics[1]:.5f}"
                                        f"    ;    1e3*TriDist: {metrics[2]:.5f} $\pm$ "
                                        f"{metrics[3]:.4f}", fontsize=13)
                axs[-1].set_yticks([])
            unit = "" if observable.unit is None else f" [{observable.unit}]"
            axs[-1].set_xlabel(f"${{{observable.tex_label}}}${unit}")
            axs[-1].set_xscale(observable.xscale)
            axs[-1].set_xlim(bins[0], bins[-1])
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close()

    def hist_line(
        self,
        ax: mpl.axes.Axes,
        bins: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        label: str,
        color: str,
        linestyle: str = "solid",
        fill: bool = False,
    ):
        """
        Plot a stepped line for a histogram, optionally with error bars.
        Args:
            ax: Matplotlib Axes
            bins: Numpy array with bin boundaries
            y: Y values for the bins
            y_err: Y errors for the bins
            label: Label of the line
            color: Color of the line
            linestyle: line style
            fill: Filled histogram
        """

        dup_last = lambda a: np.append(a, a[-1])

        if fill:
            ax.fill_between(
                bins, dup_last(y), label=label, facecolor=color, step="post", alpha=0.2
            )
        else:
            ax.step(
                bins,
                dup_last(y),
                label=label,
                color=color,
                linewidth=1.0,
                where="post",
                ls=linestyle,
            )
        if y_err is not None:
            if len(y_err.shape) == 2:
                y_low = y_err[0]
                y_high = y_err[1]
            else:
                y_low = y - y_err
                y_high = y + y_err

            ax.step(
                bins,
                dup_last(y_high),
                color=color,
                alpha=0.5,
                linewidth=0.5,
                where="post",
            )
            ax.step(
                bins,
                dup_last(y_low),
                color=color,
                alpha=0.5,
                linewidth=0.5,
                where="post",
            )
            ax.fill_between(
                bins,
                dup_last(y_low),
                dup_last(y_high),
                facecolor=color,
                alpha=0.3,
                step="post",
            )

    def corner_text(
        self, ax: mpl.axes.Axes, text: str, horizontal_pos: str, vertical_pos: str
    ):
        ax.text(
            x=0.95 if horizontal_pos == "right" else 0.05,
            y=0.95 if vertical_pos == "top" else 0.05,
            s=text,
            horizontalalignment=horizontal_pos,
            verticalalignment=vertical_pos,
            transform=ax.transAxes,
        )
        # Dummy line for automatic legend placement
        plt.plot(
            0.8 if horizontal_pos == "right" else 0.2,
            0.8 if vertical_pos == "top" else 0.2,
            transform=ax.transAxes,
            color="none",
        )

    def plot_migration(self, file: str, gt_hard=False):
        if gt_hard:
            obs_hard = self.obs_hard
            name_hard = "Hard"
        else:
            obs_hard = self.obs_gen_single
            name_hard = "Unfold"
        cmap = plt.get_cmap('viridis')
        cmap.set_bad("white")
        with PdfPages(file) as pp:
            for obs, bins, data_reco, data_hard in zip(
                self.observables, self.bins, self.obs_reco, obs_hard
                    ):
                if self.bayesian and not gt_hard:
                    plt.hist2d(data_reco, data_hard[0], density=True, bins=bins, rasterized=True, cmap=cmap, norm=mpl.colors.LogNorm())
                else:
                    plt.hist2d(data_reco, data_hard, density=True, bins=bins, rasterized=True, cmap=cmap,
                               norm=mpl.colors.LogNorm())
                unit = "" if obs.unit is None else f" [{obs.unit}]"
                plt.title(f"${{{obs.tex_label}}}${unit}")
                plt.xlabel("Reco")
                plt.ylabel(name_hard)
                plt.xlim(bins[0], bins[-1])
                plt.ylim(bins[0], bins[-1])
                plt.colorbar()
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

    def plot_migration2(self, file: str, gt_hard=False, SB_hard=False, pickle_file: Optional[str] = None):
        if gt_hard:
            obs_hard = self.obs_hard
            name_hard = "Hard"
        elif SB_hard:
            obs_hard = self.obs_compare
            name_hard = "SB"
        else:
            obs_hard = self.obs_gen_single
            name_hard = "Unfold"
        cmap = plt.get_cmap('viridis')
        cmap.set_bad("white")

        ranges = [
            [0, 0.4],
            [0, 0.3],
            [0, 0.5],
            [0, 0.5],
            [0, 0.2],
            [0, 0.25]
        ]
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_reco, data_hard, r in zip(
                self.observables, self.bins, self.obs_reco, obs_hard, ranges
                    ):
                cmap = plt.get_cmap('viridis')
                cmap.set_bad("white")
                if gt_hard or SB_hard:
                    h, x, y = np.histogram2d(data_hard, data_reco, bins=(bins, bins))
                else:
                    h, x, y = np.histogram2d(data_hard[0], data_reco, bins=(bins, bins))
                h = np.ma.divide(h, np.sum(h, -1, keepdims=True)).filled(0)
                h[h == 0] = np.nan
                plt.pcolormesh(bins, bins, h, cmap=cmap, rasterized=True, vmin=r[0], vmax=r[1])
                plt.colorbar()

                unit = "" if obs.unit is None else f" [{obs.unit}]"
                plt.title(f"${{{obs.tex_label}}}${unit}")
                plt.xlim(bins[0], bins[-1])
                plt.ylim(bins[0], bins[-1])
                plt.xlabel("Reco")
                plt.ylabel(name_hard)

                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()
                if pickle_file is not None:
                    pickle_data.append({"h": h, "bins": bins, "ranges": r, "obs": obs})
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def compute_metrics(self):
        for i, obs in enumerate(self.observables):
            obs.metrics = {}
            x_gen = self.obs_gen_single[i]
            x_true = self.obs_hard[i]
            bins = self.bins[i]

            if not self.bayesian:
                single_emd_arrs = []
                single_triangle_arrs = []
                for n in range(self.n_unfoldings):
                    print("Metrics for unfolding", n)
                    emd = GetEMD(x_true, x_gen[n], nboot=0)
                    triangle = get_triangle_distance(x_true, x_gen[n], bins, nboot=0)
                    single_emd_arrs.append(emd)
                    single_triangle_arrs.append(triangle)
                print(f"Saving metrics after {n+1} unfoldings\n")

                obs.metrics["single_emd_arrs"] = np.array(single_emd_arrs)
                obs.metrics["single_triangle_arrs"] = np.array(single_triangle_arrs)

                full_emd_mean, full_emd_std = GetEMD(x_true, x_gen.flatten(), nboot=3)
                full_triangle_mean, full_triangle_std = get_triangle_distance(x_true, x_gen.flatten(), bins, nboot=3)
                obs.metrics["full_emd_mean"] = round(full_emd_mean, 4)
                obs.metrics["full_emd_std"] = round(full_emd_std, 4)
                obs.metrics["full_triangle_mean"] = round(full_triangle_mean, 4)
                obs.metrics["full_triangle_std"] = round(full_triangle_std, 4)

            else:
                # MAP single unfolding metrics
                MAP_single_emd_arrs = []
                MAP_single_triangle_arrs = []
                
                # non-MAP single unfolding metrics
                nonMAP_single_emd_arrs = []
                nonMAP_single_triangle_arrs = []

                # n_unfoldings with MAP
                for n in range(self.n_unfoldings):
                    print("Metrics for MAP unfolding", n)
                    x_gen_MAP_nth_unfolding = x_gen[n]

                    emd = GetEMD(x_true, x_gen_MAP_nth_unfolding, nboot=0)
                    triangle = get_triangle_distance(x_true, x_gen_MAP_nth_unfolding, bins, nboot=0)
                    MAP_single_emd_arrs.append(emd)
                    MAP_single_triangle_arrs.append(triangle)
                
                print(f"Saving metrics after {n+1} unfoldings for the MAP\n")
                # emd on MAP single unfoldings
                obs.metrics["MAP_single_emd_arrs"] = np.array(MAP_single_emd_arrs)
                # triangle on MAP single unfoldings
                obs.metrics["MAP_single_triangle_arrs"] = np.array(MAP_single_triangle_arrs)


                ## MAP full unfolding
                MAP_full_unfolding = x_gen[:self.n_unfoldings].flatten()
                # emd on MAP full unfolding
                MAP_full_emd_mean, MAP_full_emd_std = GetEMD(x_true, MAP_full_unfolding, nboot=3)
                obs.metrics["MAP_full_emd_mean"] = round(MAP_full_emd_mean, 4)
                obs.metrics["MAP_full_emd_std"] = round(MAP_full_emd_std, 5)
                # triangle on MAP full unfolding
                MAP_full_triangle_mean, MAP_full_triangle_std = get_triangle_distance(x_true, MAP_full_unfolding, bins, nboot=3)
                obs.metrics["MAP_full_triangle_mean"] = round(MAP_full_triangle_mean, 4)
                obs.metrics["MAP_full_triangle_std"] = round(MAP_full_triangle_std, 5)
                
                # (bayesian_samples - n_unfoldings) unfoldings with non-MAP
                for b in range(self.n_unfoldings, len(x_gen)):
                    print("Metrics for bay. sample", b)
                    x_gen_nonMAP_single = x_gen[b]
                    emd = GetEMD(x_true, x_gen_nonMAP_single, nboot=0)
                    triangle = get_triangle_distance(x_true, x_gen_nonMAP_single, bins, nboot=0)
                    nonMAP_single_emd_arrs.append(emd)
                    nonMAP_single_triangle_arrs.append(triangle)
                print(f"Saving metrics after {len(x_gen)-self.n_unfoldings} bayesian samples for non-MAP(s)\n") if len(x_gen) > self.n_unfoldings else print("No non-MAP unfoldings\n")   
                # emd on non-MAP single unfoldings
                obs.metrics["non_MAP_single_emd_arrs"] = np.array(nonMAP_single_emd_arrs)
                # triangle on non-MAP single unfoldings
                obs.metrics["non_MAP_single_triangle_arrs"] = np.array(nonMAP_single_triangle_arrs)


    def hist_metrics_unfoldings(self, file: str, pickle_file: Optional[str] = None):
        pickle_data = {'emd': [], 'triangle': []}
        with PdfPages(file) as pp:
            for obs in self.observables:
                nbins = 64
                if self.bayesian:
                    emd_bins            =   np.linspace(0, 1.5 * max(max(obs.metrics["non_MAP_single_emd_arrs"]), max(obs.metrics["MAP_single_emd_arrs"])), nbins)
                    emd_nonMAP_hist, _         =   np.histogram(obs.metrics["non_MAP_single_emd_arrs"], bins=emd_bins, density=False)
                    emd_MAP_hist, _         =   np.histogram(obs.metrics["MAP_single_emd_arrs"], bins=emd_bins, density=False)
                    triangle_bins            =   np.linspace(0, 1.5 * max(max(obs.metrics["non_MAP_single_triangle_arrs"]), max(obs.metrics["MAP_single_triangle_arrs"])), nbins)
                    triangle_nonMAP_hist, _         =   np.histogram(obs.metrics["non_MAP_single_triangle_arrs"], bins=triangle_bins, density=False)
                    triangle_MAP_hist, _         =   np.histogram(obs.metrics["MAP_single_triangle_arrs"], bins=triangle_bins, density=False)

                    emd_lines = [
                        Line(
                            y=emd_nonMAP_hist,
                            y_err=None,
                            label=f"EMD non-MAP {len(obs.metrics['non_MAP_single_emd_arrs'])} single-Bayesians",
                            color=self.colors[0],
                            linestyle='dashed',
                        ),
                        Line(
                            y=emd_MAP_hist,
                            y_err=None,
                            label=f"EMD MAP {self.n_unfoldings} Unfoldings",
                            color=self.colors[0],
                            linestyle='solid',
                        ),
                        Line(
                            y=obs.metrics["MAP_full_emd_mean"],
                            vline=True,
                            y_err=None,
                            label="EMD MAP Full",
                            color=self.colors[1],
                            linestyle='dashed',
                        )
                    ]
                    triangle_lines = [
                        Line(
                            y=triangle_nonMAP_hist,
                            y_err=None,
                            label=f"Triangle non-MAP {len(obs.metrics['non_MAP_single_triangle_arrs'])} single-Bayesians",
                            color=self.colors[0],
                            linestyle='dashed',
                        ),
                        Line(
                            y=triangle_MAP_hist,
                            y_err=None,
                            label=f"Triangle MAP {self.n_unfoldings} Unfoldings",
                            color=self.colors[0],
                            linestyle='solid',
                        ),
                        Line(
                            y=obs.metrics["MAP_full_triangle_mean"],
                            vline=True,
                            y_err=None,
                            label="Triangle MAP Full",
                            color=self.colors[1],
                            linestyle='dashed',
                        )
                    ]
                else:
                    emd_bins            =   np.linspace(0, 1.5 * max(obs.metrics["single_emd_arrs"]), nbins)
                    emd_hist, _         =   np.histogram(obs.metrics["single_emd_arrs"], bins=emd_bins, density=False)
                    triangle_bins            =   np.linspace(0, 1.5 * max(obs.metrics["single_triangle_arrs"]), nbins)
                    triangle_hist, _         =   np.histogram(obs.metrics["single_triangle_arrs"], bins=triangle_bins, density=False)

                    emd_lines = [
                        Line(
                            y=emd_hist,
                            y_err=None,
                            label=f"EMD {self.n_unfoldings} Unfoldings",
                            color=self.colors[0],
                            linestyle='solid',
                        ),
                        Line(
                            y=obs.metrics["full_emd_mean"],
                            vline=True,
                            y_err=None,
                            label="EMD Full",
                            color=self.colors[1],
                            linestyle='dashed',
                        )
                    ]
                    triangle_lines = [
                        Line(
                            y=triangle_hist,
                            y_err=None,
                            label=f"Triangle {self.n_unfoldings} Unfoldings",
                            color=self.colors[0],
                            linestyle='solid',
                        ),
                        Line(
                            y=obs.metrics["full_triangle_mean"],
                            vline=True,
                            y_err=None,
                            label="Triangle Full",
                            color=self.colors[1],
                            linestyle='dashed',
                        )
                    ]

                self.hist_plot(pp, emd_lines, emd_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')
                self.hist_plot(pp, triangle_lines, triangle_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')

                if pickle_file is not None:
                    pickle_data["emd"].append(obs.metrics["emd_arr"])
                    pickle_data["triangle"].append(obs.metrics["triangle_arr"])
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def hist_metrics_bayesian(self, file: str, pickle_file: Optional[str] = None):
        pickle_data = {'emd': [], 'triangle': []}

        with PdfPages(file) as pp:
            for obs in self.observables:
                nbins = 64
                emd_bins = np.linspace(0, 1.5 * max(obs.metrics["emd_arr"][:, 0]), nbins)
                triangle_bins = np.linspace(0, 1.5 * max(obs.metrics["triangle_arr"][:, 0]), nbins)
                emd_hist, _ = np.histogram(obs.metrics["emd_arr"][:, 0], bins=emd_bins, density=False)
                triangle_hist, _ = np.histogram(obs.metrics["triangle_arr"][:, 0], bins=triangle_bins, density=False)

                emd_lines = [
                    Line(
                        y=emd_hist,
                        y_err=None,
                        label="EMD Bayesians",
                        color=self.colors[0],
                    ),
                    Line(
                        y=obs.metrics["emd_full_full"],
                        vline=True,
                        y_err=None,
                        label="EMD FullFull",
                        color=self.colors[1],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["emd_arr"][0][0],
                        vline=True,
                        y_err=None,
                        label="EMD MAP",
                        color=self.colors[2],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["emd_full"][0],
                        vline=True,
                        y_err=None,
                        label="EMD MAP FullUn",
                        color=self.colors[3],
                        linestyle='dashed',
                    )
                ]
                triangle_lines = [
                    Line(
                        y=triangle_hist,
                        y_err=None,
                        label="Triangle Bayesians",
                        color=self.colors[0],
                    ),
                    Line(
                        y=obs.metrics["triangle_full_full"],
                        vline=True,
                        y_err=None,
                        label="Triangle FullFull",
                        color=self.colors[1],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["triangle_arr"][0][0],
                        vline=True,
                        y_err=None,
                        label="Triangle MAP",
                        color=self.colors[2],
                        linestyle='dashed',
                    ),
                    Line(
                        y=obs.metrics["triangle_full"][0],
                        vline=True,
                        y_err=None,
                        label="Triangle MAP FullUn",
                        color=self.colors[3],
                        linestyle='dashed',
                    )
                ]

                self.hist_plot(pp, emd_lines, emd_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')
                self.hist_plot(pp, triangle_lines, triangle_bins, obs, metrics=None, show_ratios=False, no_scale=True, yscale='log')

                if pickle_file is not None:
                    pickle_data["emd"].append(obs.metrics["emd_arr"])
                    pickle_data["triangle"].append(obs.metrics["triangle_arr"])
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_multiple_unfoldings(self, file: str):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare
            ):
                if self.bayesian:
                    data_MAP = data_gen[:self.n_unfoldings]
                    data_nonMAP = data_gen[self.n_unfoldings:]
                    y_gen, y_gen_err = self.compute_hist_data(bins, data_MAP.reshape(-1), bayesian=False)
                else:
                    data_MAP = data_gen # we can just call all data_MAP for the non-bayesian
                    y_gen, y_gen_err = self.compute_hist_data(bins, data_MAP.reshape(-1), bayesian=False)
                
                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                
                lines = [
                    Line(
                        y=y_hard,
                        y_err=None,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=None,
                        y_ref=y_hard,
                        label=f"{self.n_unfoldings} MAP Unfoldings" if self.bayesian else f"{self.n_unfoldings} Unfoldings",
                        color=self.colors[4],
                    ),
                ]

                hists = np.stack(
                    [np.histogram(d, bins=bins, density=True)[0] for d in data_MAP], axis=0
                )
                y_gen_mean = np.mean(hists, axis=0)

                lines.append(
                    Line(
                        y=y_gen_mean,
                        y_err=None,
                        y_ref=y_hard,
                        label=f"MAP Mean" if self.bayesian else f"Unfoldings Mean",
                        color=self.colors[2]
                    )
                )

                obs_emds = obs.metrics["MAP_single_emd_arrs"] if self.bayesian else obs.metrics["single_emd_arrs"]
                emd_argsort = np.argsort(obs_emds)

                for i in range(2):
                    index = emd_argsort[i]
                    d = data_MAP[index]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"MAP Best, {index}" if self.bayesian else f"Unf. Best, {index}",
                            color=self.colors[5+i],
                            linestyle="dashed"
                        ) 
                    )
                    index = emd_argsort[-(i+1)]
                    d = data_MAP[index]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"MAP Worst, {index}" if self.bayesian else f"Unf. Worst, {index}",
                            color=self.colors[7 + i],
                            linestyle="dashed"
                        )
                    )
                self.hist_plot(pp, lines, bins, obs, metrics=None)

    def plot_multiple_bayesians(self, file: str):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare
            ):
                data_MAP0 = data_gen[0]
                data_nonMAP = data_gen[self.n_unfoldings:]
                
                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_gen_MAP, y_gen_MAP_err = self.compute_hist_data(bins, data_MAP0, bayesian=False)
                lines = [
                    Line(
                        y=y_hard,
                        y_err=None,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen_MAP,
                        y_err=None,
                        y_ref=y_hard,
                        label="MAP single",
                        color=self.colors[1],
                    ),
                ]

                hists = np.stack(
                    [np.histogram(d, bins=bins, density=True)[0] for d in data_nonMAP], axis=0
                )
                y_gen_mean = np.mean(hists, axis=0)

                lines.append(
                    Line(
                        y=y_gen_mean,
                        y_err=None,
                        y_ref=y_hard,
                        label=f"non-MAP Mean",
                        color=self.colors[2]
                    )
                )

                obs_emds = obs.metrics["non_MAP_single_emd_arrs"]
                emd_argsort = np.argsort(obs_emds)

                for i in range(2):
                    index = emd_argsort[i]
                    d = data_nonMAP[index]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"non-MAP Best, {index}",
                            color=self.colors[5+i],
                            linestyle="dashed"
                        )
                    )
                    index = emd_argsort[-(i+1)]
                    d = data_nonMAP[index]
                    y, _ = self.compute_hist_data(bins, d, bayesian=False)
                    lines.append(
                        Line(
                            y=y,
                            y_err=None,
                            y_ref=y_hard,
                            label=f"non-MAP Worst, {index}",
                            color=self.colors[7 + i],
                            linestyle="dashed"
                        )
                    )
                self.hist_plot(pp, lines, bins, obs, metrics=None)
    '''
    def plot_observables_full(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco, data_gen, data_compare in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco, self.obs_gen_single, self.obs_compare
            ):
                if self.bayesian:
                    bay_n, dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(bay_n, -1)
                else:
                    dist_n, data_n = data_gen.shape
                    data = data_gen.reshape(-1)


                y_hard, y_hard_err = self.compute_hist_data(bins, data_hard, bayesian=False)
                y_reco, y_reco_err = self.compute_hist_data(bins, data_reco, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data, bayesian=self.bayesian)
                lines = [
                    Line(
                        y=y_reco,
                        y_err=y_reco_err,
                        label="Reco",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Hard",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label=f"MAP {dist_n} Unf. full" if self.bayesian else f"{dist_n} Unf. full",
                        color=self.colors[1],
                    ),

                ]
                if self.bayesian:
                    y_gen_full, _ = self.compute_hist_data(bins, data.reshape(-1), bayesian=False)
                    lines.append(Line(
                        y=y_gen_full,
                        y_err=None,
                        y_ref=y_hard,
                        label=f"{bay_n} bay., {dist_n} Unf.",
                        color=self.colors[3],
                    ))
                if self.compare:
                    y_comp, y_comp_err = self.compute_hist_data(bins, data_compare, bayesian=False)
                    lines.append(
                        Line(
                            y=y_comp,
                            y_err=y_gen_err,
                            y_ref=y_hard,
                            label="SB",
                            color=self.colors[3]
                        )
                    )
                if not self.bayesian:
                    metrics = [obs.metrics["emd_full"][0],
                               obs.metrics["emd_full_std"][0],
                               obs.metrics["triangle_full"][0],
                               obs.metrics["triangle_full_std"][0]]
                else:
                    metrics = [obs.metrics["emd_full_full"],
                               obs.metrics["emd_full_full_std"],
                               obs.metrics["triangle_full_full"],
                               obs.metrics["triangle_full_full_std"]]
                self.hist_plot(pp, lines, bins, obs, metrics=metrics)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)
    '''
    def save_metrics(self, pickle_file):

        with open(pickle_file, "wb") as f:
            pickle.dump(self.observables, f)
    


class OmnifoldPlots(Plots):
    def __init__(
        self,
        observables: list[Observable],
        losses: dict,
        x_hard: torch.Tensor,
        x_reco: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        bayesian: bool = False,
        show_metrics: bool = True,
        pythia_only: bool = False,
        debug: bool = False,
        invert_reweighting: bool = False,
    ):
        """
        Initializes the plotting pipeline with the data to be plotted.
        Args:
            doc: Documenter object
            observables: List of observables
            losses: Dictionary with loss terms and learning rate as a function of the epoch
            x_hard: Hard level data
            x_reco: Reco level data
            labels: True labels
            predictions: Predicted labels
        """
        self.observables = observables
        self.losses = losses
        if invert_reweighting: print("Reweighting Herwig onto Pythia")
        self.labels = labels.cpu().bool().numpy()
        self.predictions = predictions.cpu().numpy() if not invert_reweighting else (1-predictions).cpu().numpy()
        self.weights = np.clip((1. - self.predictions)/self.predictions, 0., 200).squeeze()
        self.bayesian = bayesian
        self.show_metrics = show_metrics
        self.pythia_only = pythia_only
        self.debug = debug
        self.invert_reweighting = invert_reweighting

        self.obs_hard = []
        self.obs_reco = []
        self.bins = []
        for obs in observables:
            o_hard = obs.compute(x_hard)
            o_reco = obs.compute(x_reco)
            self.obs_hard.append(o_hard.cpu().numpy())
            self.obs_reco.append(o_reco.cpu().numpy())
            self.bins.append(obs.bins(o_hard).cpu().numpy())

        if self.show_metrics:
            print(f"    Computing metrics for observables")
            self.compute_metrics()
            print(f"    Computing metrics for reco datasets")
            self.compute_metrics_reco()
            print(f"    Computing metrics for hard datasets")
            self.compute_metrics_hard()
        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        self.colors = [f"C{i}" for i in range(10)]

    def hist_weights_plot(
        self,
        pdf: PdfPages,
        lines: list[Line],
        bins: np.ndarray,
        show_ratios: bool = False,
        title: Optional[str] = None,
        no_scale: bool = False,
        yscale: Optional[str] = None,
        show_metrics: bool = False,
        ylim: tuple[float, float] = None,
    ):
        """
        Makes a single histogram plot for the weights
        Args:
            pdf: Multipage PDF object
            lines: List of line objects describing the histograms
            bins: Numpy array with the bin boundaries
            show_ratios: If True, show a panel with ratios
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            n_panels = 1 + int(show_ratios) + int(show_metrics)
            fig, axs = plt.subplots(
                n_panels,
                1,
                sharex=True,
                figsize=(6, 4.5),
                gridspec_kw={"height_ratios": (12, 3, 1)[:n_panels], "hspace": 0.00},
            )
            if n_panels == 1:
                axs = [axs]

            for line in lines:
                if line.vline:
                    axs[0].axvline(line.y, label=line.label, color=line.color, linestyle=line.linestyle)
                    continue
                integral = np.sum((bins[1:] - bins[:-1]) * line.y)
                scale = 1 / integral if integral != 0.0 else 1.0
                if line.y_ref is not None:
                    ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                    ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
                if no_scale:
                    scale = 1.
                    ref_scale = 1.

                self.hist_line(
                    axs[0],
                    bins,
                    line.y * scale,
                    line.y_err * scale if line.y_err is not None else None,
                    label=line.label,
                    color=line.color,
                    fill=line.fill,
                    linestyle=line.linestyle
                )

                if line.y_ref is not None:
                    ratio = (line.y * scale) / (line.y_ref * ref_scale)
                    ratio_isnan = np.isnan(ratio)
                    if line.y_err is not None:
                        if len(line.y_err.shape) == 2:
                            ratio_err = (line.y_err * scale) / (line.y_ref * ref_scale)
                            ratio_err[:, ratio_isnan] = 0.0
                        else:
                            ratio_err = np.sqrt((line.y_err / line.y) ** 2)
                            ratio_err[ratio_isnan] = 0.0
                    else:
                        ratio_err = None
                    ratio[ratio_isnan] = 1.0
                    self.hist_line(
                        axs[1], bins, ratio, ratio_err, label=None, color=line.color
                    )

            axs[0].legend(frameon=False)
            axs[0].set_ylabel("normalized")
            axs[0].set_yscale("log" if yscale is None else yscale)
            if ylim is not None:
                axs[0].set_ylim(*ylim)
            if title is not None:
                self.corner_text(axs[0], title, "left", "top")

            axs[-1].set_xlabel(f"$w(x)$")
            axs[-1].set_xscale("log")
            axs[-1].set_xlim(bins[0], bins[-1])
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close()

    def compute_hist_data(self, bins: np.ndarray, data: np.ndarray, bayesian=False, weights=None):
        if bayesian:
            hists = np.stack(
                [np.histogram(data, bins=bins, density=True, weights=weight_sample)[0] for weight_sample in weights], axis=0
            )
            #y = np.median(hists, axis=0)
            y = hists[0]
            #y_err = np.stack(
            #    (np.quantile(hists, 0.159, axis=0), np.quantile(hists, 0.841, axis=0)),
            #    axis=0,
            #)
            y_err = np.std(hists, axis=0)
        else:
            y, _ = np.histogram(data, bins=bins, density=False, weights=weights)
            y_err = np.sqrt(y)
        return y, y_err

    def plot_classes(self, file: str):
        """
        Makes plots of truth and predicted classes for all observables.
        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, bins, data in zip(self.observables, self.bins, self.obs_reco):
                binned_classes_test, _, _ = binned_statistic(data, self.labels.T, bins=bins) if not self.invert_reweighting else binned_statistic(data, ~self.labels.T, bins=bins)
                if self.bayesian:
                    binned_classes_predict, _, _ = binned_statistic(data, self.predictions[0].T, bins=bins)
                    binned_weights_predict, _, _ = binned_statistic(data[self.labels.squeeze()], self.weights[0][self.labels.squeeze()].T, statistic = "mean", bins=bins) if not self.invert_reweighting else binned_statistic(data[~self.labels.squeeze()], self.weights[0][~self.labels.squeeze()].T, statistic = "mean", bins=bins) # this is the mean of the weights for Pythia1/Herwig in each bin
                    n_herwig, _, _ = binned_statistic(x = data[~self.labels.squeeze()], values =None, statistic = "count",  bins=bins)
                    n_pythia, _, _ = binned_statistic(x = data[self.labels.squeeze()], values= None, statistic = "count",  bins=bins)
                else:
                    binned_classes_predict, _, _ = binned_statistic(data, self.predictions.T, bins=bins)
                    binned_weights_predict, _, _ = binned_statistic(data[self.labels.squeeze()], self.weights[self.labels.squeeze()].T, statistic = "mean", bins=bins) if not self.invert_reweighting else binned_statistic(data[~self.labels.squeeze()], self.weights[~self.labels.squeeze()].T, statistic = "mean", bins=bins) # this is the mean of the weights for Pythia1/Herwig in each bin
                    n_herwig, _, _ = binned_statistic(x = data[~self.labels.squeeze()], values =None, statistic = "count",  bins=bins)
                    n_pythia, _, _ = binned_statistic(x = data[self.labels.squeeze()], values= None, statistic = "count",  bins=bins)
                bin_totals, _ = np.histogram(data, bins=bins)
                y_true = np.stack(binned_classes_test, axis=1)
                y_predict = np.stack(binned_classes_predict, axis=1)
                y_true_err = np.sqrt(y_true * (1 - y_true) / bin_totals[:, None])

                lines = []
                lines.append(Line(
                    y=y_true[:,0],
                    y_err=y_true_err[:,0],
                    color=self.colors[0],
                    linestyle="dashed",
                ))
                lines.append(Line(
                    y=y_predict[:,0],
                    y_ref=y_true[:,0],
                    label="acceptance",
                    color=self.colors[0],
                ))
                lines.append(Line(
                    y=n_pythia / n_herwig if self.invert_reweighting else n_herwig / n_pythia,
                    y_ref=None,
                    color=self.colors[1],
                    linestyle="dashed",
                ))
                lines.append(Line(
                    y=binned_weights_predict,
                    y_ref=n_pythia / n_herwig if self.invert_reweighting else n_herwig / n_pythia,
                    label="mean weights",
                    color=self.colors[1],
                ))
                self.hist_plot(pp, lines, bins, obs, no_scale=True, yscale="log")

    def plot_weights(self, file: str):
        """
        Makes plots of the weights learned for Pythia vs Herwig.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            if self.bayesian:
                weights_pythia = self.weights[0][self.labels.squeeze()]
                weights_herwig = self.weights[0][~self.labels.squeeze()]
            else:
                weights_pythia = self.weights[self.labels.squeeze()]
                weights_herwig = self.weights[~self.labels.squeeze()]
            
            xlim_bins = [-5, 5]
            bins = np.logspace(*xlim_bins, 128)
            y_pythia, y_pythia_err = self.compute_hist_data(bins, weights_pythia, bayesian=False)
            y_herwig, y_herwig_err = self.compute_hist_data(bins, weights_herwig, bayesian=False)

            original_bins = xlim_bins.copy()
            '''
            while True and xlim_bins[0] < xlim_bins[1]:
                bins = np.logspace(*xlim_bins, 128)
                y_pythia, y_pythia_err = self.compute_hist_data(bins, weights_pythia, bayesian=False)
                y_herwig, y_herwig_err = self.compute_hist_data(bins, weights_herwig, bayesian=False)
                if len(y_pythia[y_pythia>0])>20 or len(y_herwig[y_herwig>0])>20: # making sure that I have at least 20 non-empty bins, otherwise zoom in (number of bins stays the same)
                    break
                else:
                    if xlim_bins[0] - 0.01 * original_bins[0] > -1e-4 or xlim_bins[1] - 0.01 * original_bins[1] < 1e-4:
                        break
                    else:
                        xlim_bins[0] -= 0.01 * original_bins[0]
                        xlim_bins[1] -= 0.01 * original_bins[1]
            '''
            lines = [
                        Line(
                            y=y_pythia,
                            y_err=None,
                            label="Pythia" if not self.pythia_only else "Pythia 1",
                            color=self.colors[2],
                        ),
                        Line(
                            y=y_herwig,
                            y_err=None,
                            label="Herwig" if not self.pythia_only else "Pythia 2",
                            color=self.colors[0],
                        ),
                    ]
            self.hist_weights_plot(pp, lines, bins, show_ratios=False, ylim=[1e-8, 2])

    def plot_reco(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes plots of truth and predicted classes for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data in zip(self.observables, self.bins, self.obs_reco):
                data_pythia = data[self.labels.squeeze()]
                data_herwig = data[~self.labels.squeeze()]
                weights = self.weights[..., self.labels.squeeze()] if not self.invert_reweighting else self.weights[..., ~self.labels.squeeze()]

                if self.bayesian:
                    weights_statistic, _, _ = binned_statistic(data, self.predictions[0].T, bins=bins) # this is the mean of the predicted predictions in each bin
                else:
                    weights_statistic, _, _ = binned_statistic(data, self.predictions.T, bins=bins) # this is the mean of the predicted predictions in each bin
                weights_truth, _, _ = binned_statistic(data, ~self.labels.T, bins=bins) # this is the mean of the labels in each bin

                y_pythia, y_pythia_err = self.compute_hist_data(bins, data_pythia, bayesian=False)
                y_herwig, y_herwig_err = self.compute_hist_data(bins, data_herwig, bayesian=False)
                y_reweight, y_reweight_err = self.compute_hist_data(bins, data_pythia, bayesian=self.bayesian, weights=weights) if not self.invert_reweighting else self.compute_hist_data(bins, data_herwig, bayesian=self.bayesian, weights=weights)
                lines = [
                    Line(
                        y=y_pythia,
                        y_err=y_pythia_err,
                        y_ref=None if not self.pythia_only else y_herwig,
                        label="Pythia Reco" if not self.pythia_only else "Pythia Reco 1",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_herwig,
                        y_err=y_herwig_err,
                        label="Herwig Reco" if not self.pythia_only else "Pythia Reco 2",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_reweight,
                        y_err=y_reweight_err,
                        y_ref=y_herwig if not self.invert_reweighting else y_pythia,
                        label="Omnifold" if not self.bayesian else "bOmnifold",
                        color=self.colors[1],
                    ),
                    Line(
                        y=y_herwig / weights_statistic[0] * (1- weights_statistic[0]),
                        y_err=y_reweight_err,
                        y_ref= y_herwig if not self.invert_reweighting else y_pythia,
                        label="LH ratio x Herwig",
                        color=self.colors[4],
                        linestyle="dashed"
                    ),
                ]
                metrics = [obs.reco_emd_mean,
                           obs.reco_emd_std,
                           obs.reco_triangle_mean,
                           obs.reco_triangle_std] if self.show_metrics else None
                
                self.hist_plot(pp, lines, bins, obs, metrics=metrics)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_hard(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes plots of truth and predicted classes for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data in zip(self.observables, self.bins, self.obs_hard):
                data_pythia = data[self.labels.squeeze()]
                data_herwig = data[~self.labels.squeeze()]
                weights = self.weights[..., self.labels.squeeze()] if not self.invert_reweighting else self.weights[..., ~self.labels.squeeze()]

                y_pythia, y_pythia_err = self.compute_hist_data(bins, data_pythia, bayesian=False)
                y_herwig, y_herwig_err = self.compute_hist_data(bins, data_herwig, bayesian=False)
                y_reweight, y_reweight_err = self.compute_hist_data(bins, data_pythia, bayesian=self.bayesian, weights=weights) if not self.invert_reweighting else self.compute_hist_data(bins, data_herwig, bayesian=self.bayesian, weights=weights)

                lines = [
                    Line(
                        y=y_pythia,
                        y_err=y_pythia_err,
                        y_ref=None if not self.pythia_only else y_herwig,
                        label="Pythia Hard" if not self.pythia_only else "Pythia Hard 1",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_herwig,
                        y_err=y_herwig_err,
                        label="Herwig Hard" if not self.pythia_only else "Pythia Hard 2",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_reweight,
                        y_err=y_reweight_err,
                        y_ref=y_herwig if not self.invert_reweighting else y_pythia,
                        label="Omnifold" if not self.bayesian else "bOmnifold",
                        color=self.colors[1],
                    ),
                ]
                metrics = [obs.hard_emd_mean,
                           obs.hard_emd_std,
                           obs.hard_triangle_mean,
                           obs.hard_triangle_std] if self.show_metrics else None
                self.hist_plot(pp, lines, bins, obs, metrics=metrics)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_observables(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Output file name
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data_hard, data_reco in zip(
            self.observables, self.bins, self.obs_hard, self.obs_reco):

                
                data_reco = data_reco[~self.labels.squeeze()] if not self.invert_reweighting else data_reco[self.labels.squeeze()] # herwig reco OR pythia reco
                data_pythia_hard = data_hard[self.labels.squeeze()]
                data_herwig_hard = data_hard[~self.labels.squeeze()]
                weights = self.weights[..., self.labels.squeeze()] if not self.invert_reweighting else self.weights[..., ~self.labels.squeeze()]

                y_hard, y_hard_err = self.compute_hist_data(bins, data_herwig_hard, bayesian=False) if not self.invert_reweighting else self.compute_hist_data(bins, data_pythia_hard, bayesian=False)
                y_reco, y_reco_err = self.compute_hist_data(bins, data_reco, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data_pythia_hard, bayesian=self.bayesian, weights=weights) if not self.invert_reweighting else self.compute_hist_data(bins, data_herwig_hard, bayesian=self.bayesian, weights=weights)

                lines = [
                    Line(
                        y=y_reco,
                        y_err=y_reco_err,
                        label="Reco" if not self.pythia_only else "Pythia Reco 2",
                        color=self.colors[2],
                    ),
                    Line(
                        y=y_hard,
                        y_err=y_hard_err,
                        label="Hard" if not self.pythia_only else "Pythia Hard 2",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_hard,
                        label="Omnifold" if not self.bayesian else "bOmnifold",
                        color=self.colors[1],
                    ),
                ]
                metrics = [obs.emd_mean,
                           obs.emd_std,
                           obs.triangle_mean,
                           obs.triangle_std] if self.show_metrics else None
                self.hist_plot(pp, lines, bins, obs, metrics=metrics)

                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def compute_metrics(self):
        for obs, bins, data_hard, data_reco in zip(
                self.observables, self.bins, self.obs_hard, self.obs_reco):

            hard_ref = data_hard[~self.labels.squeeze()] if not self.invert_reweighting else data_hard[self.labels.squeeze()]
            hard_to_reweight = data_hard[self.labels.squeeze()] if not self.invert_reweighting else data_hard[~self.labels.squeeze()]
            weights = self.weights[..., self.labels.squeeze()] if not self.invert_reweighting else self.weights[..., ~self.labels.squeeze()]

            if not self.bayesian:
                emd_mean, emd_std = GetEMD(hard_ref, hard_to_reweight, nboot=10, weights_arr=weights)
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(hard_ref, hard_to_reweight, bins, nboot=10, weights=weights)
            else:
                emd_mean, emd_std = GetEMD(hard_ref, hard_to_reweight, nboot=10, weights_arr=weights[0])
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(hard_ref, hard_to_reweight, bins,
                                                                              nboot=10, weights=weights[0])
            #else:
            #    emd = []
            #    triangle_dist = []
            #    for weight_sample in weights:
            #        emd_sample, _ = GetEMD(data_hard_herwig, data_hard_pythia, nboot=1, weights_arr=weight_sample)
            #        triangle_dist_sample, _ = get_triangle_distance(data_hard_herwig, data_hard_pythia, bins, nboot=1, weights=weight_sample)
            #        emd.append(emd_sample)
            #        triangle_dist.append(triangle_dist_sample)
            #    emd_mean = np.array(emd).mean()
            #    emd_std = np.array(emd).std()
            #    triangle_dist_mean = np.array(triangle_dist).mean()
            #    triangle_dist_std = np.array(triangle_dist).std()

            obs.emd_mean = round(emd_mean, 4)
            obs.emd_std = round(emd_std, 5)
            obs.triangle_mean = round(triangle_dist_mean, 4)
            obs.triangle_std = round(triangle_dist_std, 5)

    def compute_metrics_reco(self):
        for obs, bins, data in zip(self.observables, self.bins, self.obs_reco):

            reco_ref = data[~self.labels.squeeze()] if not self.invert_reweighting else data[self.labels.squeeze()]
            reco_to_reweight = data[self.labels.squeeze()] if not self.invert_reweighting else data[~self.labels.squeeze()]
            weights = self.weights[..., self.labels.squeeze()] if not self.invert_reweighting else self.weights[..., ~self.labels.squeeze()]

            if not self.bayesian:
                emd_mean, emd_std = GetEMD(reco_ref, reco_to_reweight, nboot=10, weights_arr=weights)
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(reco_ref, reco_to_reweight, bins, nboot=10, weights=weights)
            else:
                emd_mean, emd_std = GetEMD(reco_ref, reco_to_reweight, nboot=10, weights_arr=weights[0])
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(reco_ref, reco_to_reweight, bins,
                                                                              nboot=10, weights=weights[0])

            obs.reco_emd_mean = round(emd_mean, 4)
            obs.reco_emd_std = round(emd_std, 5)
            obs.reco_triangle_mean = round(triangle_dist_mean, 4)
            obs.reco_triangle_std = round(triangle_dist_std, 5)

    def compute_metrics_hard(self):
        for obs, bins, data in zip(self.observables, self.bins, self.obs_hard):

            hard_ref = data[~self.labels.squeeze()] if not self.invert_reweighting else data[self.labels.squeeze()]
            hard_to_reweight = data[self.labels.squeeze()] if not self.invert_reweighting else data[~self.labels.squeeze()]
            weights = self.weights[..., self.labels.squeeze()] if not self.invert_reweighting else self.weights[..., ~self.labels.squeeze()]

            if not self.bayesian:
                emd_mean, emd_std = GetEMD(hard_ref, hard_to_reweight, nboot=10, weights_arr=weights)
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(hard_ref, hard_to_reweight, bins, nboot=10, weights=weights)
            else:
                emd_mean, emd_std = GetEMD(hard_ref, hard_to_reweight, nboot=10, weights_arr=weights[0])
                triangle_dist_mean, triangle_dist_std = get_triangle_distance(hard_ref, hard_to_reweight, bins,
                                                                              nboot=10, weights=weights[0])

            obs.hard_emd_mean = round(emd_mean, 4)
            obs.hard_emd_std = round(emd_std, 5)
            obs.hard_triangle_mean = round(triangle_dist_mean, 4)
            obs.hard_triangle_std = round(triangle_dist_std, 5)