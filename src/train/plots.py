import warnings
from dataclasses import dataclass
from typing import Optional
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import binned_statistic
from matplotlib.backends.backend_pdf import PdfPages
import torch

from ..processes.observables import Observable


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
        x_test: torch.Tensor,
        x_gen_single: torch.Tensor,
        x_gen_dist: torch.Tensor,
        event_type: Optional[torch.Tensor],
        bayesian: bool = False
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

        self.obs_test = []
        self.obs_gen_single = []
        self.obs_gen_dist = []
        self.bins = []
        for obs in observables:
            o_test = obs.compute(x_test, event_type)
            self.obs_test.append(o_test.cpu().numpy())
            self.obs_gen_single.append(
                obs.compute(x_gen_single, event_type).cpu().numpy()
            )
            self.obs_gen_dist.append(obs.compute(x_gen_dist, event_type).cpu().numpy())
            self.bins.append(obs.bins(o_test).cpu().numpy())

        self.bayesian = bayesian

        plt.rc("font", family="serif", size=16)
        plt.rc("axes", titlesize="medium")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("text", usetex=True)
        self.colors = [f"C{i}" for i in range(10)]

    def plot_losses(self, file: str):
        """
        Makes plots of the losses (total loss and if bayesian, BCE loss and KL loss
        separately) and learning rate as a function of the epoch.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            self.plot_single_loss(
                pp,
                "loss",
                (self.losses["train_loss"], self.losses["val_loss"]),
                ("train", "val"),
            )
            if self.bayesian:
                self.plot_single_loss(
                    pp,
                    "INN loss",
                    (self.losses["train_inn_loss"], self.losses["val_inn_loss"]),
                    ("train", "val"),
                )
                self.plot_single_loss(
                    pp,
                    "KL loss",
                    (self.losses["train_kl_loss"], self.losses["val_kl_loss"]),
                    ("train", "val"),
                )
            self.plot_single_loss(
                pp, "learning rate", (self.losses["lr"],), (None,), "log"
            )

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
            epochs = np.arange(6, len(curve) + 6)
            ax.plot(epochs[5:], curve[5:], label=label)
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
            for obs, bins, data_true, data_gen in zip(
                self.observables, self.bins, self.obs_test, self.obs_gen_single
            ):
                y_true, y_true_err = self.compute_hist_data(bins, data_true, bayesian=False)
                y_gen, y_gen_err = self.compute_hist_data(bins, data_gen, bayesian=self.bayesian)
                lines = [
                    Line(
                        y=y_true,
                        y_err=y_true_err,
                        label="Truth",
                        color=self.colors[0],
                    ),
                    Line(
                        y=y_gen,
                        y_err=y_gen_err,
                        y_ref=y_true,
                        label="Gen",
                        color=self.colors[1],
                    ),
                ]
                self.hist_plot(pp, lines, bins, obs)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})

        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_calibration(self, file: str):
        """
        Makes calibration plots for all observables.

        Args:
            file: Output file name
        """

        with PdfPages(file) as pp:
            for obs, data_true, data_gen in zip(
                self.observables, self.obs_test, self.obs_gen_dist
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
                        alpha=0.3 if self.bayesian else 1,
                    )

                plt.plot([0, 1], [0, 1], color="k", linestyle=":")
                plt.xlabel(f"quantile ${{{obs.tex_label}}}$")
                plt.ylabel("fraction of events")
                plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
                plt.close()

    def plot_pulls(self, file: str):
        """
        Plots pulls for all observables.
        Args:
            file: Output file name
        """

        bins = np.linspace(-5, 5, 40)
        with PdfPages(file) as pp:
            for obs, data_true, data_gen in zip(
                self.observables, self.obs_test, self.obs_gen_dist
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

    def plot_single_events(self, file: str):
        """
        Plots single event distributions for all observables.
        Args:
            file: Output file name
        """
        n_events = 5

        with PdfPages(file) as pp:
            for obs, bins, data_true, data_gen in zip(
                self.observables, self.bins, self.obs_test, self.obs_gen_dist
            ):
                for i in range(5):
                    x_true = data_gen
                    #TODO: the commented out changes break non-bayesian single event plots. please check
                    #y_gen, y_gen_err = self.compute_hist_data(bins, data_gen[..., :, i])
                    y_gen, y_gen_err = self.compute_hist_data(bins, data_gen[..., i, :])
                    lines = [
                        Line(
                            y=data_true[i],
                            label="Truth",
                            color=self.colors[0],
                            vline=True,
                        ),
                        Line(
                            y=y_gen,
                            y_ref=None,
                            label="Gen",
                            color=self.colors[1],
                        ),
                    ]
                    self.hist_plot(
                        pp, lines, bins, obs, title=f"Event {i}", show_ratios=False
                    )

    def compute_hist_data(self, bins: np.ndarray, data: np.ndarray, bayesian=False):
        if bayesian:
            hists = np.stack(
                [np.histogram(d, bins=bins, density=True)[0] for d in data], axis=0
            )
            y = np.median(hists, axis=0)
            y_err = np.stack(
                (np.quantile(hists, 0.159, axis=0), np.quantile(hists, 0.841, axis=0)),
                axis=0,
            )
        else:
            y, _ = np.histogram(data, bins=bins, density=False)
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
        no_scale: bool = False,
        yscale: Optional[str] = None,
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

            n_panels = 1 + int(show_ratios)
            fig, axs = plt.subplots(
                n_panels,
                1,
                sharex=True,
                figsize=(6, 4.5),
                gridspec_kw={"height_ratios": (4, 1, 1)[:n_panels], "hspace": 0.00},
            )
            if n_panels == 1:
                axs = [axs]

            for line in lines:
                if line.vline:
                    axs[0].axvline(line.y, label=line.label, color=line.color)
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
            axs[0].set_yscale(observable.yscale if yscale is None else yscale)
            if title is not None:
                self.corner_text(axs[0], title, "left", "top")

            if show_ratios:
                axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Truth}}$")
                axs[1].set_yticks([0.8, 1, 1.2])
                axs[1].set_ylim([0.75, 1.25])
                axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
                axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
                axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)

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
