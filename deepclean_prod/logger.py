
import os
import logging

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

# default plotting style
plt.style.use('seaborn-v0_8-colorblind')
mpl.rc('font', size=15)
mpl.rc('figure', figsize=(8, 5))

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


class Logger:

    def __init__(self, outdir, metrics):
        self.data_subdir = outdir
        self.metrics = dict([(m, {'steps': [], 'epochs': [],
                                  'train': [], 'test': []}) for m in metrics])

    def update_metric(self, train_metric, test_metric, name, epoch, n_batch, num_batches):
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.metrics[name]['train'].append(train_metric)
        self.metrics[name]['test'].append(test_metric)
        self.metrics[name]['steps'].append(step)
        self.metrics[name]['epochs'].append(step/num_batches)

    def log_metric(self, name=None, max_epochs=None):
        out_dir = './{}/metrics'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True)

        # If name is not given, log all metrics
        if name is not None:
            train = self.metrics[name]['train']
            test = self.metrics[name]['test']
            steps = self.metrics[name]['steps']
            epochs = self.metrics[name]['epochs']

            array = np.vstack((steps, epochs, train, test)).T
            header = 'Step     Epochs    Train     Test'
            np.savetxt('{}/{}.dat'.format(out_dir, name),
                       array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
            self._plot_metric(name)
        else:
            for name in self.metrics.keys():
                train = self.metrics[name]['train']
                test = self.metrics[name]['test']
                steps = self.metrics[name]['steps']
                epochs = self.metrics[name]['epochs']

                array = np.vstack((steps, epochs, train, test)).T
                header = 'Step     Epochs    Train     Test'
                np.savetxt('{}/{}.dat'.format(out_dir, name),
                           array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
                self._plot_metric(name)

    def _plot_metric(self, name):
        train = self.metrics[name]['train']
        test = self.metrics[name]['test']
        steps = self.metrics[name]['steps']
        epochs = self.metrics[name]['epochs']

        out_dir = './{}/metrics'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True,)

        ###########################################################################
        # Plot 1:
        # Original DeepClean style
        #
        # Bottom x-axis: Iteration
        # Top x-axis: Epoch
        ###########################################################################

        fig, ax = plt.subplots()

        ax.plot(
            steps,
            train,
            label='Train',
        )

        ax.plot(
            steps,
            test,
            label='Test',
        )

        ax.set_xlabel(
            'Iteration'
        )

        ax.set_ylabel(
            name
        )

        ax.legend()

        # Secondary x-axis showing epoch.
        axtwin = ax.twiny()

        axtwin.plot(
            epochs,
            train,
            alpha=0.0,
        )

        axtwin.grid(
            False
        )

        axtwin.set_xlabel(
            'Epoch'
        )

        fig.tight_layout()

        fig.savefig(
            f'{out_dir}/{name}_org.png',
            dpi=300,
            bbox_inches='tight',
        )

        plt.close(
            fig
        )

        ###########################################################################
        # Plot 2:
        # Epoch-only version
        #
        # x-axis: Epoch
        ###########################################################################

        fig, ax = plt.subplots()

        ax.plot(
            epochs,
            train,
            label='Train',
            linewidth=2.0,
        )

        ax.plot(
            epochs,
            test,
            label='Test',
            linewidth=2.0,
        )

        ax.set_xlabel(
            'Epoch'
        )

        ax.set_ylabel(
            name
        )

        # ax.set_title(
        #     f'{name.capitalize()} vs Epoch'
        # )

        ax.grid(
            True,
            linestyle='--',
            alpha=0.5,
        )

        ax.legend()

        fig.tight_layout()

        ###########################################################################
        # Save publication-friendly outputs.
        ###########################################################################

        fig.savefig(
            f'{out_dir}/{name}_epoch.png',
            dpi=300,
            bbox_inches='tight',
        )

        # fig.savefig(
        #     f'{out_dir}/{name}_epoch.pdf',
        #     bbox_inches='tight',
        # )

        plt.close(
            fig
        )

    def plot_loss_components_epoch(
        self,
        csv_path=None,
    ):
        """
        Plot loss_components.csv against Epoch.

        Expected columns:
            epoch
            split
            raw_psd
            raw_mse
            raw_coh
            raw_tf
            weighted_psd
            weighted_mse
            weighted_coh
            weighted_tf
            total

        Output:
            metrics/loss_components_epoch.png
            metrics/loss_components_epoch.pdf
        """

        metrics_dir = os.path.join(
            self.data_subdir,
            "metrics",
        )

        if csv_path is None:
            csv_path = os.path.join(
                metrics_dir,
                "loss_components.csv",
            )

        if not os.path.isfile(csv_path):
            logger.warning(
                "Loss component CSV does not exist: %s",
                csv_path,
            )
            return

        with open(
            csv_path,
            "r",
            newline="",
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        if not rows:
            logger.warning(
                "Loss component CSV is empty: %s",
                csv_path,
            )
            return

        required_columns = {
            "epoch",
            "split",
            "raw_psd",
            "raw_mse",
            "raw_coh",
            "raw_tf",
            "weighted_psd",
            "weighted_mse",
            "weighted_coh",
            "weighted_tf",
            "total",
        }

        missing = (
            required_columns
            - set(rows[0].keys())
        )

        if missing:
            logger.warning(
                "Cannot plot loss components. "
                "Missing columns: %s",
                sorted(missing),
            )
            return

        ###########################################################################
        # Parse CSV
        ###########################################################################

        def normalize_split(value):
            value = str(value).strip().lower()

            if value in {
                "train",
                "training",
            }:
                return "train"

            if value in {
                "validation",
                "val",
                "test",
            }:
                return "validation"

            return value

        parsed = []

        for row in rows:

            try:
                parsed.append(
                    {
                        # CSV stores epoch from 0.
                        # Plot it as human-readable Epoch 1, 2, 3, ...
                        "epoch": (
                            int(
                                float(
                                    row["epoch"]
                                )
                            )
                            + 1
                        ),

                        "split": normalize_split(
                            row["split"]
                        ),

                        "raw_psd": float(
                            row["raw_psd"]
                        ),

                        "raw_mse": float(
                            row["raw_mse"]
                        ),

                        "raw_coh": float(
                            row["raw_coh"]
                        ),

                        "raw_tf": float(
                            row["raw_tf"]
                        ),

                        "weighted_psd": float(
                            row["weighted_psd"]
                        ),

                        "weighted_mse": float(
                            row["weighted_mse"]
                        ),

                        "weighted_coh": float(
                            row["weighted_coh"]
                        ),

                        "weighted_tf": float(
                            row["weighted_tf"]
                        ),

                        "total": float(
                            row["total"]
                        ),
                    }
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                logger.warning(
                    "Skipping malformed loss-component row: "
                    "%s (%s)",
                    row,
                    exc,
                )

        train_rows = sorted(
            [
                row
                for row in parsed
                if row["split"] == "train"
            ],
            key=lambda row: row["epoch"],
        )

        val_rows = sorted(
            [
                row
                for row in parsed
                if row["split"] == "validation"
            ],
            key=lambda row: row["epoch"],
        )

        if (
            not train_rows
            and not val_rows
        ):
            logger.warning(
                "No train/validation component rows found."
            )
            return

        ###########################################################################
        # Components
        ###########################################################################

        raw_components = [
            ("raw_psd", "PSD"),
            ("raw_mse", "MSE"),
            ("raw_coh", "COH"),
            ("raw_tf", "TF"),
        ]

        weighted_components = [
            ("weighted_psd", "PSD"),
            ("weighted_mse", "MSE"),
            ("weighted_coh", "COH"),
            ("weighted_tf", "TF"),
        ]

        def component_is_active(
            component,
        ):
            values = [
                abs(
                    row[component]
                )
                for row in parsed
            ]

            if not values:
                return False

            return (
                max(values)
                > 1e-15
            )

        ###########################################################################
        # Panel plotting helper
        ###########################################################################

        def plot_components(
            ax,
            subset,
            components,
            title,
            include_total=False,
        ):

            if not subset:
                ax.set_visible(
                    False
                )
                return

            epochs = [
                row["epoch"]
                for row in subset
            ]

            plotted = False

            for key, label in components:

                # Do not draw MSE / COH / TF if it is zero for the whole run.
                if not component_is_active(
                    key
                ):
                    continue

                values = [
                    row[key]
                    for row in subset
                ]

                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    markersize=3,
                    linewidth=1.7,
                    label=label,
                )

                plotted = True

            if include_total:

                total = [
                    row["total"]
                    for row in subset
                ]

                ax.plot(
                    epochs,
                    total,
                    marker="o",
                    markersize=3,
                    linewidth=2.2,
                    label="Total",
                )

                plotted = True

            ax.set_title(
                title
            )

            ax.set_xlabel(
                "Epoch"
            )

            ax.set_ylabel(
                "Loss"
            )

            ax.grid(
                True,
                linestyle="--",
                alpha=0.4,
            )

            if plotted:
                ax.legend(
                    fontsize=9
                )

        ###########################################################################
        # Create 2 x 2 figure
        ###########################################################################

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(
                13,
                9,
            ),
            sharex=True,
        )

        plot_components(
            axes[0, 0],
            train_rows,
            raw_components,
            "Raw Components - Train",
        )

        plot_components(
            axes[0, 1],
            val_rows,
            raw_components,
            "Raw Components - Validation",
        )

        plot_components(
            axes[1, 0],
            train_rows,
            weighted_components,
            "Weighted Components - Train",
            include_total=True,
        )

        plot_components(
            axes[1, 1],
            val_rows,
            weighted_components,
            "Weighted Components - Validation",
            include_total=True,
        )

        ###########################################################################
        # Mark the best validation-total epoch
        ###########################################################################

        if val_rows:

            best_row = min(
                val_rows,
                key=lambda row: row[
                    "total"
                ],
            )

            best_epoch = (
                best_row["epoch"]
            )

            best_total = (
                best_row["total"]
            )

            for ax in axes.flat:

                if not ax.get_visible():
                    continue

                ax.axvline(
                    best_epoch,
                    linestyle=":",
                    linewidth=1.3,
                    alpha=0.8,
                )

            fig.suptitle(
                (
                    "Loss Components vs Epoch\n"
                    f"Best validation total: "
                    f"Epoch {best_epoch}, "
                    f"loss={best_total:.6g}"
                ),
                fontsize=15,
            )

        else:

            fig.suptitle(
                "Loss Components vs Epoch",
                fontsize=15,
            )

        fig.tight_layout(
            rect=(
                0.0,
                0.0,
                1.0,
                0.94,
            )
        )

        ###########################################################################
        # Save
        ###########################################################################

        output_png = os.path.join(
            metrics_dir,
            "loss_components_epoch.png",
        )

        output_pdf = os.path.join(
            metrics_dir,
            "loss_components_epoch.pdf",
        )

        fig.savefig(
            output_png,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            output_pdf,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        logger.info(
            "Saved loss component plots: "
            "%s and %s",
            output_png,
            output_pdf,
        )

    def display_status(self, epoch, num_epochs, n_batch, num_batches,
                       train_metric, test_metric, name, show_epoch=True):
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()

        if show_epoch:
            logger.info('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
                epoch, num_epochs, n_batch, num_batches)
            )
        logger.info('Train {0:}: {1:.4e}, Test {0:}: {2:.4e}'.format(
            name, train_metric, test_metric))

    def save_model(self, model, epoch, n_batch=None):
        out_dir = './{}/models'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True)
        if n_batch is not None:
            torch.save(model.state_dict(),
                       '{}/epoch_{}_batch_{}'.format(out_dir, epoch, n_batch))
        else:
            torch.save(model.state_dict(),
                       '{}/epoch_{}'.format(out_dir, epoch))

    # Private Functionality
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch
