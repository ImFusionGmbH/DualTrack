from multiprocessing.util import get_temp_dir
import os
from pathlib import Path
from typing import Optional
from attr import dataclass
import h5py
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from scipy import stats
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class FormattingInfo:
    column_remap: dict
    keep_columns: list


MICCAI2025_METRICS = {
    "ddf/5pt-avg_global_displacement_error": "GPE",
    "ddf/5pt-avg_local_displacement_error": "LPE",
    "drift/final_drift_rate": "FDR",
    "drift/max_drift": "Max. Drift",
}


class ResultsAnalyzer:
    def __init__(self):
        self.method_dir_mapping = {}
        self.metric_mapping = {}
        self.datasets = []

    def set_datasets(self, datasets):
        self.datasets = datasets

    def load_method_mapping(self, path):
        self.method_dir_mapping = OmegaConf.load(path)

    def set_metric_mapping(self, mapping):
        self.metric_mapping = mapping

    @property
    def methods(self):
        return list(self.method_dir_mapping.keys())

    @property
    def methods_with_results(self):
        return [method for method in self.methods if self.method_dir_mapping[method]]

    @property
    def metrics(self):
        return list(self.metric_mapping.values())

    def _get_test_dir(self, method, dataset) -> Path:
        return Path(self.method_dir_mapping[method]) / "test" / dataset

    def _get_sweep_dir(self, method, dataset, sweep_id) -> Path:
        return self._get_test_dir(method, dataset) / "scans" / sweep_id

    def get_errors_images(self, method, dataset, sweep_id):
        sweep_dir = self._get_sweep_dir(method, dataset, sweep_id)
        return Image.open(str(sweep_dir / "errors-global_example.png")), Image.open(
            str(sweep_dir / "errors-local_example.png")
        )

    def get_exported_predictions(self, method, dataset, sweep_id):
        predictions_path = self._get_sweep_dir(method, dataset, sweep_id) / 'export.h5'
        if not predictions_path.exists():
            raise ValueError("")
        with h5py.File(str(predictions_path)) as f:
            outputs = {
                k: v[:] for k, v in f.items()
            }
        return outputs

    def get_results_dataframe(self, method, dataset):
        if not self.method_dir_mapping[method]:
            return None
        dir = Path(self.method_dir_mapping[method]) / "test" / dataset
        if "full_metrics.csv" in os.listdir(dir):
            path = dir / "full_metrics.csv"
        elif "metrics.csv" in os.listdir(dir):
            path = dir / "metrics.csv"
        else:
            raise RuntimeError()

        return self.apply_formatting(pd.read_csv(path))

    def get_metric_values_by_method_dataframe(self, metric, dataset):
        df_by_method = {
            method: (df := self.get_results_dataframe(method, dataset))
            for method in self.methods_with_results
        }

        cols = []
        for name, df in df_by_method.items():
            cols.append(df[["sweep_id", metric]].rename(columns={metric: name}))

        combined_df = None
        for col in cols:
            if combined_df is None:
                combined_df = col
            else:
                combined_df = combined_df.merge(col, on="sweep_id", how="inner")

        return combined_df

    def create_wilcoxon_significance_report(
        self, metric, method1, method2, dataset, verbose=True, figures=False
    ):
        df = self.get_metric_values_by_method_dataframe(metric, dataset)
        if df is None:
            raise ValueError()

        method_A = df[method1]
        method_B = df[method2]
        differences = method_A - method_B

        stat, p_value = stats.wilcoxon(method_A, method_B)

        # Z-score approximation for Wilcoxon
        z = (stat - (len(differences) * (len(differences) + 1) / 4)) / np.sqrt(
            len(differences) * (len(differences) + 1) * (2 * len(differences) + 1) / 24
        )

        r = z / np.sqrt(len(differences))

        if verbose:
            print(f"Wilcoxon test statistic: {stat:.3f}, p-value: {p_value:.4e}")
            print(f"Effect size (r): {r:.4f}")

        if figures:
            differences = method_A - method_B  # Replace with your variables
            sns.histplot(differences, kde=True)
            plt.axvline(0, color="red", linestyle="--", label="No Difference")
            plt.legend()
            plt.title("Distribution of Paired Differences")
            plt.show()

        return stat, p_value, r

    def apply_formatting(self, df):
        df = df.rename(columns=self.metric_mapping)
        return df

    def _find_full_metrics_path(self, dir):
        if "full_metrics.csv" in os.listdir(dir):
            path = dir / "full_metrics.csv"
        elif "metrics.csv" in os.listdir(dir):
            path = dir / "metrics.csv"
        else:
            raise ValueError()
        return path

    def get_full_results_dataframe(self, dataset=None):
        dataset = dataset or self.datasets[0]

        dataframes = {}
        for method, dir in self.method_dir_mapping.items():
            if not dir:
                continue
            dir = Path(dir) / "test" / dataset
            dataframes[method] = self.apply_formatting(
                pd.read_csv(self._find_full_metrics_path(dir))
            )

        combined_df = pd.concat(
            [df.assign(method=key) for key, df in dataframes.items()], ignore_index=True
        )

        return combined_df


def get_accessor():
    ra = ResultsAnalyzer()
    ra.load_method_mapping(
        "experiments/miccai2024_results.yaml"
    )
    ra.set_datasets(["tus-rec", "tus-rec-val"])
    ra.set_metric_mapping(MICCAI2025_METRICS)
    return ra
