from collections import UserDict
from contextlib import contextmanager
from pathlib import Path
import random
from typing import Any, Literal, Optional, TypedDict, Union

import h5py
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm
import yaml
from torch.utils.data import Dataset
from warnings import warn
from src import DATA_DIR
from src.utils import pose


class DatasetInfoEntry(UserDict):
    @property
    def data_csv_path(self):
        return self["data_csv_path"]

    @property
    def pixel_mean(self):
        return self["pixel_mean"]

    @property
    def pixel_std(self):
        return self["pixel_std"]


_dataset_info = OmegaConf.to_object(OmegaConf.load(DATA_DIR / "datasets.yaml"))
DATASET_INFO: dict[str, DatasetInfoEntry] = {
    name: DatasetInfoEntry(value) for name, value in _dataset_info.items()
}  # type:ignore


class SweepsDatasetItem(TypedDict):
    img: Any
    images: Any  # alias of img
    original_image_shape: Any
    tracking: Any
    calibration: Any
    spacing: Any
    dimensions: Any
    sweep_id: str
    raw_sweep_path: str
    start_idx: int
    stop_idx: int
    _extra_sequence_keys: list[str]


class SweepsDataset(Dataset):
    def __init__(
        self,
        name: Optional[str] = None,
        metadata_csv_path: Optional[Union[list[str], str]] = None,
        metadata_table: Optional[pd.DataFrame] = None,
        split="train",
        transform=None,
        subsequence_length: Optional[int] = None,
        subsequence_samples_per_scan: Literal["one", "all"] = "one",
        limit_samples: Optional[int] = None,
        limit_scans: Optional[int] = None,
        mode="h5_dynamic_load",
        drop_keys=[],
        original_image_shape=None,
        auto_convert_to_increasing_z=False,
        sequence_keys=["images", "tracking"],
        features_cache_map={},
    ):
        self.transform = transform
        self.subsequence_length = subsequence_length
        self.subsequence_samples_per_scan = subsequence_samples_per_scan
        self.limit_samples = limit_samples
        self.limit_scans = limit_scans
        self.mode = mode
        self.drop_keys = drop_keys
        self.original_image_shape = original_image_shape
        self.auto_convert_to_increasing_z = auto_convert_to_increasing_z
        self.sequence_keys = sequence_keys
        self.features_cache_map = features_cache_map

        # Getting the metadata table
        if metadata_table is not None:
            self.metadata = metadata_table
        elif metadata_csv_path is not None:
            if isinstance(metadata_csv_path, str):
                self.metadata = pd.read_csv(metadata_csv_path, index_col=0)
            else:
                self.metadata = pd.read_csv(metadata_csv_path[0], index_col=0)
                for i in range(len(metadata_csv_path) - 1):
                    new = pd.read_csv(metadata_csv_path[i + 1])
                    self.metadata = pd.concat([self.metadata, new])
        elif name is not None:
            metadata_csv_path = DATASET_INFO[name].data_csv_path
            self.metadata = pd.read_csv(metadata_csv_path, index_col=0)
        else:
            raise ValueError(
                f"One of name, metadata_csv_path or metadata_table should be specified."
            )

        # filtering the rows to get the correct split
        if split != "all":
            self.metadata = self.metadata.loc[self.metadata["split"] == split]

        if self.limit_scans is not None:
            self.metadata = self.metadata.iloc[: self.limit_scans]

        assert self.metadata['sweep_id'].is_unique, "Sweep IDs are not unique"

        self.data = {}
        self.sweep_ids = []
        for i, row in tqdm(self.metadata.iterrows(), total=self.metadata.shape[0], desc="Loading dataset"):
            data_i = {}
            sweep_id = row['sweep_id']
            path = row['processed_sweep_path'] 
            N = self._get_number_of_frames(path)
            if N == -1: 
                continue
            data_i['num_frames'] = N
            data_i['path'] = path
            data_i['metadata'] = row.to_dict()
            self.data[sweep_id] = data_i
            self.sweep_ids.append(sweep_id)

        self._h5_handles = {}
        self._indices = []

        # indexing - we need to go through each sequence
        # and determine how long is the sequence, how many
        # subsequences to grab, where do they start

        for sweep_id in self.sweep_ids:
            N = self.data[sweep_id]['num_frames']
            if subsequence_length is None:
                self._indices.append((sweep_id, None))
            else:
                if subsequence_samples_per_scan == "all":
                    for j in range(N + 1 - subsequence_length):
                        self._indices.append((sweep_id, j))
                else:
                    if N > subsequence_length:
                        self._indices.append((sweep_id, None))

    def __len__(self):
        if self.limit_samples is not None:
            return self.limit_samples
        return len(self._indices)

    def _get_number_of_frames(self, path):
        if self.mode == "h5":
            try:
                h5_handle = h5py.File(path)
                N = len(h5_handle["images"])
                return N
            except Exception as e:
                return -1
        elif self.mode == "h5_dynamic_load":
            try:
                with h5py.File(path) as f:
                    N = len(f["images"])
                return N
            except Exception as e:
                return -1
        else:
            raise ValueError()

    @contextmanager
    def _get_h5_handle(self, sweep_id: str):
        if self.mode == "h5_dynamic_load":
            f = h5py.File(self.data[sweep_id]['path'])
            yield f
            f.close()
        else:
            if "sweep_id" in self._h5_handles:
                yield self._h5_handles[sweep_id]
            else:
                self._h5_handles[sweep_id] = h5py.File(self.data[sweep_id]['path'])
                yield self._h5_handles[sweep_id]

    def _get_data_dict(
        self, sweep_id: str, start_idx, stop_idx
    ):

        out = {}

        with self._get_h5_handle(sweep_id) as f:
            for k, v in f.items():
                if k not in self.drop_keys:
                    if k in self.sequence_keys:
                        out[k] = v[start_idx:stop_idx]
                    else:
                        out[k] = v[:]

        for feature_name, feature_cache_path in self.features_cache_map.items():
            if feature_name in self.drop_keys: 
                continue 

            with h5py.File(feature_cache_path) as f:
                if feature_name in self.sequence_keys: 
                    out[feature_name] = f[sweep_id][start_idx:stop_idx]
                else: 
                    out[feature_name] = f[sweep_id][:]

        return out

    def _load_raw_data(
        self, sweep_id: str, start_idx: int, stop_idx: int
    ):
        data_dict = self._get_data_dict(
            sweep_id, start_idx, stop_idx
        )
        out = data_dict.copy()

        if "images" in data_dict:
            out["original_image_shape"] = data_dict["images"].shape[-2:]
        else:
            out["original_image_shape"] = self.original_image_shape

        image_keys = [k for k in data_dict.keys() if k.startswith("images")]
        for key in image_keys:
            out[key] = data_dict[key]
        out["tracking"] = data_dict["tracking"]
        out["calibration"] = data_dict["pixel_to_image"]
        out["spacing"] = data_dict["spacing"]
        out["dimensions"] = data_dict["dimensions"]
        out["_extra_sequence_keys"] = [
            k for k in out.keys() if k.startswith("images") and not k == "images"
        ]
        out["_extra_h5_keys"] = []

        # check if Z is increasing
        if self.auto_convert_to_increasing_z:
            tracking = out["tracking"]
            global_tracking = pose.invert_pose_matrix(tracking[0])[None, ...] @ tracking
            # tracking = global_tracking
            trackings_vector = pose.matrix_to_pose_vector(global_tracking)
            delta_z = trackings_vector[-1, 2] - trackings_vector[0, 2]
            if delta_z < 0:
                for key in self.sequence_keys:
                    out[key] = np.flip(out[key], 0).copy()

        return out

    def __getitem__(self, idx) -> SweepsDatasetItem:
        if self.limit_samples is not None and idx >= self.limit_samples:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of length {self.limit_samples}"
            )

        out = {}

        sweep_id, j = self._indices[idx]

        N = self.data[sweep_id]['num_frames']
        if self.subsequence_length is None:
            start_idx = 0
            stop_idx = N
        elif self.subsequence_samples_per_scan == "all":
            start_idx = j
            stop_idx = j + self.subsequence_length
        elif self.subsequence_samples_per_scan == "one":
            start_idx = random.randint(0, N - self.subsequence_length)
            stop_idx = start_idx + self.subsequence_length
        else:
            raise ValueError("Invalid subsequence_samples_per_scan")

        out["start_idx"] = start_idx
        out["stop_idx"] = stop_idx

        metadata = self.data[sweep_id]['metadata']
        # out.update(metadata)
        out["sweep_id"] = metadata["sweep_id"]
        sweep_id = out["sweep_id"]

        out.update(
            self._load_raw_data(
                sweep_id, start_idx, stop_idx
            )
        )

        if self.transform:
            out = self.transform(out)

        return out

    def compute_statistics(self):

        data = []

        for item in tqdm(self):
            data_i = {}
            data_i["scan_id"] = item["sweep_id"]

            tracking = item["tracking"]
            glob, loc = pose.get_global_and_relative_gt_trackings(tracking)
            xyz = glob[:, :3]
            total_displacement = np.linalg.norm(xyz[-1] - xyz[0], ord=2).item()
            data_i["total_displacement"] = total_displacement
            data_i["scan_len"] = len(glob)

            data.append(data_i)

        return pd.DataFrame(data)


class SweepsDatasetWithAdditionalCachedData(SweepsDataset):
    def __init__(
        self,
        *args,
        features_paths={},
        cache=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.features_paths = features_paths
        self.cache = cache
        self._cache = {}

    @contextmanager
    def load(self, path):
        try:
            f = h5py.File(path)
            yield f
        except:
            f = None
        finally:
            if f:
                f.close()

    def _load_raw_data(self, scan_idx, sweep_id, **kwargs):
        data = super()._load_raw_data(scan_idx, sweep_id, **kwargs)

        for name, path in self.features_paths.items():
            if self.cache:
                if (path, sweep_id) in self._cache:
                    data[name] = self._cache[(path, sweep_id)]
                else:
                    with self.load(path) as f:
                        data[name] = f[sweep_id][:]
                    self._cache[(path, sweep_id)] = data[name]
            else:
                with self.load(path) as f:
                    data[name] = f[sweep_id][:]
            data["_extra_sequence_keys"].append(name)

        return data


if __name__ == "__main__":
    import os

    default_csv_path = os.environ.get(
        "SINGLE_SCAN_CSV_PATH", "data/metadata/single_scan.csv"
    )
    dataset = SweepsDataset(default_csv_path, "train")
    print(dataset[0])
