# %%
# train14_innovate_era5.py
#
# Cross-attention patch embedding variant of train14_small_era5.py.
#
# Key architectural change (inspired by ClimaX / Stormer):
#   PatchEmbed2D  →  CrossAttnPatchEmbed2D
#   PatchEmbed3D  →  CrossAttnPatchEmbed3D
#
# Design decisions (from user specification):
#   - Shared learnable query vector (one query per embed module, broadcast
#     across all spatial positions and all pressure-level queries)
#   - 3-D strategy: all C × Pl_patches tokens are flattened into the K/V set
#     at each spatial location; Pl_patches separate learnable query vectors
#     (one per output pressure-level slot) attend over that full K/V set
#   - 6 attention heads (head_dim = 192/6 = 32)
#   - Pre-norm: LayerNorm applied to K/V tokens BEFORE cross-attention
#   - Static masks loaded from individual .npy files:
#       aux_data/land_mask.npy, aux_data/soil_type.npy, aux_data/topography.npy
#     (falls back to constantMask24.npy if the individual files are absent)
#   - Training from scratch — no pretrained weight reuse
#
import os
# NCCL_DEBUG: WARN suppresses verbose channel-setup spam; set INFO only for deep NCCL debugging
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo,virbr0"  # ignore virtual/loopback interfaces
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=r".*torch.meshgrid.*")
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
import time
from tqdm import tqdm
import argparse
import re
import os
import torch
from torch import nn
import pandas as pd
import sys
from tqdm import tqdm
from collections import OrderedDict
from os import listdir
from os.path import join
import pickle
from datetime import datetime
from typing import Literal
from torchvision.transforms import Normalize, Compose
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta
from torch.utils.data import DataLoader
try:
    from timm.layers import trunc_normal_, DropPath
except ImportError:
    from timm.models.layers import trunc_normal_, DropPath  # older timm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def str2bool(v):
    import argparse
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Train PanguLite ERA5 monthly models (cross-attn embed variant)")
parser.add_argument("--num_epochs", default=10, type=int, help="train epoch number")
parser.add_argument('--launcher', default='pytorch', help='job launcher')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--dist', type=str2bool, default=False)
parser.add_argument('--backend', type=str, default='')
parser.add_argument('--data_dir', default='./data', type=str, help='path to data directory')
parser.add_argument('--aux_data_dir', default=None, type=str, help='path to aux_data directory; defaults to DATA_DIR/aux_data')
parser.add_argument('--horizon_hours', default=24, type=int, help='forecast lead time used for training targets')
parser.add_argument('--sample_stride_hours', default=6, type=int, help='training sample spacing; use 24 for one sample per day')
parser.add_argument('--batch_size', default=1, type=int, help='per-process batch size')
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers per process')
parser.add_argument('--prefetch_factor', default=2, type=int, help='DataLoader prefetch factor when workers are enabled')
parser.add_argument('--amp', type=str2bool, default=True, help='use CUDA automatic mixed precision')
parser.add_argument('--compile', type=str2bool, default=False, help='use torch.compile when available')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=3e-6, type=float, help='Adam weight decay')
parser.add_argument('--upper_loss_weight', default=1.0, type=float, help='relative weight for upper-air loss')
parser.add_argument('--surface_loss_weight', default=0.25, type=float, help='relative weight for surface loss')
parser.add_argument('--loss_type', choices=['weighted_l1', 'l1'], default='weighted_l1', help='training loss type')
parser.add_argument('--residual', type=str2bool, default=True, help='predict residual delta and add it to the input')
parser.add_argument('--output_dir', default='epochs_pangulite_innovate_era5', type=str, help='checkpoint output directory')
parser.add_argument('--log_dir', default='train_logs_pangulite_innovate_era5', type=str, help='training log directory')
parser.add_argument('--save_every', default=1, type=int, help='save checkpoint every N epochs; set 0 to save best only')
parser.add_argument('--val_every', default=1, type=int, help='run validation every N epochs; set 0 to skip validation')
parser.add_argument('--resume', action='store_true',
                    help='resume from pangu_lite_innovate_era5_latest.pth in --output_dir if it exists')
parser.add_argument('--early_stop_patience', default=0, type=int,
                    help='stop if val_score does not improve for this many validation epochs (0 = disabled)')
parser.add_argument('--train_start', default='1979-01-01', type=str, help='inclusive train start date')
parser.add_argument('--train_end', default='2018-12-31', type=str, help='inclusive train end date')
parser.add_argument('--valid_start', default='2019-01-01', type=str, help='inclusive validation start date')
parser.add_argument('--valid_end', default='2019-12-31', type=str, help='inclusive validation end date')
parser.add_argument('--test_start', default='2020-01-01', type=str, help='inclusive test start date')
parser.add_argument('--test_end', default='2024-12-31', type=str, help='inclusive test end date')
parser.add_argument('--dataset_cache_size', default=24, type=int, help='max open monthly surface/upper files cached per worker')
# Cross-attention embed hyper-parameters
parser.add_argument('--embed_heads', default=6, type=int,
                    help='number of attention heads in the cross-attention patch embedders (6 → head_dim=32)')
parser.add_argument('--embed_attn_drop', default=0.0, type=float, help='dropout in cross-attn embed layers')
parser.add_argument('--embed_checkpoint', type=str2bool, default=True,
                    help='use gradient checkpointing in cross-attn embed layers to save GPU memory (default: True)')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SURFACE_VARIABLES = ['msl', 'u10', 'v10', 't2m']
UPPER_VARIABLES = ['z', 'q', 't', 'u', 'v']
PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
STAT_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE_WEIGHTS = [1.50, 0.77, 0.66, 3.00]
UPPER_WEIGHTS = [3.00, 0.60, 1.50, 0.77, 0.54]
SURFACE_MONTHLY_RE = re.compile(r"^surface_(\d{4})_(\d{2})\.nc$")
UPPER_MONTHLY_RE = re.compile(r"^upper_air_(\d{4})_(\d{2})\.nc$")
EPSILON = 0.622

# ---------------------------------------------------------------------------
# Distributed helpers (unchanged from train14_small_era5.py)
# ---------------------------------------------------------------------------
def _pick_backend(backend_arg):
    if backend_arg:
        return backend_arg
    return 'nccl' if torch.cuda.is_available() else 'gloo'

NCCL_TIMEOUT_MINUTES = 60

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def _init_dist_pytorch(backend, **kwargs):
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise RuntimeError(
            'Distributed requested but RANK/WORLD_SIZE not set. '
            'Launch with torchrun (recommended) or set env vars manually.'
        )
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        torch.cuda.set_device(local_rank % num_gpus)
    import datetime
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=NCCL_TIMEOUT_MINUTES),
        **kwargs
    )

def _init_dist_slurm(backend, port=None):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass
    else:
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus) if num_gpus > 0 else '0'
    os.environ['RANK'] = str(proc_id)
    import datetime
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=NCCL_TIMEOUT_MINUTES),
    )

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def is_dist_ready():
    return dist.is_available() and dist.is_initialized()

def unwrap_model(model):
    model = model.module if hasattr(model, 'module') else model
    return model._orig_mod if hasattr(model, '_orig_mod') else model

# ---------------------------------------------------------------------------
# Normalisation helpers (unchanged)
# ---------------------------------------------------------------------------
def surface_transform(mean_path, std_path):
    surface_mean_npy = np.load(mean_path).astype(np.float32)
    surface_std_npy = np.load(std_path).astype(np.float32)
    mean_seq = [float(surface_mean_npy[i]) for i in range(len(SURFACE_VARIABLES))]
    std_seq = [float(surface_std_npy[i]) for i in range(len(SURFACE_VARIABLES))]
    channel_seq = list(SURFACE_VARIABLES)
    return Normalize(mean_seq, std_seq), channel_seq

def upper_air_transform(mean_path, std_path):
    upper_mean_npy = np.load(mean_path).astype(np.float32)
    upper_std_npy = np.load(std_path).astype(np.float32)
    pLevels = list(PANGU_LEVELS)
    variables = list(UPPER_VARIABLES)
    normalize = {}
    for pl in pLevels:
        pl_idx = STAT_LEVELS.index(pl)
        mean_seq = [float(upper_mean_npy[pl_idx, 0, 0, i]) for i in range(len(UPPER_VARIABLES))]
        std_seq = [float(upper_std_npy[pl_idx, 0, 0, i]) for i in range(len(UPPER_VARIABLES))]
        normalize[pl] = Normalize(mean_seq, std_seq)
    return normalize, variables, pLevels

def surface_inv_transform(mean_path, std_path):
    surface_mean_npy = np.load(mean_path).astype(np.float32)
    surface_std_npy = np.load(std_path).astype(np.float32)
    mean_seq = [float(surface_mean_npy[i]) for i in range(len(SURFACE_VARIABLES))]
    std_seq = [float(surface_std_npy[i]) for i in range(len(SURFACE_VARIABLES))]
    channel_seq = list(SURFACE_VARIABLES)
    invTrans = Compose([
        Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]),
        Normalize([-x for x in mean_seq], [1.] * len(std_seq))
    ])
    return invTrans, channel_seq

def upper_air_inv_transform(mean_path, std_path):
    upper_mean_npy = np.load(mean_path).astype(np.float32)
    upper_std_npy = np.load(std_path).astype(np.float32)
    pLevels = list(PANGU_LEVELS)
    variables = list(UPPER_VARIABLES)
    normalize = {}
    for pl in pLevels:
        pl_idx = STAT_LEVELS.index(pl)
        mean_seq = [float(upper_mean_npy[pl_idx, 0, 0, i]) for i in range(len(UPPER_VARIABLES))]
        std_seq = [float(upper_std_npy[pl_idx, 0, 0, i]) for i in range(len(UPPER_VARIABLES))]
        invTrans = Compose([
            Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]),
            Normalize([-x for x in mean_seq], [1.] * len(std_seq))
        ])
        normalize[pl] = invTrans
    return normalize, variables, pLevels

# ---------------------------------------------------------------------------
# Dataset (unchanged from train14_small_era5.py; only _load_constant_mask
# updated to support individual .npy mask files)
# ---------------------------------------------------------------------------
class ERA5DatasetFromFolder(Dataset):
    """
    Monthly ERA5 dataset.

    Static mask loading order (new):
      1. Try aux_data/land_mask.npy, aux_data/soil_type.npy, aux_data/topography.npy
         (individual files added by the user).
      2. Fall back to aux_data/constantMask24.npy (original combined file).
    """

    def __init__(
        self,
        dataset_dir,
        flag: Literal["train", "test", "valid"],
        lead_hours: int = 24,
        sample_stride_hours: int = 6,
        split_ranges=None,
        cache_size: int = 24,
        aux_data_dir: str = None,
        _prebuilt_samples=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.flag = flag
        self.lead_hours = lead_hours
        self.sample_stride_hours = sample_stride_hours
        self.cache_size = max(1, int(cache_size))
        self.aux_data_dir = aux_data_dir or join(dataset_dir, "aux_data")
        self._surface_cache = OrderedDict()
        self._upper_cache = OrderedDict()

        self.surface_dir = join(dataset_dir, "surface")
        self.upper_dir = join(dataset_dir, "upper")
        if not os.path.exists(self.surface_dir):
            raise FileNotFoundError(f"Surface directory not found at {self.surface_dir}")
        if not os.path.exists(self.upper_dir):
            raise FileNotFoundError(f"Upper-air directory not found at {self.upper_dir}")

        self.split_ranges = split_ranges or {
            "train": ("1979-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2019-12-31"),
            "test": ("2020-01-01", "2024-12-31"),
        }
        if flag not in self.split_ranges:
            raise ValueError(f"Invalid flag: {flag!r}. Must be 'train', 'valid', or 'test'.")
        self.range_start, self.range_end = self._parse_range(*self.split_ranges[flag])

        if _prebuilt_samples is not None:
            self.available_months = []
            self.samples = list(_prebuilt_samples)
        else:
            self.available_months = self._discover_months()
            self.samples = self._build_samples()
        self.date = np.array([x[0] for x in self.samples], dtype='datetime64[ns]')

        self.surface_transform, self.surface_variables = surface_transform(
            join(self.aux_data_dir, "surface_mean.npy"),
            join(self.aux_data_dir, "surface_std.npy")
        )
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(
            join(self.aux_data_dir, "upper_mean.npy"),
            join(self.aux_data_dir, "upper_std.npy")
        )

        self.land_mask, self.soil_type, self.topography = self._load_constant_mask()
        self.const_h = self._load_const_h()

    def __getitem__(self, index):
        input_time, target_time = self.samples[index]
        surface_t, upper_air_t = self._get_data(input_time)
        surface_t_1, upper_air_t_1 = self._get_data(target_time)
        if self.flag == "train":
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, torch.tensor([
            input_time.astype("datetime64[s]").astype(np.int64),
            target_time.astype("datetime64[s]").astype(np.int64),
        ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _parse_date_start(value):
        text = str(value).strip().replace(" ", "T")
        if "T" not in text:
            text += "T00:00:00"
        return np.datetime64(text, "s")

    @staticmethod
    def _parse_date_end(value):
        text = str(value).strip().replace(" ", "T")
        if "T" not in text:
            text += "T23:59:59"
        return np.datetime64(text, "s")

    @classmethod
    def _parse_range(cls, start, end):
        start_ts = cls._parse_date_start(start)
        end_ts = cls._parse_date_end(end)
        if end_ts < start_ts:
            raise ValueError(f"Invalid split range: {start} to {end}")
        return start_ts, end_ts

    @staticmethod
    def _year_month(timestamp):
        year = int(timestamp.astype("datetime64[Y]").astype(int)) + 1970
        month = int(timestamp.astype("datetime64[M]").astype(int)) % 12 + 1
        return year, month

    def _discover_months(self):
        surface_months = {}
        upper_months = {}
        for name in listdir(self.surface_dir):
            match = SURFACE_MONTHLY_RE.match(name)
            if match:
                surface_months[(int(match.group(1)), int(match.group(2)))] = join(self.surface_dir, name)
        for name in listdir(self.upper_dir):
            match = UPPER_MONTHLY_RE.match(name)
            if match:
                upper_months[(int(match.group(1)), int(match.group(2)))] = join(self.upper_dir, name)

        months = sorted(set(surface_months).intersection(upper_months))
        if not months:
            raise FileNotFoundError(
                f"No matching monthly surface_YYYY_MM.nc and upper_air_YYYY_MM.nc pairs found in {self.dataset_dir}"
            )
        return months

    def _build_samples(self):
        lead = np.timedelta64(int(self.lead_hours), 'h')
        available_times = []
        available_keys = set()
        months_scanned = 0
        ts_collected = 0

        for year, month in self.available_months:
            surface_path = join(self.surface_dir, f"surface_{year}_{month:02d}.nc")
            upper_path = join(self.upper_dir, f"upper_air_{year}_{month:02d}.nc")
            with xr.open_dataset(surface_path, decode_timedelta=False) as ds_surface:
                ds_surface = self._normalize_dataset_coords(ds_surface)
                surface_times = self._times_in_range(ds_surface)
            with xr.open_dataset(upper_path, decode_timedelta=False) as ds_upper:
                ds_upper = self._normalize_dataset_coords(ds_upper)
                upper_times = self._times_in_range(ds_upper)

            common_times = sorted(set(surface_times).intersection(upper_times))
            if common_times:
                months_scanned += 1
            for ts in common_times:
                key = int(ts.astype(np.int64))
                available_keys.add(key)
                ts_collected += 1
                if self._matches_stride(ts):
                    available_times.append(ts)

        samples = []
        for ts in sorted(available_times):
            target_ts = (ts + lead).astype('datetime64[s]')
            target_key = int(target_ts.astype(np.int64))
            if self.range_start <= target_ts <= self.range_end and target_key in available_keys:
                samples.append((ts.astype('datetime64[ns]'), target_ts.astype('datetime64[ns]')))

        if not samples:
            raise ValueError(
                f"No {self.flag} samples found in {self.dataset_dir}. "
                f"Range={self.range_start} to {self.range_end}, "
                f"months_with_times={months_scanned}, timestamps_collected={ts_collected}, "
                f"stride_candidates={len(available_times)}. "
                f"Check monthly file names, time coordinates, horizon_hours={self.lead_hours}, "
                f"and sample_stride_hours={self.sample_stride_hours}."
            )
        return samples

    def _times_in_range(self, ds):
        time_name = self._time_name(ds)
        times = np.atleast_1d(ds[time_name].values).astype('datetime64[s]')
        return [ts for ts in times if self.range_start <= ts <= self.range_end]

    def _matches_stride(self, timestamp):
        stride = int(self.sample_stride_hours)
        if stride <= 0:
            return True
        hour_index = int(timestamp.astype('datetime64[h]').astype(np.int64))
        return hour_index % stride == 0

    @staticmethod
    def _normalize_dataset_coords(ds):
        if "valid_time" in ds.variables and "time" not in ds.variables:
            ds = ds.rename({"valid_time": "time"})
        return ds

    @staticmethod
    def _time_name(ds):
        for name in ("time", "valid_time"):
            if name in ds.coords:
                return name
        raise KeyError("Could not find a time coordinate named 'time' or 'valid_time'")

    @staticmethod
    def _level_name(ds):
        for name in ("level", "isobaricInhPa", "pressure_level"):
            if name in ds.coords:
                return name
        raise KeyError("Could not find a pressure-level coordinate")

    def _cache_dataset(self, cache, key, path):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        ds = self._normalize_dataset_coords(xr.open_dataset(path, decode_timedelta=False))
        cache[key] = ds
        if len(cache) > self.cache_size:
            _, old_ds = cache.popitem(last=False)
            old_ds.close()
        return ds

    def _get_surface_ds(self, year, month):
        key = (year, month)
        path = join(self.surface_dir, f"surface_{year}_{month:02d}.nc")
        return self._cache_dataset(self._surface_cache, key, path)

    def _get_upper_ds(self, year, month):
        key = (year, month)
        path = join(self.upper_dir, f"upper_air_{year}_{month:02d}.nc")
        return self._cache_dataset(self._upper_cache, key, path)

    def close(self):
        for ds in list(self._surface_cache.values()) + list(self._upper_cache.values()):
            ds.close()
        self._surface_cache.clear()
        self._upper_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_surface_cache"] = OrderedDict()
        state["_upper_cache"] = OrderedDict()
        return state

    @staticmethod
    def _to_2d_array(value, var_name):
        arr = np.asarray(value, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected {var_name} to be 2D after squeeze, got shape {arr.shape}")
        return arr

    @staticmethod
    def _relative_humidity_to_specific_humidity(relative_humidity, temperature, pressure_hpa):
        rh = np.asarray(relative_humidity, dtype=np.float32)
        temp = np.asarray(temperature, dtype=np.float32)
        rh_max = np.nanmax(rh)
        rh_fraction = rh if rh_max <= 1.5 else rh / 100.0
        rh_fraction = np.clip(rh_fraction, 0.0, 1.0)
        temp_c = temp - 273.15
        saturation_vapor_pressure = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        vapor_pressure = rh_fraction * saturation_vapor_pressure
        pressure_pa = np.float32(pressure_hpa * 100.0)
        vapor_pressure = np.minimum(vapor_pressure, pressure_pa * 0.99)
        q = (EPSILON * vapor_pressure) / (pressure_pa - (1.0 - EPSILON) * vapor_pressure)
        return np.asarray(q, dtype=np.float32)

    def _upper_level_values(self, upper_sel, level_name, pressure_level):
        level_sel = upper_sel.sel({level_name: pressure_level})
        values = []
        temp_for_q = None
        if "q" not in level_sel and "r" in level_sel:
            if "t" not in level_sel:
                raise KeyError("Cannot convert ERA5 relative humidity r to q because temperature t is missing.")
            temp_for_q = self._to_2d_array(level_sel["t"].values, "t")

        for var in self.upper_air_variables:
            if var == "q" and "q" not in level_sel:
                if "r" not in level_sel:
                    raise KeyError("Upper-air file has neither q nor r for humidity.")
                rh = self._to_2d_array(level_sel["r"].values, "r")
                values.append(self._relative_humidity_to_specific_humidity(rh, temp_for_q, pressure_level))
            else:
                values.append(self._to_2d_array(level_sel[var].values, var))
        return np.stack(values, axis=0).astype(np.float32)

    def _get_data(self, date):
        year, month = self._year_month(date)
        date_ns = date.astype('datetime64[ns]')
        sel_kwargs = dict(method='nearest', tolerance=np.timedelta64(90, 'm'))

        surface_ds = self._get_surface_ds(year, month)
        surface_sel = surface_ds.sel({self._time_name(surface_ds): date_ns}, **sel_kwargs)
        surface_data = np.stack(
            [self._to_2d_array(surface_sel[x].values, x) for x in self.surface_variables],
            axis=0,
        )
        surface_data = torch.from_numpy(surface_data.astype(np.float32))
        surface_data = self.surface_transform(surface_data)

        upper_ds = self._get_upper_ds(year, month)
        upper_sel = upper_ds.sel({self._time_name(upper_ds): date_ns}, **sel_kwargs)
        level_name = self._level_name(upper_ds)
        upper_air_data = torch.stack([
            self.upper_air_transform[pl](
                torch.from_numpy(self._upper_level_values(upper_sel, level_name, pl))
            )
            for pl in self.upper_air_pLevels
        ], dim=1)
        return surface_data, upper_air_data

    # ------------------------------------------------------------------
    # Static mask loading — supports individual .npy files (new) with
    # fallback to the original combined constantMask24.npy
    # ------------------------------------------------------------------
    def _load_constant_mask(self):
        lm_path  = join(self.aux_data_dir, "land_mask.npy")
        st_path  = join(self.aux_data_dir, "soil_type.npy")
        topo_path = join(self.aux_data_dir, "topography.npy")

        if os.path.isfile(lm_path) and os.path.isfile(st_path) and os.path.isfile(topo_path):
            # ── New individual .npy files (preferred) ──────────────────
            land_mask  = np.load(lm_path).astype(np.float32)
            soil_type  = np.load(st_path).astype(np.float32)
            topography = np.load(topo_path).astype(np.float32)

            # Each file may be (H, W), (1, H, W), or (1, 1, H, W); squeeze to (H, W)
            land_mask  = torch.from_numpy(np.squeeze(land_mask) [:721, :])
            soil_type  = torch.from_numpy(np.squeeze(soil_type) [:721, :])
            topography = torch.from_numpy(np.squeeze(topography)[:721, :])
        else:
            # ── Legacy combined constantMask24.npy ────────────────────
            mask_path = join(self.aux_data_dir, "constantMask24.npy")
            mask = np.load(mask_path).astype(np.float32)
            mask = mask[0, :, :721, :]   # (3, 721, 1440)
            land_mask  = torch.from_numpy(mask[0])
            soil_type  = torch.from_numpy(mask[1])
            topography = torch.from_numpy(mask[2])

        return land_mask, soil_type, topography

    def get_constant_mask(self):
        return self.land_mask, self.soil_type, self.topography

    def _load_const_h(self):
        const_h_path = join(self.aux_data_dir, "Constant_17_output_0.npy")
        const_h = np.load(const_h_path).astype(np.float32)
        const_h = np.squeeze(const_h)
        if const_h.ndim != 3:
            raise ValueError(f"Expected Constant_17_output_0.npy to squeeze to 3 dims, got {const_h.shape}")
        const_h = const_h[:, :721, :]
        return torch.from_numpy(const_h).unsqueeze(0).unsqueeze(0)

    def get_const_h(self):
        return self.const_h

    def get_lat_lon(self):
        if not self.available_months:
            raise FileNotFoundError(f"No monthly surface files found in {self.surface_dir}")
        year, month = self.available_months[0]
        example = join(self.surface_dir, f"surface_{year}_{month:02d}.nc")
        with xr.open_dataset(example) as ds:
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            lat = ds[lat_name].data
            lon = ds[lon_name].data
        return lat, lon


# ---------------------------------------------------------------------------
# Cross-Attention Patch Embedders (the core innovation)
# ---------------------------------------------------------------------------

class CrossAttnPatchEmbed2D(nn.Module):
    """
    ClimaX / Stormer-style variable-aware patch embedding for 2-D surface fields.

    For each spatial patch position (h, w):
      1. Each of the C input channels contributes an independent linear projection
         of its p×p pixels → a D-dimensional token.  All C channels share the same
         weight matrix (parameter-efficient).
      2. A single shared learnable query vector attends (cross-attention, pre-norm)
         over those C key/value tokens, collapsing the variable dimension.
      3. The attended output is the final D-dim embedding for that location.

    Output shape matches PatchEmbed2D: (B, embed_dim, H//p, W//p)

    Design choices (per user specification):
      - Shared query:  one learnable query broadcast over all (h, w) positions
      - Pre-norm:      LayerNorm applied to K/V tokens before cross-attention
      - num_heads = 6  (head_dim = embed_dim / 6 = 32 for embed_dim=192)
    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        num_heads: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.img_size   = img_size    # (H, W)
        self.patch_size = patch_size  # (ph, pw)
        self.in_chans   = in_chans    # C
        self.embed_dim  = embed_dim   # D

        patch_area = patch_size[0] * patch_size[1]  # p²

        # Shared linear projection: one set of weights used for every channel.
        # Input: (B*nH*nW*C, patch_area)  →  output: (B*nH*nW*C, D)
        self.var_embed = nn.Linear(patch_area, embed_dim, bias=True)

        # Single shared learnable query vector; shape (1, 1, D).
        # Expanded to (B*nH*nW, 1, D) inside forward.
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.query, std=0.02)

        # Pre-norm: applied to K/V tokens before cross-attention
        self.norm_kv = norm_layer(embed_dim)

        # Cross-attention: 1 query token × C key-value tokens → 1 output token
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        # Output projection + dropout
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Weight initialisation
        nn.init.xavier_uniform_(self.var_embed.weight)
        nn.init.zeros_(self.var_embed.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input surface tensor (normalised)
        Returns:
            (B, embed_dim, nH, nW) patch-embedded tensor
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}×{W}) doesn't match model ({self.img_size[0]}×{self.img_size[1]})."
        )
        assert C == self.in_chans, f"Expected {self.in_chans} channels, got {C}."

        ph, pw = self.patch_size
        # Pad spatial dims to multiples of patch size
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // ph, Wp // pw
        N_loc = B * nH * nW  # total number of spatial locations in the batch

        # ── Step 1: extract patches per channel ─────────────────────────────
        # (B, C, nH, ph, nW, pw)
        x = x.view(B, C, nH, ph, nW, pw)
        # → (B, nH, nW, C, ph, pw)  → (N_loc, C, ph*pw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(N_loc, C, ph * pw)

        # ── Step 2: independently embed each channel (shared weights) ────────
        # (N_loc, C, ph*pw) → (N_loc, C, D)
        kv = self.var_embed(x)    # shared linear, applied per channel

        # ── Step 3: pre-norm on K/V tokens ───────────────────────────────────
        kv = self.norm_kv(kv)    # (N_loc, C, D)

        # ── Step 4: cross-attention — shared query attends over C K/V tokens ─
        # query: (1, 1, D) → (N_loc, 1, D)
        q = self.query.expand(N_loc, -1, -1)
        # cross_attn(query, key, value) → (N_loc, 1, D)
        out, _ = self.cross_attn(q, kv, kv)   # need_weights=False is default False

        # ── Step 5: output projection ─────────────────────────────────────────
        out = out.squeeze(1)                      # (N_loc, D)
        out = self.proj_drop(self.proj(out))

        # ── Step 6: reshape to spatial grid ──────────────────────────────────
        out = out.reshape(B, nH, nW, self.embed_dim)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, D, nH, nW)
        return out


class CrossAttnPatchEmbed3D(nn.Module):
    """
    ClimaX / Stormer-style variable-aware patch embedding for 3-D upper-air fields.

    3-D strategy (per user specification — "flatten all levels into K/V set"):
      For each 2-D spatial patch location (h, w):
        K/V set = all C channels × all Pl_patches pressure-level groups
                = C × Pl_patches tokens in total.
        Query   = Pl_patches separate learnable query vectors (one per
                  pressure-level output slot, shared across spatial positions).
        Each level's query attends over ALL C × Pl_patches tokens, so every
        output level token has access to information from every channel at
        every other level — true cross-level, cross-variable interaction.

    Output shape matches PatchEmbed3D: (B, embed_dim, Pl_patches, nH, nW)

    Design choices (per user specification):
      - "Shared query" interpreted as: one query vector per pressure-level slot
        (Pl_patches in total), each broadcast over all spatial positions.
      - Pre-norm: LayerNorm on K/V tokens before cross-attention.
      - num_heads = 6  (head_dim = embed_dim / 6 = 32 for embed_dim=192)
    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        num_heads: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.img_size   = img_size    # (L, H, W)
        self.patch_size = patch_size  # (pl, ph, pw)
        self.in_chans   = in_chans    # C
        self.embed_dim  = embed_dim   # D

        patch_l, patch_h, patch_w = patch_size
        patch_vol = patch_l * patch_h * patch_w  # volume of one channel's patch

        # Shared variable embedding: maps patch_vol → D for every (channel, level) token.
        self.var_embed = nn.Linear(patch_vol, embed_dim, bias=True)

        # We do NOT know Pl_patches at construction time (it depends on padding),
        # so the level-specific query bank is built lazily on first forward call.
        # Store the initialised queries as a buffer updated in _init_level_queries().
        self._pl_patches_cached: int = -1
        self.level_queries: nn.Parameter  # declared in _init_level_queries

        # Pre-norm on K/V tokens
        self.norm_kv = norm_layer(embed_dim)

        # Cross-attention: Pl_patches query tokens × (C*Pl_patches) K/V tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Weight init
        nn.init.xavier_uniform_(self.var_embed.weight)
        nn.init.zeros_(self.var_embed.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    # Lazy level-query initialisation (called on first forward when the
    # padded Pl dimension is known).
    # ------------------------------------------------------------------
    def _init_level_queries(self, pl_patches: int, device):
        if self._pl_patches_cached == pl_patches:
            return  # already initialised for this size
        self._pl_patches_cached = pl_patches
        queries = torch.zeros(1, pl_patches, self.embed_dim, device=device)
        trunc_normal_(queries, std=0.02)
        # Register as a proper Parameter so it is included in state_dict / optimiser
        self.level_queries = nn.Parameter(queries)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L, H, W)  upper-air tensor (normalised)
        Returns:
            (B, embed_dim, Pl_patches, nH, nW)
        """
        B, C, L, H, W = x.shape
        assert (L == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2]), (
            f"Input size ({L}×{H}×{W}) doesn't match model "
            f"({self.img_size[0]}×{self.img_size[1]}×{self.img_size[2]})."
        )
        assert C == self.in_chans, f"Expected {self.in_chans} channels, got {C}."

        pl, ph, pw = self.patch_size
        pad_l = (pl - L % pl) % pl
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_l or pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_l), 'constant', 0)
        Lp, Hp, Wp = L + pad_l, H + pad_h, W + pad_w
        nPl, nH, nW = Lp // pl, Hp // ph, Wp // pw
        N_loc = B * nH * nW  # spatial locations per batch
        N_kv  = C * nPl      # K/V tokens per spatial location (all channels × all levels)

        # Lazily build level query bank now that nPl is known
        self._init_level_queries(nPl, x.device)

        # ── Step 1: extract volumetric patches per channel ───────────────────
        # (B, C, nPl, pl, nH, ph, nW, pw)
        x = x.view(B, C, nPl, pl, nH, ph, nW, pw)
        # → (B, nH, nW, C, nPl, pl, ph, pw)  → (N_loc, C*nPl, pl*ph*pw)
        x = x.permute(0, 4, 6, 1, 2, 3, 5, 7).contiguous()
        x = x.reshape(N_loc, C * nPl, pl * ph * pw)

        # ── Step 2: independently embed each (channel, level) token ──────────
        # (N_loc, C*nPl, pl*ph*pw) → (N_loc, C*nPl, D)
        kv = self.var_embed(x)

        # ── Step 3: pre-norm on K/V tokens ───────────────────────────────────
        kv = self.norm_kv(kv)    # (N_loc, N_kv, D)

        # ── Step 4: cross-attention ───────────────────────────────────────────
        # Query: (1, nPl, D) → (N_loc, nPl, D)
        q = self.level_queries.expand(N_loc, -1, -1)
        # Each of the nPl query tokens attends over ALL N_kv = C*nPl K/V tokens
        # → (N_loc, nPl, D)
        out, _ = self.cross_attn(q, kv, kv)

        # ── Step 5: output projection ─────────────────────────────────────────
        out = self.proj_drop(self.proj(out))  # (N_loc, nPl, D)

        # ── Step 6: reshape to (B, D, nPl, nH, nW) ───────────────────────────
        out = out.reshape(B, nH, nW, nPl, self.embed_dim)
        # → (B, nPl, nH, nW, D)  → (B, D, nPl, nH, nW)
        out = out.permute(0, 4, 3, 1, 2).contiguous()
        return out


# ---------------------------------------------------------------------------
# The rest of the model (completely unchanged from train14_small_era5.py)
# ---------------------------------------------------------------------------

class Pangu_lite(nn.Module):
    """
    Pangu-Lite with cross-attention patch embedding.

    All architectural parameters (window sizes, depths, dims) are identical to
    train14_small_era5.py.  Only patchembed2d / patchembed3d are replaced.
    Training is from scratch (no pretrained weights).
    """

    def __init__(
        self,
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        residual=True,
        embed_attn_heads=6,
        embed_attn_drop=0.0,
        embed_checkpoint=True,
    ):
        # embed_checkpoint: if True, gradient checkpointing is applied to both
        # cross-attn patch embedders to avoid storing their large intermediate
        # K/V activation tensors during the forward pass (~1.2 GB per GPU saved).
        super().__init__()
        self.residual          = residual
        self.embed_checkpoint  = embed_checkpoint
        drop_path = np.linspace(0, 0.2, 8).tolist()

        # ── Cross-attention patch embedders (replaces flat PatchEmbed) ───────
        self.patchembed2d = CrossAttnPatchEmbed2D(
            img_size=(721, 1440),
            patch_size=(4, 4),
            in_chans=4 + 3,       # 4 surface vars + 3 static mask channels
            embed_dim=embed_dim,
            num_heads=embed_attn_heads,
            attn_drop=embed_attn_drop,
        )
        self.patchembed3d = CrossAttnPatchEmbed3D(
            img_size=(13, 721, 1440),
            patch_size=(2, 4, 4),
            in_chans=5 + 1,       # 5 upper-air vars + 1 const_h channel
            embed_dim=embed_dim,
            num_heads=embed_attn_heads,
            attn_drop=embed_attn_drop,
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(8, 181, 360),
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        self.downsample = DownSample(
            in_dim=embed_dim,
            input_resolution=(8, 181, 360),
            output_resolution=(8, 91, 180),
        )
        # ── Bottleneck ───────────────────────────────────────────────────────
        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(8, 91, 180),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(8, 91, 180),
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        # ── Decoder ──────────────────────────────────────────────────────────
        self.upsample = UpSample(embed_dim * 2, embed_dim, (8, 91, 180), (8, 181, 360))
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=(8, 181, 360),
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        # Skip-connection concatenation before patch recovery
        self.patchrecovery2d = PatchRecovery2D((721, 1440), (4, 4), 2 * embed_dim, 4)
        self.patchrecovery3d = PatchRecovery3D((13, 721, 1440), (2, 4, 4), 2 * embed_dim, 5)

        if self.residual:
            self._init_residual_heads()

    def _init_residual_heads(self):
        nn.init.zeros_(self.patchrecovery2d.conv.weight)
        nn.init.zeros_(self.patchrecovery2d.conv.bias)
        nn.init.zeros_(self.patchrecovery3d.conv.weight)
        nn.init.zeros_(self.patchrecovery3d.conv.bias)

    def forward(self, surface, surface_mask, upper_air, const_h):
        input_surface   = surface
        input_upper_air = upper_air

        # Expand surface_mask and const_h to match batch size if needed
        if surface_mask.dim() == 3:
            surface_mask = surface_mask.unsqueeze(0).expand(surface.size(0), -1, -1, -1)
        elif surface_mask.size(0) == 1 and surface.size(0) > 1:
            surface_mask = surface_mask.expand(surface.size(0), -1, -1, -1)
        if const_h.dim() == 3:
            const_h = const_h.unsqueeze(0).unsqueeze(0)
        elif const_h.dim() == 4:
            const_h = const_h.unsqueeze(0)
        if const_h.size(0) == 1 and upper_air.size(0) > 1:
            const_h = const_h.expand(upper_air.size(0), -1, -1, -1, -1)

        # Concatenate static masks / const_h before embedding
        surface   = torch.cat([surface, surface_mask], dim=1)      # (B, 7, H, W)
        upper_air = torch.cat([upper_air, const_h],    dim=1)      # (B, 6, 13, H, W)

        # ── Cross-attention patch embedding ───────────────────────────────────
        # Gradient checkpointing (embed_checkpoint=True by default) avoids
        # storing the large K/V activation tensors during forward:
        #   2D embed: (N_loc, C=7, D)          ≈  170 MB  float16
        #   3D embed: (N_loc, C*Pl=42, D)      ≈  1.0 GB  float16
        # During backward, PyTorch re-runs the embed forward to recompute them.
        if self.embed_checkpoint and self.training:
            surface   = grad_checkpoint(self.patchembed2d, surface,   use_reentrant=False)
            upper_air = grad_checkpoint(self.patchembed3d, upper_air, use_reentrant=False)
        else:
            surface   = self.patchembed2d(surface)     # (B, 192, 181, 360)
            upper_air = self.patchembed3d(upper_air)   # (B, 192, 7,  181, 360)

        # Merge into a single 3-D token sequence
        x = torch.cat([surface.unsqueeze(2), upper_air], dim=2)    # (B, 192, 8, 181, 360)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)                    # (B, N, 192)

        # ── Encoder → bottleneck → decoder ───────────────────────────────────
        x    = self.layer1(x)
        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        # ── Skip connection + patch recovery ─────────────────────────────────
        output = torch.cat([x, skip], dim=-1)                       # (B, N, 2*192)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface   = output[:, :, 0,  :, :]                  # (B, 2*192, 181, 360)
        output_upper_air = output[:, :, 1:, :, :]                  # (B, 2*192, 7,  181, 360)

        output_surface   = self.patchrecovery2d(output_surface)     # (B, 4,  721, 1440)
        output_upper_air = self.patchrecovery3d(output_upper_air)   # (B, 5, 13, 721, 1440)

        if self.residual:
            output_surface   = input_surface   + output_surface
            output_upper_air = input_upper_air + output_upper_air
        return output_surface, output_upper_air


# ---------------------------------------------------------------------------
# Supporting modules (identical to train14_small_era5.py)
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            EarthSpecificBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=None,
                 shift_size=None, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size  = (1, 3, 6)  if shift_size  is None else shift_size
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio

        self.norm1   = norm_layer(dim)
        padding      = get_pad3d(input_resolution, window_size)
        self.pad     = nn.ZeroPad3d(padding)

        pad_resolution    = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2]  + padding[3])
        pad_resolution[2] += (padding[0]  + padding[1])

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size) if self.roll else None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            x_windows = window_partition(x, self.window_size)

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1],
                                    win_pl * win_lat * win_lon, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1],
                                          win_pl, win_lat, win_lon, C)
        if self.roll:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)

        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_pad3d(input_resolution, window_size):
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder  = Pl  % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon
    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back  = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top    = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left  = lon_pad // 2
        padding_right = lon_pad - padding_left
    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back

def get_pad2d(input_resolution, window_size):
    input_resolution = [2] + list(input_resolution)
    window_size      = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


class EarthAttention3D(nn.Module):
    def __init__(self, dim, input_resolution, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        self.type_of_windows = (
            (input_resolution[0] // window_size[0]) *
            (input_resolution[1] // window_size[1])
        )
        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                self.type_of_windows, num_heads
            )
        )
        earth_position_index = get_earth_position_index(window_size)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, nW_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows, -1
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = (attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N)
                    + mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_earth_position_index(window_size):
    win_pl, win_lat, win_lon = window_size
    coords_zi = torch.arange(win_pl)
    coords_zj = -torch.arange(win_pl) * win_pl
    coords_hi = torch.arange(win_lat)
    coords_hj = -torch.arange(win_lat) * win_lat
    coords_w  = torch.arange(win_lon)
    coords_1  = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2  = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()
    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    position_index = coords.sum(-1)
    return position_index


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size):
    B, Pl, Lat, Lon, C = x.shape
    win_pl, win_lat, win_lon = window_size
    x = x.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
    windows = (x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous()
                .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C))
    return windows


def window_reverse(windows, window_size, Pl, Lat, Lon):
    win_pl, win_lat, win_lon = window_size
    B = int(windows.shape[0] / (Lon / win_lon))
    x = windows.view(B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1)
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
    return x


def get_shift_window_mask(input_resolution, window_size, shift_size):
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    shift_pl, shift_lat, shift_lon = shift_size
    img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    pl_slices  = (slice(0, -win_pl),  slice(-win_pl,  -shift_pl),  slice(-shift_pl, None))
    lat_slices = (slice(0, -win_lat), slice(-win_lat, -shift_lat), slice(-shift_lat, None))
    lon_slices = (slice(0, -win_lon), slice(-win_lon, -shift_lon), slice(-shift_lon, None))
    cnt = 0
    for pl in pl_slices:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, pl, lat, lon, :] = cnt
                cnt += 1
    img_mask    = img_mask[:, :, :, :Lon, :]
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(mask_windows.shape[0], mask_windows.shape[1],
                                      win_pl * win_lat * win_lon)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class DownSample(nn.Module):
    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear           = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm             = nn.LayerNorm(4 * in_dim)
        self.input_resolution  = input_resolution
        self.output_resolution = output_resolution
        in_pl, in_lat, in_lon   = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        assert in_pl == out_pl, "pressure level dimension should not change in DownSample"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon
        pad_top = h_pad // 2;  pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2; pad_right  = w_pad - pad_left
        self.pad = nn.ZeroPad3d((pad_left, pad_right, pad_top, pad_bottom, 0, 0))

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon   = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)
        x = self.norm(x)
        x = self.linear(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1           = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2           = nn.Linear(out_dim, out_dim, bias=False)
        self.norm              = nn.LayerNorm(out_dim)
        self.input_resolution  = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_pl, in_lat, in_lon   = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)
        assert in_pl == out_pl, "pressure level dimension should not change in UpSample"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon
        pad_top   = pad_h // 2;  pad_bottom = pad_h - pad_top
        pad_left  = pad_w // 2;  pad_right  = pad_w - pad_left
        x = x[:, :out_pl, pad_top: 2 * in_lat - pad_bottom, pad_left: 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class PatchRecovery2D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.out_chans  = out_chans
        self.conv = nn.Conv1d(in_chans, out_chans * patch_size[0] * patch_size[1],
                              kernel_size=1, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        output = x.view(B, C, -1)
        output = self.conv(output)
        output = output.view(B, self.out_chans, patch_h, patch_w, H, W)
        output = output.permute(0, 1, 4, 2, 5, 3).contiguous()
        output = output.reshape(B, self.out_chans, H * patch_h, W * patch_w)
        return output[:, :, :self.img_size[0], :self.img_size[1]]


class PatchRecovery3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.out_chans  = out_chans
        self.conv = nn.Conv1d(
            in_chans,
            out_chans * patch_size[0] * patch_size[1] * patch_size[2],
            kernel_size=1, stride=1,
        )

    def forward(self, x: torch.Tensor):
        B, C, Pl, Lat, Lon = x.shape
        patch_l, patch_h, patch_w = self.patch_size
        output = x.view(B, C, -1)
        output = self.conv(output)
        output = output.view(B, self.out_chans, patch_l, patch_h, patch_w, Pl, Lat, Lon)
        output = output.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        output = output.reshape(B, self.out_chans, Pl * patch_l, Lat * patch_h, Lon * patch_w)
        return output[:, :, :self.img_size[0], :self.img_size[1], :self.img_size[2]]


def crop3d(x: torch.Tensor, resolution):
    _, _, Pl, Lat, Lon = x.shape
    pl_pad  = Pl  - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]
    padding_front = pl_pad  // 2;  padding_back   = pl_pad  - padding_front
    padding_top   = lat_pad // 2;  padding_bottom = lat_pad - padding_top
    padding_left  = lon_pad // 2;  padding_right  = lon_pad - padding_left
    return x[
        :, :,
        padding_front : Pl  - padding_back,
        padding_top   : Lat - padding_bottom,
        padding_left  : Lon - padding_right,
    ]


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    opt = parser.parse_args()
    NUM_EPOCHS = opt.num_epochs

    if opt.dist:
        backend = _pick_backend(opt.backend)
        init_dist(opt.launcher, backend=backend)
        rank, world_size = get_dist_info()
    else:
        rank, world_size = 0, 1

    if torch.cuda.is_available():
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        elif opt.local_rank >= 0:
            local_rank = opt.local_rank
        else:
            local_rank = rank % max(1, torch.cuda.device_count())
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print(f"[rank{rank}] host={os.uname().nodename}  local_rank={local_rank}  device={device}", flush=True)
    if rank == 0:
        print(f"[rank0] world_size={world_size}  (total GPUs in use)", flush=True)

    split_ranges = {
        "train": (opt.train_start, opt.train_end),
        "valid": (opt.valid_start, opt.valid_end),
        "test":  (opt.test_start,  opt.test_end),
    }
    aux_data_dir = opt.aux_data_dir or os.path.join(opt.data_dir, "aux_data")

    if rank == 0:
        print("[rank0] Scanning dataset (rank-0 only to avoid NFS storm) ...", flush=True)

    run_validation = opt.val_every > 0

    if rank == 0:
        train_set = ERA5DatasetFromFolder(
            opt.data_dir, "train",
            lead_hours=opt.horizon_hours,
            sample_stride_hours=opt.sample_stride_hours,
            split_ranges=split_ranges,
            cache_size=opt.dataset_cache_size,
            aux_data_dir=aux_data_dir,
        )
        val_set = ERA5DatasetFromFolder(
            opt.data_dir, "valid",
            lead_hours=opt.horizon_hours,
            sample_stride_hours=opt.sample_stride_hours,
            split_ranges=split_ranges,
            cache_size=opt.dataset_cache_size,
            aux_data_dir=aux_data_dir,
        ) if run_validation else None
        broadcast_payload = {
            "train_samples": train_set.samples,
            "val_samples":   val_set.samples if val_set is not None else [],
        }
    else:
        broadcast_payload = None

    if opt.dist:
        obj_list = [broadcast_payload]
        dist.broadcast_object_list(obj_list, src=0)
        broadcast_payload = obj_list[0]

    if rank != 0:
        train_set = ERA5DatasetFromFolder(
            opt.data_dir, "train",
            lead_hours=opt.horizon_hours,
            sample_stride_hours=opt.sample_stride_hours,
            split_ranges=split_ranges,
            cache_size=opt.dataset_cache_size,
            aux_data_dir=aux_data_dir,
            _prebuilt_samples=broadcast_payload["train_samples"],
        )
        val_set = None

    if rank == 0:
        print(f"Data dir:    {opt.data_dir}")
        print(f"Aux data dir:{aux_data_dir}")
        print(f"Train range: {opt.train_start} -> {opt.train_end}")
        print(f"Valid range: {opt.valid_start} -> {opt.valid_end}")
        print(f"Test range:  {opt.test_start} -> {opt.test_end}")
        print(f"Monthly dataset cache size per worker: {opt.dataset_cache_size}")
        print(f"Train samples: {len(train_set)}")
        if run_validation:
            print(f"Valid samples: {len(val_set)}")
        else:
            print("Validation disabled")

    if opt.dist:
        dist.barrier()
        if rank == 0:
            print("[rank0] All ranks ready — starting training.", flush=True)

    loader_kwargs = {
        "batch_size": opt.batch_size,
        "num_workers": opt.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if opt.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = opt.prefetch_factor

    if opt.dist:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
        train_loader  = DataLoader(train_set, sampler=train_sampler, **loader_kwargs)
    else:
        train_sampler = None
        train_loader  = DataLoader(train_set, shuffle=True, **loader_kwargs)

    val_loader = (
        DataLoader(val_set, shuffle=False, **loader_kwargs)
        if rank == 0 and run_validation else None
    )

    land_mask, soil_type, topography = train_set.get_constant_mask()
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0)  # (3, H, W)
    const_h      = train_set.get_const_h()

    # ── Build model (from scratch — no pretrained weights) ───────────────────
    pangu_lite = Pangu_lite(
        residual=opt.residual,
        embed_attn_heads=opt.embed_heads,
        embed_attn_drop=opt.embed_attn_drop,
        embed_checkpoint=opt.embed_checkpoint,
    )
    surface_criterion   = nn.L1Loss(reduction='none')
    upper_air_criterion = nn.L1Loss(reduction='none')

    pangu_lite.to(device)
    surface_criterion.to(device)
    upper_air_criterion.to(device)
    surface_mask = surface_mask.to(device)
    const_h      = const_h.to(device)
    surface_weights = torch.tensor(SURFACE_WEIGHTS, dtype=torch.float32, device=device).view(1, 4, 1, 1)
    upper_weights   = torch.tensor(UPPER_WEIGHTS,   dtype=torch.float32, device=device).view(1, 5, 1, 1, 1)

    if opt.compile and hasattr(torch, "compile"):
        pangu_lite = torch.compile(pangu_lite)

    if opt.dist:
        if torch.cuda.is_available():
            pangu_lite = DDP(pangu_lite, device_ids=[local_rank], output_device=local_rank)
        else:
            pangu_lite = DDP(pangu_lite)

    if rank == 0:
        n_params = sum(p.numel() for p in pangu_lite.parameters())
        print(f"PanguLite-Innovate ERA5 parameters: {n_params:,}", flush=True)
        print(f"Residual output: {opt.residual} | AMP: {bool(opt.amp)} | loss: {opt.loss_type}", flush=True)
        print(f"Embed heads: {opt.embed_heads} | Embed attn_drop: {opt.embed_attn_drop}", flush=True)
        print(f"Surface variables: {SURFACE_VARIABLES}", flush=True)
        print(f"Upper variables:   {UPPER_VARIABLES}", flush=True)
        print(f"Pressure levels:   {PANGU_LEVELS}", flush=True)
        print(f"LR={opt.lr}  weight_decay={opt.weight_decay}  batch_size={opt.batch_size}  "
              f"num_workers={opt.num_workers}", flush=True)

    surface_invTrans, surface_variables = surface_inv_transform(
        os.path.join(aux_data_dir, "surface_mean.npy"),
        os.path.join(aux_data_dir, "surface_std.npy"),
    )
    upper_air_invTrans, upper_air_variables, upper_air_pLevels = upper_air_inv_transform(
        os.path.join(aux_data_dir, "upper_mean.npy"),
        os.path.join(aux_data_dir, "upper_std.npy"),
    )

    optimizer = torch.optim.Adam(pangu_lite.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    use_amp   = bool(opt.amp and torch.cuda.is_available())
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    results          = []
    best_val         = float("inf")
    start_epoch      = 1
    patience_counter = 0
    show_progress    = rank == 0 and sys.stderr.isatty()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if opt.resume and rank == 0:
        latest_path = os.path.join(opt.output_dir, "pangu_lite_innovate_era5_latest.pth")
        if os.path.isfile(latest_path):
            print(f"[rank0] Resuming from checkpoint: {latest_path}", flush=True)
            ckpt   = torch.load(latest_path, map_location=device)
            raw_sd = ckpt.get("model", ckpt)
            raw_sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                      for k, v in raw_sd.items()}
            unwrap_model(pangu_lite).load_state_dict(raw_sd, strict=True)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler"])
            if "best_val" in ckpt:
                best_val = ckpt["best_val"]
            if "patience_counter" in ckpt:
                patience_counter = ckpt["patience_counter"]
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"[rank0] Resumed at epoch {start_epoch}  best_val={best_val:.6f}", flush=True)
        else:
            print(f"[rank0] --resume set but no checkpoint found at {latest_path}; starting fresh.", flush=True)

    if is_dist_ready():
        meta = torch.tensor([start_epoch, best_val, patience_counter], dtype=torch.float64, device=device)
        dist.broadcast(meta, src=0)
        start_epoch      = int(meta[0].item())
        best_val         = float(meta[1].item())
        patience_counter = int(meta[2].item())

    LOG_EVERY_N_BATCHES = 50

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start      = time.perf_counter()
        epoch_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if opt.dist:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            n_batches = len(train_loader)
            print(
                f"\n[rank0] === Epoch {epoch}/{NUM_EPOCHS} | {epoch_started_at} "
                f"| {n_batches} batches | Loading first batch from NFS ...",
                flush=True,
            )
            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d}/{NUM_EPOCHS}",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress,
            )
        else:
            train_bar = train_loader

        running_results = {"batch_sizes": 0, "loss": 0, "surface_l1": 0, "upper_air_l1": 0}
        pangu_lite.train()

        _t_data_start  = time.perf_counter()
        _total_data_sec = 0.0
        _total_gpu_sec  = 0.0

        for batch_idx, (input_surface, input_upper_air, target_surface, target_upper_air) in enumerate(train_bar):
            _t_loaded = time.perf_counter()
            _total_data_sec += _t_loaded - _t_data_start
            batch_size = input_surface.size(0)
            input_surface    = input_surface.to(device, non_blocking=True)
            input_upper_air  = input_upper_air.to(device, non_blocking=True)
            target_surface   = target_surface.to(device, non_blocking=True)
            target_upper_air = target_upper_air.to(device, non_blocking=True)

            if rank == 0 and batch_idx == 0:
                print(f"[rank0] Epoch {epoch} batch 0 | data_load={_t_loaded - _t_data_start:.1f}s", flush=True)

            optimizer.zero_grad(set_to_none=True)
            _t_fwd = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output_surface, output_upper_air = pangu_lite(
                    input_surface, surface_mask, input_upper_air, const_h
                )
                surface_loss_map   = surface_criterion(output_surface, target_surface)
                upper_air_loss_map = upper_air_criterion(output_upper_air, target_upper_air)
                if opt.loss_type == 'weighted_l1':
                    surface_loss   = torch.mean(surface_loss_map   * surface_weights)
                    upper_air_loss = torch.mean(upper_air_loss_map * upper_weights)
                else:
                    surface_loss   = torch.mean(surface_loss_map)
                    upper_air_loss = torch.mean(upper_air_loss_map)
                loss = upper_air_loss * opt.upper_loss_weight + surface_loss * opt.surface_loss_weight

            _t_bwd = time.perf_counter()
            if rank == 0 and batch_idx == 0:
                print(
                    f"[rank0] Epoch {epoch} batch 0 | forward={_t_bwd - _t_fwd:.1f}s "
                    f"| Backward+allreduce starting ...", flush=True,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pangu_lite.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            _t_end = time.perf_counter()
            _total_gpu_sec += _t_end - _t_loaded

            if rank == 0 and batch_idx == 0:
                print(
                    f"[rank0] Epoch {epoch} batch 0 | backward+allreduce={_t_end - _t_bwd:.1f}s | batch DONE",
                    flush=True,
                )

            running_results["loss"]          += loss.item()          * batch_size
            running_results["surface_l1"]    += surface_loss.detach().cpu().item() * batch_size
            running_results["upper_air_l1"]  += upper_air_loss.detach().cpu().item() * batch_size
            running_results["batch_sizes"]   += batch_size

            if rank == 0 and (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0:
                n_done   = batch_idx + 1
                elapsed  = time.perf_counter() - epoch_start
                eta_sec  = elapsed / n_done * (n_batches - n_done)
                avg_loss = running_results["loss"] / max(1, running_results["batch_sizes"])
                avg_data = _total_data_sec / n_done
                avg_gpu  = _total_gpu_sec  / n_done
                print(
                    f"[rank0] Epoch {epoch} | batch {n_done}/{n_batches} "
                    f"| loss={avg_loss:.4f} "
                    f"| data={avg_data:.1f}s/batch gpu={avg_gpu:.1f}s/batch "
                    f"| elapsed={elapsed/60:.1f}min | ETA={eta_sec/60:.1f}min",
                    flush=True,
                )

            if rank == 0 and show_progress:
                train_bar.set_postfix(
                    loss=running_results["loss"]         / max(1, running_results["batch_sizes"]),
                    upper=running_results["upper_air_l1"]/ max(1, running_results["batch_sizes"]),
                    surface=running_results["surface_l1"]/ max(1, running_results["batch_sizes"]),
                )
            _t_data_start = time.perf_counter()

        # ── Epoch timing summary ──────────────────────────────────────────────
        if rank == 0:
            n_batches_done = max(1, batch_idx + 1)
            print(
                f"[rank0] Epoch {epoch} timing | "
                f"avg_data={_total_data_sec/n_batches_done:.1f}s/batch "
                f"avg_gpu={_total_gpu_sec/n_batches_done:.1f}s/batch "
                f"gpu_util={100*_total_gpu_sec/max(1e-6,_total_data_sec+_total_gpu_sec):.1f}%",
                flush=True,
            )

        train_stats = torch.tensor(
            [running_results["loss"], running_results["surface_l1"],
             running_results["upper_air_l1"], running_results["batch_sizes"]],
            dtype=torch.float64, device=device,
        )
        if is_dist_ready():
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        train_loss         = (train_stats[0] / train_stats[3]).item()
        train_surface_l1   = (train_stats[1] / train_stats[3]).item()
        train_upper_air_l1 = (train_stats[2] / train_stats[3]).item()

        should_validate       = run_validation and epoch % opt.val_every == 0
        surface_mse_epoch     = float("nan")
        upper_air_mse_epoch   = float("nan")
        val_score             = float("inf")

        # ── Validation (rank-0 only) ──────────────────────────────────────────
        if rank == 0 and should_validate:
            eval_model = unwrap_model(pangu_lite)
            eval_model.eval()
            valing_results = {"batch_sizes": 0, "surface_mse": 0, "upper_air_mse": 0}
            val_bar = tqdm(
                val_loader, desc=f"Valid {epoch:03d}/{NUM_EPOCHS}",
                leave=False, dynamic_ncols=True, disable=not show_progress,
            )
            with torch.no_grad():
                for (val_input_surface, val_input_upper_air,
                     val_target_surface, val_target_upper_air, times) in val_bar:
                    batch_size       = val_input_surface.size(0)
                    val_input_surface    = val_input_surface.to(device, non_blocking=True)
                    val_input_upper_air  = val_input_upper_air.to(device, non_blocking=True)
                    val_target_surface   = val_target_surface.to(device, non_blocking=True)
                    val_target_upper_air = val_target_upper_air.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_output_surface, val_output_upper_air = eval_model(
                            val_input_surface, surface_mask, val_input_upper_air, const_h,
                        )
                    valing_results["batch_sizes"]  += batch_size
                    surface_mse   = ((val_output_surface   - val_target_surface  ) ** 2).mean().detach().cpu().item()
                    upper_air_mse = ((val_output_upper_air - val_target_upper_air) ** 2).mean().detach().cpu().item()
                    valing_results["surface_mse"]   += surface_mse   * batch_size
                    valing_results["upper_air_mse"] += upper_air_mse * batch_size
                    if show_progress:
                        val_bar.set_postfix(
                            surface_mse=valing_results["surface_mse"]   / valing_results["batch_sizes"],
                            upper_mse=  valing_results["upper_air_mse"] / valing_results["batch_sizes"],
                        )
            surface_mse_epoch   = valing_results["surface_mse"]   / valing_results["batch_sizes"]
            upper_air_mse_epoch = valing_results["upper_air_mse"] / valing_results["batch_sizes"]
            val_score           = (upper_air_mse_epoch * opt.upper_loss_weight
                                   + surface_mse_epoch * opt.surface_loss_weight)

        # ── Checkpoint saving (rank-0 only) ───────────────────────────────────
        if rank == 0:
            os.makedirs(opt.output_dir, exist_ok=True)
            if should_validate:
                if val_score < best_val:
                    best_val         = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1

            checkpoint = {
                "model":             unwrap_model(pangu_lite).state_dict(),
                "optimizer":         optimizer.state_dict(),
                "scaler":            scaler.state_dict(),
                "epoch":             epoch,
                "best_val":          best_val,
                "patience_counter":  patience_counter,
                "model_version":     "pangulite_innovate_era5_crossattn_embed_v1",
                "residual":          opt.residual,
                "horizon_hours":     opt.horizon_hours,
                "sample_stride_hours": opt.sample_stride_hours,
                "surface_variables": SURFACE_VARIABLES,
                "upper_variables":   UPPER_VARIABLES,
                "pressure_levels":   PANGU_LEVELS,
                "patch_size_surface": (4, 4),
                "patch_size_upper":   (2, 4, 4),
                "embed_attn_heads":  opt.embed_heads,
                "loss_type":         opt.loss_type,
                "args":              vars(opt),
                "train_loss":        train_loss,
                "train_surface_l1":  train_surface_l1,
                "train_upper_air_l1": train_upper_air_l1,
                "surface_mse":       surface_mse_epoch,
                "upper_air_mse":     upper_air_mse_epoch,
                "val_score":         val_score,
            }
            latest_path = os.path.join(opt.output_dir, "pangu_lite_innovate_era5_latest.pth")
            torch.save(checkpoint, latest_path)
            if opt.save_every > 0 and epoch % opt.save_every == 0:
                torch.save(checkpoint, os.path.join(
                    opt.output_dir, f"pangu_lite_innovate_era5_epoch_{epoch:03d}.pth"
                ))
            if should_validate and val_score == best_val and patience_counter == 0:
                torch.save(checkpoint, os.path.join(opt.output_dir, "pangu_lite_innovate_era5_best.pth"))
                print(f"[rank0] New best val_score={best_val:.6f} — saved best.pth", flush=True)
            if should_validate and opt.early_stop_patience > 0:
                print(
                    f"[rank0] Early-stop patience: {patience_counter}/{opt.early_stop_patience}",
                    flush=True,
                )

        epoch_sec        = time.perf_counter() - epoch_start
        epoch_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if rank == 0:
            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
                f"train_loss={train_loss:.6f} | "
                f"train_upper_l1={train_upper_air_l1:.6f} | "
                f"train_surface_l1={train_surface_l1:.6f} | "
                f"val_upper_mse={upper_air_mse_epoch:.6f} | "
                f"val_surface_mse={surface_mse_epoch:.6f} | "
                f"time={epoch_sec / 60:.2f} min"
            )
            time_log_dir  = opt.log_dir
            os.makedirs(time_log_dir, exist_ok=True)
            time_log_path = os.path.join(time_log_dir, f"epoch_{epoch:03d}_time.txt")
            with open(time_log_path, "w", encoding="utf-8") as f:
                f.write(f"epoch={epoch}\n")
                f.write(f"started_at={epoch_started_at}\n")
                f.write(f"finished_at={epoch_finished_at}\n")
                f.write(f"seconds={epoch_sec:.3f}\n")
                f.write(f"minutes={epoch_sec / 60:.3f}\n")

            results.append({
                "epoch":         epoch,
                "started_at":    epoch_started_at,
                "finished_at":   epoch_finished_at,
                "seconds":       epoch_sec,
                "minutes":       epoch_sec / 60,
                "loss":          train_loss,
                "surface_l1":    train_surface_l1,
                "upper_air_l1":  train_upper_air_l1,
                "surface_mse":   surface_mse_epoch,
                "upper_air_mse": upper_air_mse_epoch,
                "val_score":     val_score,
                "best_val_score": best_val,
            })
            data_frame = pd.DataFrame(data=results)
            save_root  = opt.log_dir
            os.makedirs(save_root, exist_ok=True)
            data_frame.to_csv(os.path.join(save_root, "pangulite_innovate_era5_logs.csv"), index=False)

            summary_path = os.path.join(save_root, "training_summary.log")
            best_marker  = " *** BEST ***" if (should_validate and patience_counter == 0) else ""
            with open(summary_path, "a", encoding="utf-8") as sf:
                sf.write(
                    f"Epoch {epoch:03d}/{NUM_EPOCHS} "
                    f"| train={train_loss:.4f} "
                    f"| val={val_score:.4f} "
                    f"| best={best_val:.4f} "
                    f"| patience={patience_counter}/{opt.early_stop_patience if opt.early_stop_patience else '-'} "
                    f"| {epoch_sec/60:.1f}min"
                    f"{best_marker}\n"
                )

        # ── Early-stop broadcast ──────────────────────────────────────────────
        stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
        if rank == 0 and opt.early_stop_patience > 0 and patience_counter >= opt.early_stop_patience:
            print(
                f"[rank0] Early stopping triggered: val_score did not improve for "
                f"{patience_counter} consecutive validation epochs. Stopping.",
                flush=True,
            )
            stop_flag[0] = 1
        if is_dist_ready():
            dist.broadcast(stop_flag, src=0)

        if is_dist_ready():
            dist.barrier()

        if stop_flag[0].item() == 1:
            break

    train_set.close()
    if val_set is not None:
        val_set.close()
    if is_dist_ready():
        dist.destroy_process_group()
