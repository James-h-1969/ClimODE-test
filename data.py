"""Data loading utilities for ClimODE global forecasting.

Loads preprocessed ERA5 data from ClimaX .npz format (output of nc2np_equally_era5.py).
Variables: Z500, T850, T2m, U10, V10.

Expected data directory structure (same as ClimaX output):
    data_npz/
    ├── train/          # {year}_{shard}.npz files
    ├── val/
    ├── test/
    ├── lat.npy
    ├── lon.npy
    ├── normalize_mean.npz
    └── normalize_std.npz
"""

import logging
import os
from typing import List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from model import VelocityOptimizer

logger = logging.getLogger(__name__)

# Weather variables: short names and their keys in the npz files
VARIABLES = ["z", "t", "t2m", "u10", "v10"]
NPZ_KEYS = [
    "geopotential_500",
    "temperature_850",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# Data splits
TRAIN_YEARS = list(range(2006, 2016))
VAL_YEARS = [2016]
TEST_YEARS = [2017, 2018]

NUM_SHARDS = 8
SUBSAMPLE_FACTOR = 6  # hourly -> 6-hourly

# Grid dimensions (set at runtime by create_dataloaders)
GRID_H, GRID_W = 32, 64


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def _load_year_from_npz(data_dir: str, partition: str, year: int, keys: List[str]) -> dict:
    """Load all shards for a year, concatenate along time axis."""
    arrays = {k: [] for k in keys}
    for shard_id in range(NUM_SHARDS):
        path = os.path.join(data_dir, partition, f"{year}_{shard_id}.npz")
        data = np.load(path)
        for k in keys:
            arrays[k].append(data[k])
        data.close()
    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


def _get_available_keys(data_dir: str, partition: str, year: int) -> List[str]:
    """Check what keys are in the first shard of a year."""
    path = os.path.join(data_dir, partition, f"{year}_0.npz")
    data = np.load(path)
    keys = list(data.keys())
    data.close()
    return keys


def load_all_variables(data_root: str, dev_run: bool = False) -> dict:
    """Load ERA5 variables from ClimaX npz files.

    Args:
        data_root: Path to npz directory with train/, val/, test/, lat.npy, lon.npy.
        dev_run: If True, load only 1 train year and 1 val/test year for quick testing.

    Returns:
        Dictionary with train_data, val_data, test_data, time_steps,
        lat, lon, max_vals, min_vals, const_info, lat_map, lon_map.
    """
    logger.info(f"Loading ERA5 data from: {data_root}")

    lat = np.load(os.path.join(data_root, "lat.npy"))
    lon = np.load(os.path.join(data_root, "lon.npy"))
    H, W = len(lat), len(lon)
    logger.info(f"Grid: {H}x{W}")

    # Check what keys are available in the data
    available = _get_available_keys(data_root, "train", TRAIN_YEARS[0])
    logger.info(f"Available keys in npz: {available[:10]}...")

    # Verify our required variables exist
    missing = [k for k in NPZ_KEYS if k not in available]
    if missing:
        raise ValueError(f"Required variables missing from npz: {missing}")

    # Determine year ranges
    train_years = TRAIN_YEARS[:1] if dev_run else TRAIN_YEARS
    val_years = VAL_YEARS[:1] if dev_run else VAL_YEARS
    test_years = TEST_YEARS[:1] if dev_run else TEST_YEARS

    if dev_run:
        logger.info(f"DEV RUN: using train={train_years}, val={val_years}, test={test_years}")

    # Load data
    all_data = {}
    global_max = {k: -np.inf for k in NPZ_KEYS}
    global_min = {k: np.inf for k in NPZ_KEYS}

    for partition, years in [("train", train_years), ("val", val_years), ("test", test_years)]:
        all_data[partition] = []
        for year in years:
            year_arrays = _load_year_from_npz(data_root, partition, year, NPZ_KEYS)
            max_steps = 365 * (24 // SUBSAMPLE_FACTOR)  # 1460
            year_6h = {}
            for k in NPZ_KEYS:
                arr = year_arrays[k][::SUBSAMPLE_FACTOR][:max_steps]  # (1460, 1, H, W)
                year_6h[k] = arr
                if partition == "train":
                    global_max[k] = max(global_max[k], float(arr.max()))
                    global_min[k] = min(global_min[k], float(arr.min()))
            all_data[partition].append(year_6h)

    max_vals = [global_max[k] for k in NPZ_KEYS]
    min_vals = [global_min[k] for k in NPZ_KEYS]
    for var, k, mx, mn in zip(VARIABLES, NPZ_KEYS, max_vals, min_vals):
        logger.info(f"  {var} ({k}): [{mn:.2f}, {mx:.2f}]")

    def build_tensor(partition_data: list) -> torch.Tensor:
        """Build (T, num_years, K, H, W) tensor."""
        year_tensors = []
        for year_6h in partition_data:
            channels = []
            for k, mx, mn in zip(NPZ_KEYS, max_vals, min_vals):
                arr = (year_6h[k] - mn) / (mx - mn)
                channels.append(arr[:, 0, :, :])  # (T, H, W)
            year_tensors.append(np.stack(channels, axis=1))  # (T, K, H, W)
        return torch.from_numpy(np.stack(year_tensors, axis=1)).float()

    train_data = build_tensor(all_data["train"])
    val_data = build_tensor(all_data["val"])
    test_data = build_tensor(all_data["test"])
    logger.info(f"Shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Constants - try loading from npz, fall back to zeros
    first_shard = np.load(os.path.join(data_root, "train", f"{train_years[0]}_0.npz"))
    shard_keys = list(first_shard.keys())

    if "orography" in shard_keys and "land_sea_mask" in shard_keys:
        oro = torch.from_numpy(first_shard["orography"][0]).float().view(1, 1, H, W)
        lsm = torch.from_numpy(first_shard["land_sea_mask"][0]).float().view(1, 1, H, W)
        logger.info("Loaded constants: orography, land_sea_mask")
    else:
        logger.warning("Constants not in npz, using zeros for orography/land_sea_mask")
        oro = torch.zeros(1, 1, H, W)
        lsm = torch.zeros(1, 1, H, W)
    first_shard.close()

    const_info = torch.cat([oro, lsm], dim=1)
    lat_map = torch.from_numpy(
        np.expand_dims(lat, 1).repeat(W, axis=1)
    ).float()
    lon_map = torch.from_numpy(
        np.expand_dims(lon, 0).repeat(H, axis=0)
    ).float()

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "time_steps": torch.arange(365 * 4).view(-1, 1),
        "lat": lat,
        "lon": lon,
        "max_vals": max_vals,
        "min_vals": min_vals,
        "const_info": const_info,
        "lat_map": lat_map,
        "lon_map": lon_map,
    }


def create_dataloaders(data: dict, batch_size: int = 13) -> dict:
    """Create DataLoaders for training, validation, and testing."""
    global GRID_H, GRID_W
    GRID_H = data["train_data"].shape[3]
    GRID_W = data["train_data"].shape[4]

    train_loader = DataLoader(data["train_data"][2:], batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(data["val_data"][2:], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data["test_data"][2:], batch_size=batch_size, shuffle=False)
    time_loader = DataLoader(data["time_steps"][2:], batch_size=batch_size, shuffle=False)
    time_idx = DataLoader(
        torch.arange(365 * 4).view(-1, 1)[2:], batch_size=batch_size, shuffle=False
    )

    logger.info(f"DataLoaders: batch_size={batch_size}, lead_time={(batch_size-1)*6}h")
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "time_loader": time_loader,
        "time_idx": time_idx,
    }


def compute_gaussian_kernel(lat: np.ndarray, lon: np.ndarray, save_path: str) -> torch.Tensor:
    """Compute Gaussian RBF kernel for velocity smoothing."""
    if os.path.exists(save_path):
        logger.info(f"Loading cached kernel from {save_path}")
        return torch.from_numpy(np.load(save_path))

    logger.info("Computing Gaussian RBF kernel...")
    H, W = len(lat), len(lon)
    N = H * W
    positions = [[lat[i], lon[j]] for i in range(H) for j in range(W)]

    kernel = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            dist = sum((positions[i][d] - positions[j][d]) ** 2 for d in range(2))
            kernel[i, j] = torch.exp(torch.tensor(-dist / 2.0))

    kernel_inv = torch.linalg.inv(kernel).numpy()
    np.save(save_path, kernel_inv)
    logger.info(f"Kernel saved to {save_path}")
    return torch.from_numpy(kernel_inv)


def compute_temporal_derivative(past_states: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Estimate du/dt using cubic spline interpolation."""
    t = time_steps.flatten().float() * 6
    flat = past_states.view(past_states.shape[0], past_states.shape[1], -1)
    coeffs = natural_cubic_spline_coeffs(t, flat)
    spline = NaturalCubicSpline(coeffs)
    return spline.derivative(t[-1]).view(
        -1, past_states.shape[2], past_states.shape[3], past_states.shape[4]
    )


def fit_velocity_field(
    time_idx_loader: DataLoader,
    time_loader: DataLoader,
    full_data: torch.Tensor,
    data_loader: DataLoader,
    num_years: int,
    num_variables: int,
    kernel: torch.Tensor,
    save_path: str,
    device: torch.device = torch.device("cpu"),
    optim_steps: int = 200,
    max_batches: int = -1,
) -> torch.Tensor:
    """Fit initial velocity fields by solving the inverse advection problem.

    Args:
        max_batches: If > 0, only process this many batches (for dev_run).
    """
    if os.path.exists(save_path):
        logger.info(f"Loading cached velocity from {save_path}")
        return torch.from_numpy(np.load(save_path))

    logger.info(f"Fitting velocity fields ({num_years} years, {optim_steps} steps/timestep)...")
    H, W = GRID_H, GRID_W
    all_velocities = []

    for batch_idx, (idx_steps, time_steps, batch) in enumerate(
        zip(time_idx_loader, time_loader, data_loader)
    ):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        past_indices = [idx_steps[0].item() - i for i in range(3)]
        past_indices.reverse()
        past_time = torch.tensor([time_steps[0].item() - i for i in range(3)][::-1]).to(device)

        past_states = torch.stack([
            full_data[j].view(num_years, -1, num_variables, H, W) for j in past_indices
        ]).view(num_years, 3, -1, H, W).to(device)

        delta_u = compute_temporal_derivative(past_states, past_time)
        current_state = batch[0].to(device).view(num_years, 1, num_variables, H, W)

        vel_model = VelocityOptimizer(num_years, num_variables, H, W).to(device)
        optimizer = optim.Adam(vel_model.parameters(), lr=2.0)
        best_loss = float("inf")
        best_vx, best_vy = None, None

        for step in range(optim_steps):
            optimizer.zero_grad()
            advection, v_x, v_y = vel_model(current_state)

            vx_flat = v_x.view(num_years, num_variables, -1, 1)
            vy_flat = v_y.view(num_years, num_variables, -1, 1)
            kernel_exp = kernel.expand(num_years, num_variables, kernel.shape[0], kernel.shape[1])
            reg_x = torch.matmul(vx_flat.transpose(2, 3), torch.matmul(kernel_exp, vx_flat)).mean()
            reg_y = torch.matmul(vy_flat.transpose(2, 3), torch.matmul(kernel_exp, vy_flat)).mean()

            loss = (
                torch.nn.functional.mse_loss(delta_u, advection.squeeze(1))
                + 1e-7 * (reg_x + reg_y)
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_vx = v_x.detach().clone()
                best_vy = v_y.detach().clone()

            loss.backward()
            optimizer.step()

        velocity = torch.cat([best_vx, best_vy], dim=1).unsqueeze(0)
        all_velocities.append(velocity)

        if batch_idx % 50 == 0:
            logger.info(f"  Velocity batch {batch_idx}: loss={best_loss:.6f}")

    result = torch.cat(all_velocities, dim=0)
    np.save(save_path, result.detach().numpy())
    logger.info(f"Velocity saved to {save_path} (shape: {result.shape})")
    return result
