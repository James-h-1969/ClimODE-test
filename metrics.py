"""Evaluation metrics for ClimODE global weather forecasting.

Implements latitude-weighted RMSE, ACC, and CRPS following WeatherBench conventions.
All metrics operate on denormalized (physical) values.
"""

import logging
from typing import List, Union

import numpy as np
import properscoring as ps
import torch
import xarray as xr

from data import VARIABLES

logger = logging.getLogger(__name__)


def latitude_weights(lat: np.ndarray) -> np.ndarray:
    """Compute latitude-based area weights (cosine weighting).

    Grid cells near the equator cover more area than those near the poles.
    Weights are normalized so their mean equals 1.
    """
    weights = np.cos(np.deg2rad(lat))
    weights /= weights.mean()
    return weights


def denormalize(
    data: Union[torch.Tensor, np.ndarray], max_val: float, min_val: float
) -> np.ndarray:
    """Convert from [0,1] normalized to physical units."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return data * (max_val - min_val) + min_val


def compute_rmse(
    pred: torch.Tensor,
    truth: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    max_vals: List[float],
    min_vals: List[float],
) -> dict[str, float]:
    """Compute latitude-weighted RMSE for each variable.

    Args:
        pred: Predicted state (K, H, W) in normalized [0,1] space.
        truth: Ground truth state (K, H, W) in normalized [0,1] space.
        lat: Latitude array.
        lon: Longitude array.
        max_vals: Per-variable max values for denormalization.
        min_vals: Per-variable min values for denormalization.

    Returns:
        Dictionary mapping variable name to RMSE value.
    """
    H, W = len(lat), len(lon)
    weights = latitude_weights(lat).reshape(H, 1)  # (H, 1) for broadcasting
    results = {}

    for idx, var in enumerate(VARIABLES):
        pred_phys = denormalize(pred[idx], max_vals[idx], min_vals[idx])
        truth_phys = denormalize(truth[idx], max_vals[idx], min_vals[idx])
        error = pred_phys - truth_phys
        rmse = np.sqrt(np.mean(error**2 * weights))
        results[var] = float(rmse)

    return results


def compute_acc(
    pred: torch.Tensor,
    truth: torch.Tensor,
    climatology: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    max_vals: List[float],
    min_vals: List[float],
) -> dict[str, float]:
    """Compute Anomaly Correlation Coefficient for each variable.

    ACC measures the correlation between predicted and true anomalies
    (deviations from climatology), weighted by latitude.

    Args:
        pred: Predicted state (K, H, W) in normalized space.
        truth: Ground truth state (K, H, W) in normalized space.
        climatology: Climatological mean (K, H, W) in normalized space.
        lat: Latitude array.
        lon: Longitude array.
        max_vals: Per-variable max values for denormalization.
        min_vals: Per-variable min values for denormalization.

    Returns:
        Dictionary mapping variable name to ACC value.
    """
    H, W = len(lat), len(lon)
    weights = latitude_weights(lat).reshape(H, 1).repeat(W, axis=1)  # (H, W)
    results = {}

    for idx, var in enumerate(VARIABLES):
        # Compute anomalies (subtract climatology before denormalization)
        pred_anom = denormalize(pred[idx] - climatology[idx], max_vals[idx], min_vals[idx])
        truth_anom = denormalize(truth[idx] - climatology[idx], max_vals[idx], min_vals[idx])

        # Remove mean (centered anomalies)
        pred_prime = pred_anom - np.mean(pred_anom)
        truth_prime = truth_anom - np.mean(truth_anom)

        # Weighted correlation
        numerator = np.sum(weights * pred_prime * truth_prime)
        denominator = np.sqrt(
            np.sum(weights * pred_prime**2) * np.sum(weights * truth_prime**2)
        )
        acc = numerator / (denominator + 1e-10)
        results[var] = float(acc)

    return results


def compute_crps(
    pred: torch.Tensor,
    truth: torch.Tensor,
    std: torch.Tensor,
    max_vals: List[float],
    min_vals: List[float],
) -> dict[str, float]:
    """Compute Continuous Ranked Probability Score for each variable.

    CRPS evaluates probabilistic forecasts by comparing the predicted
    Gaussian distribution against the observed value.

    Args:
        pred: Predicted mean (K, H, W) in normalized space.
        truth: Ground truth (K, H, W) in normalized space.
        std: Predicted standard deviation (K, H, W) in normalized space.
        max_vals: Per-variable max values.
        min_vals: Per-variable min values.

    Returns:
        Dictionary mapping variable name to mean CRPS value.
    """
    results = {}

    for idx, var in enumerate(VARIABLES):
        pred_np = pred[idx].detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred[idx]
        truth_np = truth[idx].detach().cpu().numpy() if isinstance(truth, torch.Tensor) else truth[idx]
        std_np = std[idx].detach().cpu().numpy() if isinstance(std, torch.Tensor) else std[idx]

        crps = ps.crps_gaussian(truth_np, mu=pred_np, sig=std_np)
        results[var] = float(np.mean(crps))

    return results


def evaluate_batch(
    pred: torch.Tensor,
    truth: torch.Tensor,
    std: torch.Tensor,
    climatology: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    max_vals: List[float],
    min_vals: List[float],
    lead_time_hours: int,
) -> dict[str, dict[str, float]]:
    """Evaluate a single prediction at a given lead time.

    Args:
        pred: Predicted state (K, H, W).
        truth: Ground truth (K, H, W).
        std: Predicted uncertainty (K, H, W).
        climatology: Climatological mean (K, H, W).
        lat, lon: Coordinate arrays.
        max_vals, min_vals: Normalization parameters.
        lead_time_hours: Lead time in hours for logging.

    Returns:
        Dictionary with 'rmse', 'acc', 'crps' sub-dictionaries.
    """
    rmse = compute_rmse(pred, truth, lat, lon, max_vals, min_vals)
    acc = compute_acc(pred, truth, climatology, lat, lon, max_vals, min_vals)
    crps = compute_crps(pred, truth, std, max_vals, min_vals)

    logger.debug(
        f"Lead {lead_time_hours:3d}h | "
        f"RMSE z={rmse['z']:.1f} t={rmse['t']:.2f} t2m={rmse['t2m']:.2f} | "
        f"ACC z={acc['z']:.4f} t={acc['t']:.4f}"
    )

    return {"rmse": rmse, "acc": acc, "crps": crps}
