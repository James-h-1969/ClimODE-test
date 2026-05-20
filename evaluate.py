"""Evaluation script for ClimODE global weather forecasting at 72hr lead time.

Evaluates a trained model on the test set (2017-2018) computing:
- Latitude-weighted RMSE
- Anomaly Correlation Coefficient (ACC)
- Continuous Ranked Probability Score (CRPS)

at lead times from 6h to 72h in 6h increments.

Usage:
    python evaluate.py --model_path checkpoints/ClimODE_global.pt --batch_size 13
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from data import (
    GRID_H,
    GRID_W,
    TEST_YEARS,
    VARIABLES,
    compute_gaussian_kernel,
    create_dataloaders,
    fit_velocity_field,
    load_all_variables,
    set_seed,
)
from metrics import evaluate_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClimODE at 72hr lead time")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--batch_size", type=int, default=13, help="13 for 72hr lead time")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--vel_path", type=str, default=None, help="Path to test velocity .npy")
    parser.add_argument("--vel_steps", type=int, default=200)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Lead time: {(args.batch_size - 1) * 6}h ({args.batch_size - 1} steps)")

    # Load data
    data = load_all_variables(args.data_root)
    loaders = create_dataloaders(data, batch_size=args.batch_size)

    num_years_test = len(TEST_YEARS)
    num_vars = len(VARIABLES)
    lat, lon = data["lat"], data["lon"]
    max_vals, min_vals = data["max_vals"], data["min_vals"]

    # Compute climatology from test set (mean over time)
    climatology = torch.mean(data["test_data"], dim=0).numpy()  # (num_years, K, H, W)
    logger.info(f"Climatology shape: {climatology.shape}")

    # Fit or load test velocities
    vel_path = args.vel_path or os.path.join(args.data_root, "vel_test.npy")
    kernel_path = os.path.join(args.data_root, "kernel.npy")
    kernel = compute_gaussian_kernel(lat, lon, kernel_path)

    vel_test = fit_velocity_field(
        loaders["time_idx"], loaders["time_loader"], data["test_data"],
        loaders["test_loader"], num_years_test, num_vars, kernel,
        vel_path, optim_steps=args.vel_steps,
    )

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    logger.info("Model loaded successfully")

    # Evaluation
    max_lead_steps = args.batch_size - 1  # 12 steps = 72h
    lead_times_hours = [(i + 1) * 6 for i in range(max_lead_steps)]

    # Accumulate results per lead time
    results = {
        "rmse": {var: [[] for _ in range(max_lead_steps)] for var in VARIABLES},
        "acc": {var: [[] for _ in range(max_lead_steps)] for var in VARIABLES},
        "crps": {var: [[] for _ in range(max_lead_steps)] for var in VARIABLES},
    }

    num_batches = 0
    logger.info("Starting evaluation...")

    with torch.no_grad():
        for entry, (time_steps, batch) in enumerate(
            zip(loaders["time_loader"], loaders["test_loader"])
        ):
            # Prepare inputs
            initial = batch[0].to(device).view(num_years_test, 1, num_vars, GRID_H, GRID_W)
            past_vel = vel_test[entry].view(num_years_test, 2 * num_vars, GRID_H, GRID_W).to(device)

            model.update_state(
                past_vel,
                data["const_info"].to(device),
                data["lat_map"].to(device),
                data["lon_map"].to(device),
            )

            t = time_steps.float().to(device).flatten()
            mean_pred, std_pred, _ = model(t, initial.squeeze(1))

            # Evaluate at each lead time for each test year
            for yr in range(num_years_test):
                for step in range(1, len(time_steps)):
                    if step > max_lead_steps:
                        break

                    lead_idx = step - 1
                    lead_hours = (step) * 6

                    metrics = evaluate_batch(
                        pred=mean_pred[step, yr].cpu(),
                        truth=batch[step, yr].cpu(),
                        std=std_pred[step, yr].cpu(),
                        climatology=climatology[yr],
                        lat=lat,
                        lon=lon,
                        max_vals=max_vals,
                        min_vals=min_vals,
                        lead_time_hours=lead_hours,
                    )

                    for var in VARIABLES:
                        results["rmse"][var][lead_idx].append(metrics["rmse"][var])
                        results["acc"][var][lead_idx].append(metrics["acc"][var])
                        results["crps"][var][lead_idx].append(metrics["crps"][var])

            num_batches += 1
            if num_batches % 25 == 0:
                logger.info(f"  Processed {num_batches} batches...")

    # Aggregate results
    logger.info(f"\nEvaluation complete ({num_batches} batches)")
    logger.info("=" * 80)
    logger.info(f"{'Lead (h)':<10} {'Variable':<8} {'RMSE (mean±std)':<22} "
                f"{'ACC (mean±std)':<22} {'CRPS (mean±std)':<22}")
    logger.info("-" * 80)

    summary = {}
    for step_idx, lead_h in enumerate(lead_times_hours):
        summary[f"{lead_h}h"] = {}
        for var in VARIABLES:
            rmse_vals = results["rmse"][var][step_idx]
            acc_vals = results["acc"][var][step_idx]
            crps_vals = results["crps"][var][step_idx]

            if not rmse_vals:
                continue

            rmse_mean, rmse_std = np.mean(rmse_vals), np.std(rmse_vals)
            acc_mean, acc_std = np.mean(acc_vals), np.std(acc_vals)
            crps_mean, crps_std = np.mean(crps_vals), np.std(crps_vals)

            summary[f"{lead_h}h"][var] = {
                "rmse_mean": float(rmse_mean),
                "rmse_std": float(rmse_std),
                "acc_mean": float(acc_mean),
                "acc_std": float(acc_std),
                "crps_mean": float(crps_mean),
                "crps_std": float(crps_std),
            }

            logger.info(
                f"{lead_h:<10} {var:<8} "
                f"{rmse_mean:>8.2f} ± {rmse_std:<8.2f}  "
                f"{acc_mean:>8.4f} ± {acc_std:<8.4f}  "
                f"{crps_mean:>8.4f} ± {crps_std:<8.4f}"
            )

    logger.info("=" * 80)

    # Highlight 72hr results
    if "72h" in summary:
        logger.info("\n72hr Lead Time Summary:")
        for var in VARIABLES:
            if var in summary["72h"]:
                s = summary["72h"][var]
                logger.info(
                    f"  {var}: RMSE={s['rmse_mean']:.2f}, "
                    f"ACC={s['acc_mean']:.4f}, CRPS={s['crps_mean']:.4f}"
                )

    # Save results
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
