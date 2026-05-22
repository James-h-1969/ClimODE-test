"""Training script for ClimODE global weather forecasting.

Trains the model to predict 42hr (7 steps × 6h) ahead on ERA5 5.625° data.
Uses gradient accumulation over year groups to fit 10 years on a 15GB GPU.

Usage:
    python train.py --solver euler --batch_size 8 --lr 0.0005 --epochs 300
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

from data import (
    GRID_H,
    GRID_W,
    VARIABLES,
    YEARS_PER_GROUP,
    compute_gaussian_kernel,
    create_dataloaders,
    fit_velocity_field,
    load_all_variables,
    set_seed,
)
from model import ClimODE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SOLVERS = ["euler", "dopri5", "dopri8", "rk4", "midpoint", "adaptive_heun"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ClimODE for global forecasting")
    parser.add_argument("--solver", type=str, default="euler", choices=SOLVERS)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8, help="8 for 42hr lead time (paper default)")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--l2_lambda", type=float, default=0.001)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default="Models")
    parser.add_argument("--vel_steps", type=int, default=200, help="Velocity optimization steps")
    parser.add_argument("--dev_run", action="store_true", help="Quick pipeline test: 1 year, 1 epoch, 5 batches")
    return parser.parse_args()


def nll_loss(
    mean: torch.Tensor, std: torch.Tensor, truth: torch.Tensor, var_coeff: float
) -> torch.Tensor:
    """Negative log-likelihood loss with variance regularization."""
    dist = torch.distributions.Normal(mean, std + 1e-3)
    nll = -dist.log_prob(truth).mean()
    var_penalty = var_coeff * (std**2).sum()
    return nll + var_penalty


def train_epoch(
    model: ClimODE,
    loaders: dict,
    vel_train: torch.Tensor,
    const_info: torch.Tensor,
    lat_map: torch.Tensor,
    lon_map: torch.Tensor,
    optimizer: optim.Optimizer,
    var_coeff: float,
    l2_lambda: float,
    device: torch.device,
    num_years: int,
    num_vars: int,
    max_batches: int = -1,
) -> float:
    """Run one training epoch with gradient accumulation over year groups."""
    model.train()
    total_loss = 0.0
    num_groups = (num_years + YEARS_PER_GROUP - 1) // YEARS_PER_GROUP

    for entry, (time_steps, batch) in enumerate(
        zip(loaders["time_loader"], loaders["train_loader"])
    ):
        if max_batches > 0 and entry >= max_batches:
            break

        optimizer.zero_grad()
        batch_loss = 0.0

        # Gradient accumulation over year groups
        for g in range(num_groups):
            yr_start = g * YEARS_PER_GROUP
            yr_end = min(yr_start + YEARS_PER_GROUP, num_years)
            n_yrs = yr_end - yr_start

            # Slice year dimension
            initial = batch[0, yr_start:yr_end].to(device).view(n_yrs, 1, num_vars, GRID_H, GRID_W)
            past_vel = vel_train[entry, yr_start:yr_end].view(n_yrs, 2 * num_vars, GRID_H, GRID_W).to(device)
            target = batch[:, yr_start:yr_end].float().to(device)

            model.update_state(past_vel, const_info.to(device), lat_map.to(device), lon_map.to(device))

            t = time_steps.float().to(device).flatten()
            mean, std, _ = model(t, initial.squeeze(1))

            loss = nll_loss(mean, std, target, var_coeff) / num_groups
            loss.backward()
            batch_loss += loss.item()

        # L2 regularization (added once, not per group)
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        l2_loss = l2_lambda * l2_norm
        l2_loss.backward()
        batch_loss += l2_loss.item()

        if torch.isnan(torch.tensor(batch_loss)):
            logger.error(f"NaN loss at batch {entry}. Stopping.")
            sys.exit(1)

        optimizer.step()
        total_loss += batch_loss

        if entry % 20 == 0:
            logger.debug(f"  Batch {entry}: loss={batch_loss:.4f}")

    return total_loss


def validate(
    model: ClimODE,
    loaders: dict,
    vel_val: torch.Tensor,
    const_info: torch.Tensor,
    lat_map: torch.Tensor,
    lon_map: torch.Tensor,
    var_coeff: float,
    device: torch.device,
    num_years: int,
    num_vars: int,
    max_batches: int = -1,
) -> float:
    """Run validation (no gradient accumulation needed — val has 1 year)."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for entry, (time_steps, batch) in enumerate(
            zip(loaders["time_loader"], loaders["val_loader"])
        ):
            if max_batches > 0 and entry >= max_batches:
                break

            initial = batch[0].to(device).view(num_years, 1, num_vars, GRID_H, GRID_W)
            past_vel = vel_val[entry].view(num_years, 2 * num_vars, GRID_H, GRID_W).to(device)

            model.update_state(
                past_vel, const_info.to(device), lat_map.to(device), lon_map.to(device)
            )

            t = time_steps.float().to(device).flatten()
            mean, std, _ = model(t, initial.squeeze(1))
            loss = nll_loss(mean, std, batch.float().to(device), var_coeff)

            if torch.isnan(loss):
                logger.error(f"NaN validation loss at batch {entry}. Stopping.")
                sys.exit(1)

            total_loss += loss.item()

    return total_loss


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: solver={args.solver}, batch_size={args.batch_size}, "
                f"lr={args.lr}, epochs={args.epochs}")
    logger.info(f"Lead time: {(args.batch_size - 1) * 6}h ({args.batch_size - 1} steps)")

    # Load data
    data = load_all_variables(args.data_root, dev_run=args.dev_run)
    loaders = create_dataloaders(data, batch_size=args.batch_size)

    num_years_train = data["train_data"].shape[1]
    num_years_val = data["val_data"].shape[1]
    num_vars = len(VARIABLES)

    logger.info(f"Training: {num_years_train} years, gradient accumulation over groups of {YEARS_PER_GROUP}")

    # Compute kernel and fit velocities
    kernel_path = os.path.join(args.data_root, "kernel.npy")
    kernel = compute_gaussian_kernel(data["lat"], data["lon"], kernel_path)

    max_vel_batches = 5 if args.dev_run else -1
    vel_steps = 5 if args.dev_run else args.vel_steps

    logger.info("Fitting training velocities...")
    vel_train = fit_velocity_field(
        loaders["time_idx"], loaders["time_loader"], data["train_data"],
        loaders["train_loader"], num_years_train, num_vars, kernel,
        os.path.join(args.data_root, "vel_train.npy"),
        optim_steps=vel_steps, max_batches=max_vel_batches,
    )

    logger.info("Fitting validation velocities...")
    vel_val = fit_velocity_field(
        loaders["time_idx"], loaders["time_loader"], data["val_data"],
        loaders["val_loader"], num_years_val, num_vars, kernel,
        os.path.join(args.data_root, "vel_val.npy"),
        optim_steps=vel_steps, max_batches=max_vel_batches,
    )

    # Create model
    model = ClimODE(
        num_variables=num_vars, solver=args.solver, use_attention=True, use_uncertainty=True
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # Optimizer and scheduler
    epochs = 1 if args.dev_run else args.epochs
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        epoch_start = time.time()

        var_coeff = 0.001 if epoch == 0 else 2 * scheduler.get_last_lr()[0]

        train_loss = train_epoch(
            model, loaders, vel_train, data["const_info"], data["lat_map"], data["lon_map"],
            optimizer, var_coeff, args.l2_lambda, device, num_years_train, num_vars,
            max_batches=max_vel_batches if args.dev_run else -1,
        )

        val_loss = validate(
            model, loaders, vel_val, data["const_info"], data["lat_map"], data["lon_map"],
            var_coeff, device, num_years_val, num_vars,
            max_batches=max_vel_batches if args.dev_run else -1,
        )

        scheduler.step()
        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_path = os.path.join(
                args.save_dir, f"ClimODE_global_{args.solver}_epoch{epoch}.pt"
            )
            torch.save(model, save_path)
            logger.info(f"  New best model saved: {save_path}")

    logger.info(f"Training complete. Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
