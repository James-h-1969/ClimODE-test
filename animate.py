"""Animate ClimODE T2m predictions as a heatmap over Earth.

Runs the ODE solver forward from an initial state, saves each timestep,
and produces an animation showing temperature evolution.

Usage:
    CUDA_VISIBLE_DEVICES=0 python animate.py --model_path Models/ClimODE_global_euler_epoch139.pt --data_root ../data_npz
"""

import argparse
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import (
    GRID_H,
    GRID_W,
    VARIABLES,
    YEARS_PER_GROUP,
    create_dataloaders,
    compute_gaussian_kernel,
    fit_velocity_field,
    load_all_variables,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="../data_npz")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=3, help="Number of consecutive batches to animate")
    parser.add_argument("--output", type=str, default="climode_t2m_animation.gif")
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = load_all_variables(args.data_root)
    loaders = create_dataloaders(data, batch_size=args.batch_size)

    num_years_test = data["test_data"].shape[1]
    num_vars = len(VARIABLES)
    lat = data["lat"]
    lon = data["lon"]
    max_vals = data["max_vals"]
    min_vals = data["min_vals"]

    # T2m is index 2 in VARIABLES
    t2m_idx = VARIABLES.index("t2m")
    t2m_max = max_vals[t2m_idx]
    t2m_min = min_vals[t2m_idx]

    # Fit/load test velocities
    kernel_path = os.path.join(args.data_root, "kernel.npy")
    kernel = compute_gaussian_kernel(lat, lon, kernel_path)
    vel_path = os.path.join(args.data_root, "vel_test.npy")
    vel_test = fit_velocity_field(
        loaders["time_idx"], loaders["time_loader"], data["test_data"],
        loaders["test_loader"], num_years_test, num_vars, kernel,
        vel_path, optim_steps=200,
    )

    # Load model
    model = torch.load(args.model_path, map_location=device)
    model.eval()

    # Collect predictions and ground truth
    print("Running ODE solver...")
    pred_frames = []
    truth_frames = []
    yr = 0  # use first test year

    with torch.no_grad():
        for entry, (time_steps, batch) in enumerate(
            zip(loaders["time_loader"], loaders["test_loader"])
        ):
            if entry >= args.num_batches:
                break

            initial = batch[0, yr:yr+1].to(device).view(1, 1, num_vars, GRID_H, GRID_W)
            past_vel = vel_test[entry, yr:yr+1].view(1, 2 * num_vars, GRID_H, GRID_W).to(device)

            model.update_state(
                past_vel,
                data["const_info"].to(device),
                data["lat_map"].to(device),
                data["lon_map"].to(device),
            )

            t = time_steps.float().to(device).flatten()
            mean_pred, _, _ = model(t, initial.squeeze(1))

            # Denormalize T2m for each step
            for step in range(len(time_steps)):
                pred_t2m = mean_pred[step, 0, t2m_idx].cpu().numpy()
                truth_t2m = batch[step, yr, t2m_idx].numpy()
                pred_frames.append(pred_t2m * (t2m_max - t2m_min) + t2m_min)
                truth_frames.append(truth_t2m * (t2m_max - t2m_min) + t2m_min)

    print(f"Collected {len(pred_frames)} frames")

    # Create animation
    print("Creating animation...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 4),
                             subplot_kw={"projection": ccrs.PlateCarree()})

    vmin = min(f.min() for f in truth_frames)
    vmax = max(f.max() for f in truth_frames)
    err_max = max(np.abs(p - t).max() for p, t in zip(pred_frames, truth_frames))

    lon2d, lat2d = np.meshgrid(lon, lat)

    def animate_frame(i):
        for ax in axes:
            ax.clear()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_global()

        im0 = axes[0].pcolormesh(lon2d, lat2d, pred_frames[i],
                                  cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                                  transform=ccrs.PlateCarree())
        axes[0].set_title(f"ClimODE Prediction | t+{i*6}h")

        im1 = axes[1].pcolormesh(lon2d, lat2d, truth_frames[i],
                                  cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                                  transform=ccrs.PlateCarree())
        axes[1].set_title(f"ERA5 Ground Truth | t+{i*6}h")

        error = pred_frames[i] - truth_frames[i]
        im2 = axes[2].pcolormesh(lon2d, lat2d, error,
                                  cmap="bwr", vmin=-err_max, vmax=err_max,
                                  transform=ccrs.PlateCarree())
        axes[2].set_title(f"Error (Pred - Truth) | t+{i*6}h")

        return [im0, im1, im2]

    ani = animation.FuncAnimation(fig, animate_frame, frames=len(pred_frames),
                                  interval=1000 // args.fps, blit=False)

    ani.save(args.output, writer="pillow", fps=args.fps, dpi=100)
    print(f"Animation saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
