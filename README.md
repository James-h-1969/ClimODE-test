# ClimODE Training Guide

Physics-informed Neural ODE for global weather forecasting (72hr lead time).
Reads preprocessed ERA5 data in ClimaX `.npz` format.

## Data Requirements

The data directory (same output as ClimaX `nc2np_equally_era5.py`) should look like:

```
data_npz/
├── train/          # 2006_0.npz ... 2015_7.npz (8 shards/year)
├── val/            # 2016_0.npz ... 2016_7.npz
├── test/           # 2017_0.npz ... 2018_7.npz
├── lat.npy
├── lon.npy
├── normalize_mean.npz
└── normalize_std.npz
```

Each `.npz` shard contains hourly data with keys like `geopotential_500`, `temperature_850`, `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, etc.

ClimODE uses 5 variables: Z500, T850, T2m, U10, V10 (subsampled to 6-hourly).

## Setup on Cloud Machine

```bash
# SSH in
ssh james.hocking@<machine-ip>

# Activate environment (or create one with the deps below)
conda activate climaX

# Required packages (if not already installed):
# pip install torch torchdiffeq torchcubicspline numpy xarray properscoring
```

## Quick Pipeline Test (Dev Run)

Verifies data loading, kernel computation, velocity fitting, and 1 training step:

```bash
cd ~/ClimODE-test
python train.py --data_root ../data_npz --dev_run
```

This loads only 1 year of data, fits velocity for 5 batches with 5 optim steps, and runs 1 epoch over 5 batches. Should complete in a few minutes.

## Full Training

Uses gradient accumulation to train on all 10 years (2006-2015) while fitting in 15GB GPU memory.
Processes 2 years at a time, accumulates gradients over 5 groups.

```bash
cd ~/ClimODE-test

# Delete old cached velocities (required when changing batch_size or years)
rm -f ../data_npz/vel_train.npy ../data_npz/vel_val.npy

# In tmux/screen so it survives SSH disconnect:
tmux new -s climode

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root ../data_npz \
  --solver euler \
  --batch_size 8 \
  --lr 0.0005 \
  --epochs 300 \
  --vel_steps 200 \
  --save_dir Models
```

Detach tmux: `Ctrl+B` then `D`. Reattach: `tmux attach -t climode`.

### Key Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--data_root` | `.` | Path to npz data directory |
| `--solver` | `euler` | ODE solver (euler, dopri5, rk4, midpoint) |
| `--batch_size` | `8` | Timesteps per batch (8 = 42hr lead time, paper default) |
| `--epochs` | `300` | Training epochs |
| `--lr` | `0.0005` | Learning rate |
| `--vel_steps` | `200` | Velocity optimization steps per timestep |
| `--save_dir` | `Models` | Checkpoint save directory |
| `--dev_run` | off | Quick pipeline test mode |

### GPU Usage

```bash
# Check available GPUs
nvidia-smi

# Use a specific GPU
CUDA_VISIBLE_DEVICES=3 python train.py --data_root ../data_npz ...
```

## Training Pipeline

1. **Data loading** — Reads npz shards, subsamples hourly→6-hourly, min-max normalizes
2. **Kernel computation** — Gaussian RBF kernel for velocity smoothing (cached to `kernel.npy`)
3. **Velocity fitting** — Solves inverse advection problem per timestep (cached to `vel_train.npy`, `vel_val.npy`)
4. **Training** — Neural ODE with NLL loss + L2 regularization
5. **Checkpointing** — Best model saved based on validation loss

Note: Steps 2-3 are expensive on first run but cached for subsequent runs.

## Evaluation

```bash
python evaluate.py --model_path Models/ClimODE_global_euler_epoch42.pt --data_root ../data_npz
```

Outputs latitude-weighted RMSE, ACC, and CRPS at 6h–72h lead times.
