"""ClimODE model components for global weather forecasting.

Implements the physics-informed Neural ODE architecture from:
    Verma et al., "ClimODE: Climate and Weather Forecasting with
    Physics-informed Neural ODEs", ICLR 2024.

The model encodes the advection (continuity) equation:
    du/dt = -(v · ∇u) - u(∇ · v)
as a 2nd-order Neural ODE system where velocity acceleration is learned.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

logger = logging.getLogger(__name__)


class BoundaryPad(nn.Module):
    """Applies physics-appropriate boundary padding for spherical geometry.

    - Circular padding in longitude (east-west wraps around the globe)
    - Reflective padding in latitude (poles reflect)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reflect in lat (dim H), circular in lon (dim W)
        return F.pad(F.pad(x, (0, 0, 1, 1), "reflect"), (1, 1, 0, 0), "circular")


class ResidualBlock(nn.Module):
    """Residual convolutional block with boundary-aware padding.

    Uses LeakyReLU activation, BatchNorm, and dropout for regularization.
    Shortcut connection projects channels if in_channels != out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pad = BoundaryPad()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.3)
        self.drop = nn.Dropout(p=0.1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.bn1(self.conv1(self.pad(x))))
        h = self.activation(self.bn2(self.conv2(self.pad(h))))
        h = self.drop(h)
        return h + self.shortcut(x)


class ResNet2D(nn.Module):
    """Stack of residual blocks forming the local convolution network.

    Args:
        in_channels: Number of input channels.
        layer_counts: List of block counts per stage, e.g. [5, 3, 2].
        hidden_sizes: List of channel sizes per stage, e.g. [128, 64, 10].
    """

    def __init__(self, in_channels: int, layer_counts: List[int], hidden_sizes: List[int]):
        super().__init__()
        stages = []
        ch_in = in_channels
        for num_blocks, ch_out in zip(layer_counts, hidden_sizes):
            blocks = [ResidualBlock(ch_in, ch_out)]
            blocks += [ResidualBlock(ch_out, ch_out) for _ in range(num_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            ch_in = ch_out
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class SelfAttentionConv(nn.Module):
    """Global self-attention via CNN-based Key/Query/Value projections.

    Captures long-range spatial dependencies (teleconnections) that local
    convolutions cannot reach. Uses downsampled keys/values for efficiency.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = in_channels // 2
        qk_channels = in_channels // 8
        pad = BoundaryPad()

        self.query = nn.Sequential(
            pad, nn.Conv2d(in_channels, mid, 3, stride=1, padding=0), nn.LeakyReLU(0.3),
            pad, nn.Conv2d(mid, qk_channels, 3, stride=1, padding=0), nn.LeakyReLU(0.3),
            pad, nn.Conv2d(qk_channels, qk_channels, 3, stride=1, padding=0),
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, stride=2, padding=0), nn.LeakyReLU(0.3),
            nn.Conv2d(mid, qk_channels, 3, stride=2, padding=0), nn.LeakyReLU(0.3),
            nn.Conv2d(qk_channels, qk_channels, 3, stride=1, padding=0),
        )
        self.value = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, stride=2, padding=0), nn.LeakyReLU(0.3),
            nn.Conv2d(mid, out_channels, 3, stride=2, padding=0), nn.LeakyReLU(0.3),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=0),
        )
        self.post_map = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        q = self.query(x.float()).flatten(-2, -1)  # (B, qk, H*W)
        k = self.key(x.float()).flatten(-2, -1)    # (B, qk, H'*W')
        v = self.value(x.float()).flatten(-2, -1)  # (B, out, H'*W')

        attn = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)  # (B, H*W, H'*W')
        out = torch.bmm(v, attn.transpose(1, 2))  # (B, out, H*W)
        out = self.post_map(out.view(B, self.out_channels, H, W))
        return out


class VelocityOptimizer(nn.Module):
    """Learnable initial velocity field for the ODE system.

    Optimized per-timestep to satisfy the continuity equation given
    observed state changes. Parameters are v_x and v_y velocity components.

    Args:
        num_years: Number of years (batch dimension for velocity).
        num_channels: Number of weather variables (K).
        H: Grid height (latitude points).
        W: Grid width (longitude points).
    """

    def __init__(self, num_years: int, num_channels: int, H: int, W: int):
        super().__init__()
        self.v_x = nn.Parameter(torch.randn(num_years, 1, num_channels, H, W))
        self.v_y = nn.Parameter(torch.randn(num_years, 1, num_channels, H, W))

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute advection from current velocity estimate.

        Args:
            data: Weather state tensor (num_years, 1, K, H, W).

        Returns:
            advection: The advection term -(v·∇u + u∇·v).
            v_x: X-component of velocity.
            v_y: Y-component of velocity.
        """
        u_y = torch.gradient(data, dim=3)[0]  # ∂u/∂y (latitude)
        u_x = torch.gradient(data, dim=4)[0]  # ∂u/∂x (longitude)
        div_v = torch.gradient(self.v_y, dim=3)[0] + torch.gradient(self.v_x, dim=4)[0]
        advection = self.v_x * u_x + self.v_y * u_y + data * div_v
        return advection, self.v_x, self.v_y


class ClimODE(nn.Module):
    """ClimODE global weather forecasting model.

    Implements a 2nd-order Neural ODE system:
        du/dt = -(v · ∇u) - u(∇ · v)     [advection, physics-based]
        dv/dt = f_θ(u, ∇u, v, ψ)          [learned velocity acceleration]

    The emission/noise network provides uncertainty quantification:
        u_obs ~ N(u_advection + μ, σ²)

    Args:
        num_variables: Number of weather variables (K=5 for z500, t850, t2m, u10, v10).
        solver: ODE solver method (e.g. "euler", "dopri5").
        use_attention: Whether to include global attention network.
        use_uncertainty: Whether to include emission/noise network.
    """

    def __init__(
        self,
        num_variables: int = 5,
        solver: str = "euler",
        use_attention: bool = True,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.K = num_variables
        self.solver = solver
        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty

        # Input channels for velocity network:
        # t_emb/24(1) + day_emb(2) + seas_emb(2) + nabla_u(2K) + v(2K) + u(K) +
        # lat(1) + lon(1) + lsm(1) + oro(1) + pos_feats(6) + pos_time_ft(4×6=24)
        # = 5 + 5K + 34 = 64 when K=5
        input_channels = 5 + 5 * num_variables + 34
        logger.info(f"Velocity network input channels: {input_channels}")

        # Local convolution network for velocity acceleration
        self.vel_conv = ResNet2D(input_channels, [5, 3, 2], [128, 64, 2 * num_variables])

        # Global attention network
        if use_attention:
            self.vel_attn = SelfAttentionConv(input_channels, 2 * num_variables)
            self.gamma = nn.Parameter(torch.tensor([0.1]))

        # Emission/noise network for uncertainty
        if use_uncertainty:
            # t_cyc(4) + u(K) + pos_enc(10) + pos_time_ft(4×6=24) = 38 + K = 43 when K=5
            noise_input = 4 + num_variables + 10 + 24
            self.noise_net = ResNet2D(noise_input, [3, 2, 2], [128, 64, 2 * num_variables])

        # State holders (set via update_state before forward)
        self._past_velocity: Optional[torch.Tensor] = None
        self._const_info: Optional[torch.Tensor] = None
        self._lat_map: Optional[torch.Tensor] = None
        self._lon_map: Optional[torch.Tensor] = None

    def update_state(
        self,
        past_velocity: torch.Tensor,
        const_info: torch.Tensor,
        lat_map: torch.Tensor,
        lon_map: torch.Tensor,
    ) -> None:
        """Set the external state needed for ODE integration.

        Args:
            past_velocity: Initial velocity field (B, 2K, H, W).
            const_info: Constant fields tensor (1, 2, H, W) containing orography and land-sea mask.
            lat_map: Latitude map (1, 1, H, W) in degrees.
            lon_map: Longitude map (1, 1, H, W) in degrees.
        """
        self._past_velocity = past_velocity
        self._const_info = const_info
        self._lat_map = lat_map
        self._lon_map = lon_map

    def _build_spatiotemporal_features(self, batch_size: int, H: int, W: int, device: torch.device):
        """Pre-compute spatial features that are constant during ODE integration."""
        # Orography and land-sea mask
        self._oro = F.normalize(self._const_info[0, 0]).unsqueeze(0).expand(batch_size, -1, H, W)
        self._lsm = self._const_info[0, 1].unsqueeze(0).expand(batch_size, -1, H, W)

        # Lat/lon in radians
        self._lat_rad = self._lat_map.expand(batch_size, 1, H, W) * torch.pi / 180
        self._lon_rad = self._lon_map.expand(batch_size, 1, H, W) * torch.pi / 180

        # Spherical position features
        cos_lat = torch.cos(self._lat_rad)
        sin_lat = torch.sin(self._lat_rad)
        cos_lon = torch.cos(self._lon_rad)
        sin_lon = torch.sin(self._lon_rad)
        self._pos_feats = torch.cat(
            [cos_lat, cos_lon, sin_lat, sin_lon, sin_lat * cos_lon, sin_lat * sin_lon], dim=1
        )  # (B, 6, H, W)

        # Full position encoding for noise network
        self._pos_enc = torch.cat(
            [self._lat_rad, self._lon_rad, self._pos_feats, self._lsm, self._oro], dim=1
        )  # (B, 10, H, W)

    @staticmethod
    def _time_position_embedding(time_feats: torch.Tensor, pos_feats: torch.Tensor) -> torch.Tensor:
        """Compute outer product of time and position features.

        Args:
            time_feats: (B, T_channels, H, W)
            pos_feats: (B, P_channels, H, W)

        Returns:
            Joint embedding (B, T_channels * P_channels, H, W)
        """
        parts = [time_feats[:, i : i + 1] * pos_feats for i in range(time_feats.shape[1])]
        return torch.cat(parts, dim=1)

    def _pde(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Right-hand side of the ODE system: [dv/dt, du/dt].

        Args:
            t: Current time (scalar tensor).
            state: Concatenated [velocity(2K), quantity(K)] tensor (B, 3K, H, W).

        Returns:
            Time derivatives [dv, du] with same shape as state.
        """
        K = self.K
        H, W = state.shape[2], state.shape[3]

        # Split state into velocity and quantity
        v = state[:, :2 * K].float()  # (B, 2K, H, W)
        u = state[:, -K:].float()     # (B, K, H, W)

        # Time embeddings
        t_val = ((t * 100) % 24).view(1, 1, 1, 1).expand(u.shape[0], 1, H, W)
        sin_day = torch.sin(torch.pi * t_val / 12 - torch.pi / 2)
        cos_day = torch.cos(torch.pi * t_val / 12 - torch.pi / 2)
        sin_seas = torch.sin(torch.pi * t_val / (12 * 365) - torch.pi / 2)
        cos_seas = torch.cos(torch.pi * t_val / (12 * 365) - torch.pi / 2)
        day_emb = torch.cat([sin_day, cos_day], dim=1)
        seas_emb = torch.cat([sin_seas, cos_seas], dim=1)

        # Spatial gradients of quantity
        du_dx = torch.gradient(u, dim=3)[0]  # longitude gradient
        du_dy = torch.gradient(u, dim=2)[0]  # latitude gradient
        nabla_u = torch.cat([du_dx, du_dy], dim=1)  # (B, 2K, H, W)

        # Build input for velocity network
        t_cyc_emb = torch.cat([day_emb, seas_emb], dim=1)  # (B, 4, H, W)
        pos_time_ft = self._time_position_embedding(t_cyc_emb, self._pos_feats)  # (B, 24, H, W)

        vel_input = torch.cat([
            t_val / 24, day_emb, seas_emb,  # 5 channels
            nabla_u, v, u,                   # 2K + 2K + K = 5K channels
            self._lat_rad, self._lon_rad,    # 2 channels
            self._lsm, self._oro,            # 2 channels
            self._pos_feats,                 # 6 channels
            pos_time_ft,                     # 24 channels
        ], dim=1)  # Total: 5 + 5K + 34 = 64 when K=5

        # Velocity acceleration: dv/dt = f_conv + γ * f_attn
        dv = self.vel_conv(vel_input)
        if self.use_attention:
            dv = dv + self.gamma * self.vel_attn(vel_input)

        # Advection: du/dt = -(v_x · ∂u/∂x + v_y · ∂u/∂y) - u · (∂v_x/∂x + ∂v_y/∂y)
        v_x = v[:, :K]   # (B, K, H, W)
        v_y = v[:, K:]   # (B, K, H, W)
        transport = v_x * du_dx + v_y * du_dy
        compression = u * (torch.gradient(v_x, dim=3)[0] + torch.gradient(v_y, dim=2)[0])
        du = transport + compression

        return torch.cat([dv, du], dim=1)

    def _compute_uncertainty(
        self, time_steps: torch.Tensor, u_trajectory: torch.Tensor, H: int, W: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply emission/noise network for uncertainty quantification.

        Args:
            time_steps: Original time indices (T,).
            u_trajectory: ODE solution for u at 6-hourly steps (T, B, K, H, W).

        Returns:
            mean: Bias-corrected predictions (T, B, K, H, W).
            std: Prediction uncertainty (T, B, K, H, W).
        """
        T, B = u_trajectory.shape[0], u_trajectory.shape[1]

        # Time embeddings for each output step
        t_emb = (time_steps % 24).view(-1, 1, 1, 1, 1)
        sin_t = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2).expand(T, B, 1, H, W)
        cos_t = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2).expand(T, B, 1, H, W)
        sin_s = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2).expand(T, B, 1, H, W)
        cos_s = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2).expand(T, B, 1, H, W)

        # Expand position encoding for all timesteps
        pos_enc = self._pos_enc.unsqueeze(0).expand(T, B, -1, H, W).flatten(0, 1)  # (T*B, 10, H, W)
        t_cyc = torch.cat([sin_t, cos_t, sin_s, cos_s], dim=2).flatten(0, 1)  # (T*B, 4, H, W)

        # Position-time features (exclude lat/lon raw from pos_enc for interaction)
        pos_time_ft = self._time_position_embedding(t_cyc, pos_enc[:, 2:-2])  # (T*B, 4*6, H, W)

        noise_input = torch.cat(
            [t_cyc, u_trajectory.flatten(0, 1), pos_enc, pos_time_ft], dim=1
        )

        noise_out = self.noise_net(noise_input).view(T, B, 2 * self.K, H, W)
        mean = u_trajectory + noise_out[:, :, :self.K]
        std = F.softplus(noise_out[:, :, self.K:])

        return mean, std

    def forward(
        self,
        time_steps: torch.Tensor,
        initial_state: torch.Tensor,
        atol: float = 0.1,
        rtol: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the ClimODE forward pass.

        Args:
            time_steps: Time indices for the batch (T,), in units of 6-hour steps.
            initial_state: Initial weather state (B, K, H, W).
            atol: Absolute tolerance for ODE solver.
            rtol: Relative tolerance for ODE solver.

        Returns:
            mean: Predicted mean at 6-hourly intervals (T_out, B, K, H, W).
            std: Predicted std at 6-hourly intervals (T_out, B, K, H, W).
            raw_trajectory: Raw ODE solution without noise correction (T_out, B, K, H, W).
        """
        B, K, H, W = initial_state.shape
        assert K == self.K, f"Expected {self.K} variables, got {K}"
        assert self._past_velocity is not None, "Call update_state() before forward()"

        # Build spatial features
        self._build_spatiotemporal_features(B, H, W, initial_state.device)

        # Initial ODE state: [velocity(2K), quantity(K)]
        ode_state = torch.cat([self._past_velocity, initial_state], dim=1)  # (B, 3K, H, W)

        # Time grid for ODE integration (hourly resolution within 6-hour steps)
        init_time = time_steps[0].item() * 6
        final_time = time_steps[-1].item() * 6
        num_steps = int(final_time - init_time)
        t_grid = 0.01 * torch.linspace(init_time, final_time, steps=num_steps + 1).to(
            initial_state.device
        )

        logger.debug(
            f"ODE integration: t0={init_time}h, tf={final_time}h, "
            f"steps={num_steps}, solver={self.solver}"
        )

        # Solve ODE
        trajectory = odeint(
            self._pde, ode_state, t_grid, method=self.solver, atol=atol, rtol=rtol
        )  # (num_steps+1, B, 3K, H, W)

        # Extract quantity u from trajectory, subsample to 6-hourly
        u_full = trajectory[:, :, -K:].view(len(t_grid), B, K, H, W)
        u_6hourly = u_full[::6]  # Every 6th step = every 6 hours

        logger.debug(f"ODE output shape: {trajectory.shape}, 6-hourly samples: {u_6hourly.shape}")

        if self.use_uncertainty:
            mean, std = self._compute_uncertainty(time_steps, u_6hourly, H, W)
            return mean, std, u_6hourly
        else:
            return u_6hourly, torch.zeros_like(u_6hourly), u_6hourly
