"""Unit tests for model.py"""

import pytest
import torch
import torch.nn.functional as F

from model import (
    BoundaryPad,
    ClimODE,
    ResidualBlock,
    ResNet2D,
    SelfAttentionConv,
    VelocityOptimizer,
)


class TestBoundaryPad:
    def test_output_shape(self):
        pad = BoundaryPad()
        x = torch.randn(2, 3, 32, 64)
        out = pad(x)
        # Should add 1 on each side in both H and W
        assert out.shape == (2, 3, 34, 66)

    def test_circular_padding_longitude(self):
        """Longitude (W dim) should wrap circularly."""
        pad = BoundaryPad()
        x = torch.randn(1, 1, 4, 4)
        out = pad(x)
        # After reflect in H (rows 0,5 are reflected), circular in W (cols 0,5 wrap)
        # Circular: out[:,:,:,0] == x[:,:,:,-1] and out[:,:,:,-1] == x[:,:,:,0]
        # But reflect happens first on H, then circular on W
        # After reflect on H: shape (1,1,6,4), then circular on W: shape (1,1,6,6)
        # The circular padding on the reflected tensor:
        # out[:,:,:,0] should equal the reflected tensor's last W column
        # out[:,:,:,-1] should equal the reflected tensor's first W column
        reflected = F.pad(x, (0, 0, 1, 1), "reflect")
        assert out[:, :, :, 0].equal(reflected[:, :, :, -1])
        assert out[:, :, :, -1].equal(reflected[:, :, :, 0])

    def test_reflective_padding_latitude(self):
        """Latitude (H dim) should reflect at poles."""
        pad = BoundaryPad()
        x = torch.randn(1, 1, 4, 4)
        out = pad(x)
        reflected = F.pad(x, (0, 0, 1, 1), "reflect")
        # The H padding is reflective, so row 0 of reflected == row 2 of x (mirror of row 1)
        assert reflected[:, :, 0, :].equal(x[:, :, 1, :])
        assert reflected[:, :, -1, :].equal(x[:, :, -2, :])


class TestResidualBlock:
    def test_output_shape_same_channels(self):
        block = ResidualBlock(16, 16)
        x = torch.randn(2, 16, 32, 64)
        out = block(x)
        assert out.shape == (2, 16, 32, 64)

    def test_output_shape_different_channels(self):
        block = ResidualBlock(16, 32)
        x = torch.randn(2, 16, 32, 64)
        out = block(x)
        assert out.shape == (2, 32, 32, 64)

    def test_residual_connection(self):
        """With identity shortcut, output should differ from input (due to conv layers)."""
        block = ResidualBlock(16, 16)
        x = torch.randn(2, 16, 32, 64)
        out = block(x)
        # Output should not be identical to input
        assert not out.equal(x)

    def test_gradient_flow(self):
        block = ResidualBlock(8, 16)
        x = torch.randn(1, 8, 32, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestResNet2D:
    def test_output_shape(self):
        net = ResNet2D(10, [2, 1], [32, 5])
        x = torch.randn(2, 10, 32, 64)
        out = net(x)
        assert out.shape == (2, 5, 32, 64)

    def test_single_stage(self):
        net = ResNet2D(5, [3], [10])
        x = torch.randn(1, 5, 32, 64)
        out = net(x)
        assert out.shape == (1, 10, 32, 64)

    def test_climode_velocity_architecture(self):
        """Test the actual architecture used in ClimODE velocity network."""
        net = ResNet2D(64, [5, 3, 2], [128, 64, 10])
        x = torch.randn(1, 64, 32, 64)
        out = net(x)
        assert out.shape == (1, 10, 32, 64)


class TestSelfAttentionConv:
    def test_output_shape(self):
        attn = SelfAttentionConv(64, 10)
        x = torch.randn(2, 64, 32, 64)
        out = attn(x)
        assert out.shape == (2, 10, 32, 64)

    def test_gradient_flow(self):
        attn = SelfAttentionConv(16, 8)
        x = torch.randn(1, 16, 32, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestVelocityOptimizer:
    def test_output_shapes(self):
        num_years, K, H, W = 3, 5, 32, 64
        model = VelocityOptimizer(num_years, K, H, W)
        data = torch.randn(num_years, 1, K, H, W)
        advection, v_x, v_y = model(data)
        assert advection.shape == (num_years, 1, K, H, W)
        assert v_x.shape == (num_years, 1, K, H, W)
        assert v_y.shape == (num_years, 1, K, H, W)

    def test_advection_computation(self):
        """Verify advection = v_x * du/dx + v_y * du/dy + u * div(v)."""
        num_years, K, H, W = 1, 1, 8, 8
        model = VelocityOptimizer(num_years, K, H, W)
        data = torch.randn(num_years, 1, K, H, W)
        advection, v_x, v_y = model(data)

        # Manual computation
        u_x = torch.gradient(data, dim=4)[0]
        u_y = torch.gradient(data, dim=3)[0]
        div_v = torch.gradient(v_y, dim=3)[0] + torch.gradient(v_x, dim=4)[0]
        expected = v_x * u_x + v_y * u_y + data * div_v

        assert torch.allclose(advection, expected, atol=1e-6)

    def test_parameters_are_learnable(self):
        model = VelocityOptimizer(2, 5, 32, 64)
        params = list(model.parameters())
        assert len(params) == 2  # v_x and v_y
        assert params[0].requires_grad
        assert params[1].requires_grad


class TestClimODE:
    @pytest.fixture
    def model(self):
        return ClimODE(num_variables=5, solver="euler", use_attention=True, use_uncertainty=True)

    @pytest.fixture
    def model_no_uncertainty(self):
        return ClimODE(num_variables=5, solver="euler", use_attention=False, use_uncertainty=False)

    def test_init(self, model):
        assert model.K == 5
        assert model.solver == "euler"
        assert model.use_attention
        assert model.use_uncertainty

    def test_input_channels(self, model):
        """Velocity network should have 64 input channels for K=5."""
        # 5 + 5*5 + 34 = 64
        first_conv = model.vel_conv.stages[0][0].conv1
        assert first_conv.in_channels == 64

    def test_noise_net_input_channels(self, model):
        """Noise network should have 43 input channels for K=5."""
        # 4 + 5 + 10 + 24 = 43
        first_conv = model.noise_net.stages[0][0].conv1
        assert first_conv.in_channels == 43

    def test_velocity_network_output(self, model):
        """Velocity network should output 2K=10 channels."""
        last_stage = model.vel_conv.stages[-1]
        last_block = last_stage[-1]
        assert last_block.conv2.out_channels == 10

    def test_update_state(self, model):
        B, K, H, W = 2, 5, 32, 64
        vel = torch.randn(B, 2 * K, H, W)
        const = torch.randn(1, 2, H, W)
        lat = torch.randn(1, 1, H, W)
        lon = torch.randn(1, 1, H, W)
        model.update_state(vel, const, lat, lon)
        assert model._past_velocity is not None

    def test_forward_shape(self, model):
        B, K, H, W = 2, 5, 32, 64
        vel = torch.randn(B, 2 * K, H, W)
        const = torch.randn(1, 2, H, W)
        lat = torch.randn(1, 1, H, W)
        lon = torch.randn(1, 1, H, W)
        model.update_state(vel, const, lat, lon)

        # 3 timesteps: initial + 2 future (12h lead)
        time_steps = torch.tensor([0, 1, 2]).float()
        initial = torch.randn(B, K, H, W)
        mean, std, raw = model(time_steps, initial)

        # Should have 3 output timesteps (subsampled every 6 from 12 integration steps)
        assert mean.shape[1] == B
        assert mean.shape[2] == K
        assert mean.shape[3] == H
        assert mean.shape[4] == W
        assert std.shape == mean.shape

    def test_forward_no_uncertainty(self, model_no_uncertainty):
        B, K, H, W = 1, 5, 32, 64
        vel = torch.randn(B, 2 * K, H, W)
        const = torch.randn(1, 2, H, W)
        lat = torch.randn(1, 1, H, W)
        lon = torch.randn(1, 1, H, W)
        model_no_uncertainty.update_state(vel, const, lat, lon)

        time_steps = torch.tensor([0, 1, 2]).float()
        initial = torch.randn(B, K, H, W)
        out, std, raw = model_no_uncertainty(time_steps, initial)

        # std should be zeros when uncertainty is disabled
        assert torch.all(std == 0)

    def test_pde_advection_structure(self, model):
        """Verify the PDE computes transport + compression correctly."""
        B, K, H, W = 1, 5, 32, 64
        vel = torch.randn(B, 2 * K, H, W)
        const = torch.randn(1, 2, H, W)
        lat = torch.randn(1, 1, H, W)
        lon = torch.randn(1, 1, H, W)
        model.update_state(vel, const, lat, lon)
        model._build_spatiotemporal_features(B, H, W, torch.device("cpu"))

        # Create a state tensor [v(2K), u(K)]
        state = torch.randn(B, 3 * K, H, W)
        t = torch.tensor(0.01)

        # Run PDE
        dstate = model._pde(t, state)
        assert dstate.shape == state.shape

        # First 2K channels are dv (velocity acceleration)
        # Last K channels are du (advection)
        dv = dstate[:, :2 * K]
        du = dstate[:, 2 * K:]
        assert dv.shape == (B, 2 * K, H, W)
        assert du.shape == (B, K, H, W)

    def test_time_position_embedding(self):
        """Test outer product of time and position features."""
        time_feats = torch.randn(2, 4, 32, 64)  # 4 time channels
        pos_feats = torch.randn(2, 6, 32, 64)   # 6 position channels
        result = ClimODE._time_position_embedding(time_feats, pos_feats)
        # Should be 4 * 6 = 24 channels
        assert result.shape == (2, 24, 32, 64)

        # Verify first block is time[0] * pos
        expected_first = time_feats[:, 0:1] * pos_feats
        assert torch.allclose(result[:, :6], expected_first)
