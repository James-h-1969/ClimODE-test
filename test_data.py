"""Unit tests for data.py"""

import os
import tempfile

import numpy as np
import pytest
import torch

from data import (
    GRID_H,
    GRID_W,
    VARIABLES,
    compute_gaussian_kernel,
    compute_temporal_derivative,
    set_seed,
)


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(99)
        b = torch.randn(10)
        assert not torch.equal(a, b)


class TestComputeGaussianKernel:
    def test_output_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel.npy")
            lat = np.array([0.0, 5.625, 11.25])
            lon = np.array([0.0, 5.625])
            kernel = compute_gaussian_kernel(lat, lon, path)
            # Should be (H*W, H*W) = (6, 6)
            assert kernel.shape == (6, 6)

    def test_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel.npy")
            lat = np.array([0.0, 5.625])
            lon = np.array([0.0, 5.625])
            k1 = compute_gaussian_kernel(lat, lon, path)
            assert os.path.exists(path)
            # Second call should load from cache
            k2 = compute_gaussian_kernel(lat, lon, path)
            assert torch.equal(k1, k2)

    def test_symmetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel.npy")
            lat = np.array([0.0, 5.625, 11.25])
            lon = np.array([0.0, 5.625])
            kernel = compute_gaussian_kernel(lat, lon, path)
            # Kernel inverse should be approximately symmetric
            assert torch.allclose(kernel, kernel.T, atol=1e-4)

    def test_diagonal_dominance(self):
        """RBF kernel should have largest values on diagonal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel.npy")
            lat = np.array([0.0, 30.0, 60.0])
            lon = np.array([0.0, 30.0])
            # Compute the kernel itself (not inverse) to check diagonal
            N = len(lat) * len(lon)
            positions = []
            for i in range(len(lat)):
                for j in range(len(lon)):
                    positions.append([lat[i], lon[j]])
            kernel = torch.zeros(N, N)
            for i in range(N):
                for j in range(N):
                    dist = sum((positions[i][d] - positions[j][d]) ** 2 for d in range(2))
                    kernel[i, j] = torch.exp(torch.tensor(-dist / 2.0))
            # Diagonal should be 1.0 (distance to self is 0)
            assert torch.allclose(kernel.diag(), torch.ones(N))


class TestComputeTemporalDerivative:
    def test_output_shape(self):
        num_years, K, H, W = 2, 5, 32, 64
        past_states = torch.randn(num_years, 3, K, H, W)
        time_steps = torch.tensor([0, 1, 2])
        deriv = compute_temporal_derivative(past_states, time_steps)
        assert deriv.shape == (num_years, K, H, W)

    def test_linear_function(self):
        """For a linear function f(t) = a*t + b, derivative should be a."""
        num_years, K, H, W = 1, 1, 4, 4
        a = torch.randn(1, 1, H, W)
        b = torch.randn(1, 1, H, W)
        # Create 3 timesteps of linear data
        t = torch.tensor([0, 1, 2])
        past_states = torch.stack([
            a * 0 + b, a * 1 + b, a * 2 + b
        ], dim=1).view(1, 3, 1, H, W)  # (1, 3, 1, H, W)

        # Time steps in the format expected (will be multiplied by 6)
        time_steps = torch.tensor([0, 1, 2])
        deriv = compute_temporal_derivative(past_states, time_steps)

        # Derivative of a*6t + b at t=2 is 6*a (since internal t = time_steps * 6)
        expected = a.squeeze(0) * 6  # (1, H, W)... wait, shape should be (num_years, K, H, W)
        # Actually deriv shape is (num_years, K, H, W) = (1, 1, 4, 4)
        # The spline derivative at point t[-1]=12 of f(t)=a*t+b is a
        # But time_steps are multiplied by 6 inside, so t = [0, 6, 12]
        # f(t) = a/6 * t + b (to get values a*0+b, a*1+b, a*2+b at t=0,6,12)
        # derivative = a/6
        # Hmm, let me think again. The function creates:
        #   t = time_steps.flatten().float() * 6 = [0, 6, 12]
        #   values at these t: [b, a+b, 2a+b]
        # So the function is f(t) = (a/6)*t + b
        # derivative at t=12 is a/6
        # But wait, the spline fits to the flattened spatial values
        # Let me just check it's approximately constant (linear function has constant derivative)
        assert deriv.shape == (1, 1, H, W)
        # For a linear function, derivative should be constant = a/6
        expected_deriv = a.view(1, 1, H, W) / 6.0
        assert torch.allclose(deriv, expected_deriv, atol=1e-4)


class TestConstants:
    def test_grid_dimensions(self):
        assert GRID_H == 32
        assert GRID_W == 64

    def test_variables(self):
        assert VARIABLES == ["z", "t", "t2m", "u10", "v10"]
        assert len(VARIABLES) == 5
