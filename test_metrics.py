"""Unit tests for metrics.py"""

import numpy as np
import pytest
import torch

from metrics import compute_acc, compute_crps, compute_rmse, denormalize, latitude_weights


class TestLatitudeWeights:
    def test_equator_weight_highest(self):
        lat = np.array([-60, -30, 0, 30, 60])
        weights = latitude_weights(lat)
        # Equator (index 2) should have highest weight
        assert weights[2] == weights.max()

    def test_symmetry(self):
        lat = np.array([-60, -30, 0, 30, 60])
        weights = latitude_weights(lat)
        assert np.isclose(weights[0], weights[4])  # -60 == 60
        assert np.isclose(weights[1], weights[3])  # -30 == 30

    def test_mean_is_one(self):
        lat = np.linspace(-90, 90, 32)
        weights = latitude_weights(lat)
        assert np.isclose(weights.mean(), 1.0)

    def test_poles_lowest(self):
        lat = np.array([-90, -45, 0, 45, 90])
        weights = latitude_weights(lat)
        # Poles should have lowest weight (cos(90°) ≈ 0)
        assert weights[0] < weights[2]
        assert weights[4] < weights[2]


class TestDenormalize:
    def test_basic(self):
        data = torch.tensor([0.0, 0.5, 1.0])
        result = denormalize(data, max_val=100.0, min_val=0.0)
        np.testing.assert_allclose(result, [0.0, 50.0, 100.0])

    def test_with_offset(self):
        data = torch.tensor([0.0, 1.0])
        result = denormalize(data, max_val=300.0, min_val=200.0)
        np.testing.assert_allclose(result, [200.0, 300.0])

    def test_numpy_input(self):
        data = np.array([0.5])
        result = denormalize(data, max_val=10.0, min_val=0.0)
        assert result[0] == 5.0


class TestComputeRMSE:
    def test_perfect_prediction(self):
        lat = np.linspace(-90, 90, 32)
        lon = np.linspace(0, 360, 64)
        pred = torch.ones(5, 32, 64) * 0.5
        truth = torch.ones(5, 32, 64) * 0.5
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        rmse = compute_rmse(pred, truth, lat, lon, max_vals, min_vals)
        for var in rmse:
            assert rmse[var] == 0.0

    def test_nonzero_error(self):
        lat = np.linspace(-90, 90, 32)
        lon = np.linspace(0, 360, 64)
        pred = torch.ones(5, 32, 64) * 0.6
        truth = torch.ones(5, 32, 64) * 0.5
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        rmse = compute_rmse(pred, truth, lat, lon, max_vals, min_vals)
        for var in rmse:
            assert rmse[var] > 0.0
            # Error is 0.1 * 100 = 10 in physical units, RMSE should be ~10
            assert np.isclose(rmse[var], 10.0, atol=1.0)

    def test_returns_all_variables(self):
        lat = np.linspace(-90, 90, 32)
        lon = np.linspace(0, 360, 64)
        pred = torch.randn(5, 32, 64)
        truth = torch.randn(5, 32, 64)
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        rmse = compute_rmse(pred, truth, lat, lon, max_vals, min_vals)
        assert set(rmse.keys()) == {"z", "t", "t2m", "u10", "v10"}


class TestComputeACC:
    def test_perfect_correlation(self):
        lat = np.linspace(-90, 90, 32)
        lon = np.linspace(0, 360, 64)
        # Anomaly = pred - clim, if pred anomaly == truth anomaly, ACC = 1
        clim = np.zeros((5, 32, 64))
        pred = torch.randn(5, 32, 64)
        truth = pred.clone()  # Perfect prediction
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        acc = compute_acc(pred, truth, clim, lat, lon, max_vals, min_vals)
        for var in acc:
            assert np.isclose(acc[var], 1.0, atol=1e-5)

    def test_anticorrelation(self):
        lat = np.linspace(-90, 90, 32)
        lon = np.linspace(0, 360, 64)
        clim = np.zeros((5, 32, 64))
        pred = torch.randn(5, 32, 64)
        truth = -pred  # Opposite
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        acc = compute_acc(pred, truth, clim, lat, lon, max_vals, min_vals)
        for var in acc:
            assert np.isclose(acc[var], -1.0, atol=1e-5)

    def test_returns_all_variables(self):
        lat = np.linspace(-90, 90, 32)
        lon = np.linspace(0, 360, 64)
        clim = np.zeros((5, 32, 64))
        pred = torch.randn(5, 32, 64)
        truth = torch.randn(5, 32, 64)
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        acc = compute_acc(pred, truth, clim, lat, lon, max_vals, min_vals)
        assert set(acc.keys()) == {"z", "t", "t2m", "u10", "v10"}


class TestComputeCRPS:
    def test_perfect_prediction_low_crps(self):
        pred = torch.ones(5, 32, 64) * 0.5
        truth = torch.ones(5, 32, 64) * 0.5
        std = torch.ones(5, 32, 64) * 0.01  # Very small std
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        crps = compute_crps(pred, truth, std, max_vals, min_vals)
        for var in crps:
            assert crps[var] < 0.01  # Should be very small

    def test_large_error_high_crps(self):
        pred = torch.ones(5, 32, 64) * 0.9
        truth = torch.ones(5, 32, 64) * 0.1
        std = torch.ones(5, 32, 64) * 0.01
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        crps = compute_crps(pred, truth, std, max_vals, min_vals)
        for var in crps:
            assert crps[var] > 0.1  # Should be large

    def test_returns_all_variables(self):
        pred = torch.randn(5, 32, 64)
        truth = torch.randn(5, 32, 64)
        std = torch.ones(5, 32, 64) * 0.1
        max_vals = [100.0] * 5
        min_vals = [0.0] * 5
        crps = compute_crps(pred, truth, std, max_vals, min_vals)
        assert set(crps.keys()) == {"z", "t", "t2m", "u10", "v10"}
