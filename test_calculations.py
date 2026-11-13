"""Проверки размерностей, спектральной энергии и правила выбора порога."""

import numpy as np
import pytest

from cnc_data import SAMPLE_RATE, signal_features
from experiment import operating_threshold


def test_constant_signal() -> None:
    values = signal_features(np.tile([1.0, -2.0, 0.0], (4096, 1))).reshape(8, 3)
    np.testing.assert_array_equal(values[0], [1, 2, 0])
    np.testing.assert_array_equal(values[[1, 2, 4, 5, 6, 7]], 0)
    assert np.isnan(values[3]).all()


def test_sinusoid_energy() -> None:
    t = np.arange(8192) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * 200 * t)
    result = signal_features(np.column_stack((signal, signal, signal))).reshape(8, 3)
    np.testing.assert_allclose(result[0], np.sqrt(0.5), atol=0.001)
    np.testing.assert_allclose(result[1], np.sqrt(0.5), atol=0.001)
    np.testing.assert_allclose(result[3], -1.5, atol=0.003)
    np.testing.assert_allclose(result[5], 0.5, atol=0.002)
    np.testing.assert_allclose(result[[4, 6, 7]], 0, atol=0.001)


def test_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="N×3"):
        signal_features(np.zeros((100, 2)))


def test_missing_channel() -> None:
    values = np.zeros((1024, 3))
    values[0, 1] = np.nan
    result = signal_features(values).reshape(8, 3)
    assert np.isnan(result[:, 1]).all()
    assert result[0, 0] == 0


def test_operating_threshold() -> None:
    assert operating_threshold(np.array([0, 0, 1, 1]), np.array([0.02, 0.05, 0.1, 0.2])) == 0.1
