from __future__ import annotations

import numpy as np

from src.swis_august import _opdi_for_pair


def test_opdi_identical_distribution_is_zero():
    grid = np.geomspace(320, 2100, 5)
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    js, hellinger, wasserstein = _opdi_for_pair(p, p, grid, min_valid_points=5)
    assert js == 0
    assert hellinger == 0
    assert wasserstein == 0


def test_opdi_different_distribution_is_larger_than_identical():
    grid = np.geomspace(320, 2100, 5)
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    q = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    js, hellinger, wasserstein = _opdi_for_pair(p, q, grid, min_valid_points=5)
    assert js > 0
    assert hellinger > 0
    assert wasserstein > 0


def test_opdi_invalid_distribution_returns_nan():
    grid = np.geomspace(320, 2100, 5)
    p = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    q = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    js, hellinger, wasserstein = _opdi_for_pair(p, q, grid, min_valid_points=5)
    assert np.isnan(js)
    assert np.isnan(hellinger)
    assert np.isnan(wasserstein)
