import pytest

import numpy as np

from libpyhat.analytics import analytics

@pytest.fixture
def setUp():
    np.random.seed(seed=42)
    return np.random.random(25)

def test_band_minima(setUp):
    minidx, minvalue = analytics.band_minima(setUp)
    assert minidx == 10
    assert minvalue == pytest.approx(0.02058449)

@pytest.mark.parametrize("lower_bound, upper_bound, expected_idx, expected_val", [
                                            (0, 7, 6, 0.05808361),
                                            pytest.param(6, 1, 0, 0, marks=pytest.mark.xfail)]
)
def test_band_minima_bounds(lower_bound, upper_bound, expected_idx, expected_val, setUp):
    minidx, minvalue = analytics.band_minima(setUp, lower_bound, upper_bound)
    assert minidx == expected_idx
    assert minvalue == pytest.approx(expected_val)

def test_band_center(setUp):
    center, center_fit = analytics.band_center(setUp)
    assert center_fit[0] == max(center_fit)
    assert center_fit[-1] == min(center_fit)
    assert center[0] == 22

def test_band_area():
    x = np.arange(-2, 2, 0.1)
    y = x ** 2
    parabola = y
    area = analytics.band_area(parabola)
    assert area == [370.5]

@pytest.mark.parametrize("spectrum, expected_val", [
                                            (setUp(), 0.99447513),
                                            (np.ones(24), 1)]
)
def test_band_asymmetry(spectrum, expected_val):
    assymetry = analytics.band_asymmetry(spectrum)
    assert assymetry == pytest.approx(expected_val)
