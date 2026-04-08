"""
Unit tests for derivative instruments.

Verify payoff functions produce correct values at known points.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.derivatives import (
    VanillaOption, DigitalOption, BullCallSpread, BearPutSpread,
    Straddle, Strangle, Butterfly, VarianceSwap, MarginSwap,
    build_instrument_catalog,
)


class TestVanillaOption:
    def test_call_itm(self):
        call = VanillaOption(strike=0.02, option_type="call")
        delta = np.array([0.05])
        assert call.payoff(delta) == pytest.approx(0.03)

    def test_call_otm(self):
        call = VanillaOption(strike=0.02, option_type="call")
        delta = np.array([0.01])
        assert call.payoff(delta) == pytest.approx(0.0)

    def test_call_atm(self):
        call = VanillaOption(strike=0.02, option_type="call")
        delta = np.array([0.02])
        assert call.payoff(delta) == pytest.approx(0.0)

    def test_put_itm(self):
        put = VanillaOption(strike=-0.02, option_type="put")
        delta = np.array([-0.05])
        assert put.payoff(delta) == pytest.approx(0.03)

    def test_put_otm(self):
        put = VanillaOption(strike=-0.02, option_type="put")
        delta = np.array([0.01])
        assert put.payoff(delta) == pytest.approx(0.0)

    def test_notional(self):
        call = VanillaOption(strike=0.02, option_type="call", notional=100)
        delta = np.array([0.05])
        assert call.payoff(delta) == pytest.approx(3.0)

    def test_vectorized(self):
        call = VanillaOption(strike=0.02, option_type="call")
        deltas = np.array([-0.05, 0.0, 0.02, 0.05, 0.10])
        expected = np.array([0.0, 0.0, 0.0, 0.03, 0.08])
        np.testing.assert_array_almost_equal(call.payoff(deltas), expected)


class TestDigitalOption:
    def test_digital_call_itm(self):
        d = DigitalOption(strike=0.02, option_type="call", payout=1.0)
        assert d.payoff(np.array([0.05])) == pytest.approx(1.0)

    def test_digital_call_otm(self):
        d = DigitalOption(strike=0.02, option_type="call", payout=1.0)
        assert d.payoff(np.array([0.01])) == pytest.approx(0.0)

    def test_digital_put(self):
        d = DigitalOption(strike=-0.02, option_type="put", payout=5.0)
        assert d.payoff(np.array([-0.05])) == pytest.approx(5.0)


class TestStraddle:
    def test_positive_movement(self):
        s = Straddle(strike=0.0)
        assert s.payoff(np.array([0.05])) == pytest.approx(0.05)

    def test_negative_movement(self):
        s = Straddle(strike=0.0)
        assert s.payoff(np.array([-0.05])) == pytest.approx(0.05)

    def test_no_movement(self):
        s = Straddle(strike=0.0)
        assert s.payoff(np.array([0.0])) == pytest.approx(0.0)

    def test_symmetry(self):
        s = Straddle(strike=0.0)
        deltas = np.array([-0.03, 0.03])
        payoffs = s.payoff(deltas)
        assert payoffs[0] == pytest.approx(payoffs[1])


class TestStrangle:
    def test_large_positive(self):
        s = Strangle(k_call=0.03, k_put=-0.03)
        assert s.payoff(np.array([0.05])) == pytest.approx(0.02)

    def test_large_negative(self):
        s = Strangle(k_call=0.03, k_put=-0.03)
        assert s.payoff(np.array([-0.05])) == pytest.approx(0.02)

    def test_inside_range(self):
        s = Strangle(k_call=0.03, k_put=-0.03)
        assert s.payoff(np.array([0.01])) == pytest.approx(0.0)
        assert s.payoff(np.array([-0.01])) == pytest.approx(0.0)


class TestBullCallSpread:
    def test_below_low_strike(self):
        b = BullCallSpread(k_low=0.01, k_high=0.05)
        assert b.payoff(np.array([0.0])) == pytest.approx(0.0)

    def test_between_strikes(self):
        b = BullCallSpread(k_low=0.01, k_high=0.05)
        assert b.payoff(np.array([0.03])) == pytest.approx(0.02)

    def test_above_high_strike(self):
        b = BullCallSpread(k_low=0.01, k_high=0.05)
        assert b.payoff(np.array([0.10])) == pytest.approx(0.04)  # max profit

    def test_max_profit_capped(self):
        b = BullCallSpread(k_low=0.01, k_high=0.05)
        # Max profit should be k_high - k_low = 0.04
        payoff_at_10 = b.payoff(np.array([0.10]))
        payoff_at_05 = b.payoff(np.array([0.05]))
        assert payoff_at_10 == pytest.approx(payoff_at_05)


class TestButterfly:
    def test_at_center(self):
        bf = Butterfly(k_low=-0.02, k_mid=0.0, k_high=0.02)
        assert bf.payoff(np.array([0.0])) == pytest.approx(0.02)

    def test_at_wings(self):
        bf = Butterfly(k_low=-0.02, k_mid=0.0, k_high=0.02)
        assert bf.payoff(np.array([-0.02])) == pytest.approx(0.0)
        assert bf.payoff(np.array([0.02])) == pytest.approx(0.0)

    def test_outside_wings(self):
        bf = Butterfly(k_low=-0.02, k_mid=0.0, k_high=0.02)
        assert bf.payoff(np.array([0.05])) == pytest.approx(0.0)
        assert bf.payoff(np.array([-0.05])) == pytest.approx(0.0)


class TestVarianceSwap:
    def test_set_strike(self):
        vs = VarianceSwap(window=10)
        data = np.random.randn(100) * 0.02
        vs.set_strike_from_data(data)
        assert vs.strike_var > 0

    def test_payoff_shape(self):
        vs = VarianceSwap(strike_var=0.001, window=10)
        delta = np.random.randn(50) * 0.02
        payoffs = vs.payoff(delta)
        assert len(payoffs) == 50


class TestCatalog:
    def test_catalog_builds(self):
        catalog = build_instrument_catalog()
        assert len(catalog) > 10

    def test_all_payoffs_work(self):
        catalog = build_instrument_catalog()
        delta = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
        for name, deriv in catalog.items():
            if isinstance(deriv, VarianceSwap):
                # Needs longer input
                payoff = deriv.payoff(np.random.randn(100) * 0.02)
            elif isinstance(deriv, MarginSwap):
                payoff = deriv.payoff(delta)
            else:
                payoff = deriv.payoff(delta)
            assert isinstance(payoff, np.ndarray), f"Failed for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
