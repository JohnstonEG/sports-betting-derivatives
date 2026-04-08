"""
Synthetic derivative instruments on sports betting markets.

Each derivative is defined by its payoff function relative to line movement
(delta_ip = close_implied_prob - open_implied_prob).

Tier 1: Vanilla options (calls, puts)
Tier 2: Spread strategies (bull/bear spreads, straddles, strangles)
Tier 3: Exotic instruments (variance swaps, margin swaps, digital options)
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from abc import ABC, abstractmethod


class Derivative(ABC):
    """Base class for all synthetic derivatives."""

    @abstractmethod
    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        """
        Compute payoff given realized line movements.

        Parameters
        ----------
        delta_ip : np.ndarray
            Realized line movements (close_ip - open_ip).

        Returns
        -------
        np.ndarray
            Payoff for each observation.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the instrument."""
        pass

    def __repr__(self):
        return self.description()


# ================================================================
# TIER 1: VANILLA OPTIONS
# ================================================================

@dataclass
class VanillaOption(Derivative):
    """
    Option on line movement.

    A call pays max(delta_ip - K, 0): profits when the closing line
    moves MORE than K in the home direction (probability increases).

    A put pays max(K - delta_ip, 0): profits when the closing line
    moves LESS than K (or reverses).

    Parameters
    ----------
    strike : float
        Strike in probability space. E.g., K=0.02 means the option
        is in-the-money if the line moves by more than 2pp.
    option_type : str
        "call" or "put".
    notional : float
        Notional amount (scales payoff).
    """
    strike: float = 0.02
    option_type: str = "call"
    notional: float = 1.0

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            return self.notional * np.maximum(delta_ip - self.strike, 0)
        else:
            return self.notional * np.maximum(self.strike - delta_ip, 0)

    def description(self) -> str:
        return (f"Vanilla {self.option_type.upper()} | "
                f"K={self.strike:.3f} | notional={self.notional:.0f}")


@dataclass
class DigitalOption(Derivative):
    """
    Binary/digital option on line movement.

    Pays a fixed amount if delta_ip crosses the strike.
    Digital call: pays 1 if delta_ip > K.
    Digital put: pays 1 if delta_ip < K.
    """
    strike: float = 0.02
    option_type: str = "call"
    payout: float = 1.0

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            return self.payout * (delta_ip > self.strike).astype(float)
        else:
            return self.payout * (delta_ip < self.strike).astype(float)

    def description(self) -> str:
        return (f"Digital {self.option_type.upper()} | "
                f"K={self.strike:.3f} | payout={self.payout:.2f}")


# ================================================================
# TIER 2: SPREAD STRATEGIES
# ================================================================

@dataclass
class BullCallSpread(Derivative):
    """
    Long call at K1, short call at K2 (K2 > K1).

    Profits from moderate upward line movement.
    Max profit = K2 - K1 (capped).
    """
    k_low: float = 0.01
    k_high: float = 0.05
    notional: float = 1.0

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        long_call = np.maximum(delta_ip - self.k_low, 0)
        short_call = np.maximum(delta_ip - self.k_high, 0)
        return self.notional * (long_call - short_call)

    def description(self) -> str:
        return (f"Bull Call Spread | K1={self.k_low:.3f}, K2={self.k_high:.3f}")


@dataclass
class BearPutSpread(Derivative):
    """
    Long put at K1, short put at K2 (K2 < K1).

    Profits from moderate downward line movement.
    """
    k_high: float = -0.01
    k_low: float = -0.05
    notional: float = 1.0

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        long_put = np.maximum(self.k_high - delta_ip, 0)
        short_put = np.maximum(self.k_low - delta_ip, 0)
        return self.notional * (long_put - short_put)

    def description(self) -> str:
        return (f"Bear Put Spread | K1={self.k_high:.3f}, K2={self.k_low:.3f}")


@dataclass
class Straddle(Derivative):
    """
    Long call + long put at the same strike.

    Profits from ANY large line movement, regardless of direction.
    This is a pure volatility play.
    """
    strike: float = 0.0
    notional: float = 1.0

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        call = np.maximum(delta_ip - self.strike, 0)
        put = np.maximum(self.strike - delta_ip, 0)
        return self.notional * (call + put)

    def description(self) -> str:
        return f"Straddle | K={self.strike:.3f}"


@dataclass
class Strangle(Derivative):
    """
    Long OTM call at K_high + long OTM put at K_low.

    Cheaper than straddle but needs larger movement to profit.
    """
    k_call: float = 0.02
    k_put: float = -0.02
    notional: float = 1.0

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        call = np.maximum(delta_ip - self.k_call, 0)
        put = np.maximum(self.k_put - delta_ip, 0)
        return self.notional * (call + put)

    def description(self) -> str:
        return f"Strangle | K_call={self.k_call:.3f}, K_put={self.k_put:.3f}"


@dataclass
class Butterfly(Derivative):
    """
    Long call at K1, short 2 calls at K2, long call at K3.
    
    Profits if delta_ip lands near K2. A bet on LOW volatility.
    """
    k_low: float = -0.02
    k_mid: float = 0.0
    k_high: float = 0.02
    notional: float = 1.0

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        c1 = np.maximum(delta_ip - self.k_low, 0)
        c2 = np.maximum(delta_ip - self.k_mid, 0)
        c3 = np.maximum(delta_ip - self.k_high, 0)
        return self.notional * (c1 - 2 * c2 + c3)

    def description(self) -> str:
        return (f"Butterfly | K1={self.k_low:.3f}, K2={self.k_mid:.3f}, "
                f"K3={self.k_high:.3f}")


# ================================================================
# TIER 3: EXOTIC / VOLATILITY INSTRUMENTS
# ================================================================

@dataclass
class VarianceSwap(Derivative):
    """
    Variance swap on line movements.

    Pays realized_variance - strike_variance.
    Long position profits when actual volatility exceeds the
    implied/expected volatility.

    In practice: the strike is set to the historical variance
    from a training period, and the realized variance is computed
    from a test window.
    """
    strike_var: float = 0.0  # set from training data
    notional: float = 1.0
    window: int = 50  # number of observations in the realized window

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        """
        For backtesting: computes rolling realized variance and
        returns the swap payoff at each point.
        """
        if len(delta_ip) < self.window:
            return np.zeros(len(delta_ip))

        # Rolling realized variance
        realized_var = np.array([
            np.var(delta_ip[max(0, i - self.window):i], ddof=1)
            if i >= self.window else np.nan
            for i in range(len(delta_ip))
        ])

        payoff = self.notional * (realized_var - self.strike_var)
        payoff[np.isnan(payoff)] = 0
        return payoff

    def set_strike_from_data(self, training_delta_ip: np.ndarray):
        """Set the strike variance from training data."""
        self.strike_var = np.var(training_delta_ip, ddof=1)

    def description(self) -> str:
        return (f"Variance Swap | K_var={self.strike_var:.6f} | "
                f"window={self.window}")


@dataclass
class MarginSwap(Derivative):
    """
    Swap on margin (overround) change.

    Pays realized_margin_change - expected_margin_change.
    Allows taking a view on whether bookmakers will tighten or
    widen their margins.
    """
    strike_margin_change: float = 0.0
    notional: float = 1.0

    def payoff(self, delta_margin: np.ndarray) -> np.ndarray:
        """Note: input is delta_margin, not delta_ip."""
        return self.notional * (delta_margin - self.strike_margin_change)

    def description(self) -> str:
        return f"Margin Swap | K_Δm={self.strike_margin_change:.4f}"


@dataclass
class CustomDerivative(Derivative):
    """
    User-defined derivative with arbitrary payoff function.

    Parameters
    ----------
    payoff_func : callable
        Function: np.ndarray → np.ndarray
    name : str
        Human-readable name.
    """
    payoff_func: Callable = None
    name: str = "Custom"

    def payoff(self, delta_ip: np.ndarray) -> np.ndarray:
        if self.payoff_func is None:
            raise ValueError("Must provide payoff_func")
        return self.payoff_func(delta_ip)

    def description(self) -> str:
        return f"Custom: {self.name}"


# ================================================================
# INSTRUMENT CATALOG
# ================================================================

def build_instrument_catalog(
    strikes: list[float] = None,
) -> dict[str, Derivative]:
    """
    Build a catalog of instruments to test.

    Returns a dict mapping instrument name → Derivative object.
    """
    if strikes is None:
        strikes = [0.01, 0.02, 0.03, 0.05]

    catalog = {}

    # Vanilla options
    for k in strikes:
        catalog[f"call_K{k:.2f}"] = VanillaOption(strike=k, option_type="call")
        catalog[f"put_K{k:.2f}"] = VanillaOption(strike=-k, option_type="put")

    # Digital options
    for k in strikes:
        catalog[f"digital_call_K{k:.2f}"] = DigitalOption(strike=k, option_type="call")
        catalog[f"digital_put_K{k:.2f}"] = DigitalOption(strike=-k, option_type="put")

    # Spreads
    catalog["bull_spread_1_5"] = BullCallSpread(k_low=0.01, k_high=0.05)
    catalog["bear_spread_1_5"] = BearPutSpread(k_high=-0.01, k_low=-0.05)

    # Volatility plays
    catalog["straddle_ATM"] = Straddle(strike=0.0)
    catalog["strangle_2pp"] = Strangle(k_call=0.02, k_put=-0.02)
    catalog["strangle_5pp"] = Strangle(k_call=0.05, k_put=-0.05)
    catalog["butterfly_2pp"] = Butterfly(k_low=-0.02, k_mid=0.0, k_high=0.02)

    # Variance swap (strike set later from data)
    catalog["var_swap_50"] = VarianceSwap(window=50)
    catalog["var_swap_200"] = VarianceSwap(window=200)

    # Margin swap
    catalog["margin_swap"] = MarginSwap()

    return catalog
