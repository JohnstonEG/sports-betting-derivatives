# Methodology: Synthetic Derivatives on Sports Betting Markets

## 1. The Underlying Asset

In traditional finance, derivatives are written on underlying assets (stocks, bonds, commodities). Here, the "underlying" is the **implied probability** extracted from bookmaker odds.

### Implied Probability Extraction

Given decimal odds $d$, the raw implied probability is:

$$p = \frac{1}{d}$$

For a two-outcome market (e.g., basketball), the sum of implied probabilities exceeds 1:

$$p_{home} + p_{away} = \frac{1}{d_{home}} + \frac{1}{d_{away}} > 1$$

The excess is the **overround** (or vig/margin):

$$\text{margin} = p_{home} + p_{away} - 1$$

Normalized (vig-free) probabilities:

$$\hat{p}_{home} = \frac{p_{home}}{p_{home} + p_{away}}$$

### The "Return" on the Underlying

The line movement — the change in implied probability from opening to closing — is our analog of a financial return:

$$\Delta p = p_{close} - p_{open}$$

This is what our derivatives are written on. Empirically (N = 2,522,019):
- Mean: −0.0013 (slight drift away from home team)
- Std: 0.0526
- Skewness: 0.171 (right-skewed)
- Excess kurtosis: 5.46 (fat-tailed)

The negative mean reflects systematic line correction away from home favorites — consistent with "public money" initially inflating home probabilities, followed by sharp-money correction.

## 2. Why Bachelier, Not Black-Scholes

A critical methodological point: we use the **Bachelier (1900) normal model**, not the Black-Scholes-Merton lognormal model, for parametric pricing.

**The reason is structural.** BSM assumes the underlying follows geometric Brownian motion (GBM), which produces lognormally-distributed prices that are strictly positive. This is appropriate for stock prices, which cannot go below zero. However, our underlying — line movement $\Delta p$ — can be negative (lines move in either direction). Applying BSM would require log-transforming negative values, which is undefined.

The Bachelier model assumes arithmetic Brownian motion:

$$dS = \sigma \, dW$$

where $S$ is the underlying (here, $\Delta p$), and prices options on normally-distributed underlyings.

### Bachelier Call Price

$$C = \sigma\sqrt{T} \left[ d \cdot \Phi(d) + \phi(d) \right]$$

where:
- $d = \frac{F - K}{\sigma\sqrt{T}}$
- $F$ = forward price (mean of $\Delta p$ distribution)
- $K$ = strike
- $\sigma$ = volatility of line movements
- $T = 1$ (normalized per event)
- $\Phi(\cdot)$ = standard normal CDF
- $\phi(\cdot)$ = standard normal PDF

### Bachelier Put Price

$$P = \sigma\sqrt{T} \left[ -d \cdot \Phi(-d) + \phi(d) \right]$$

### Bachelier Greeks

- **Delta**: $\Delta_{call} = \Phi(d)$, $\Delta_{put} = \Phi(d) - 1$
- **Gamma**: $\Gamma = \frac{\phi(d)}{\sigma\sqrt{T}}$
- **Vega**: $\nu = \sqrt{T} \cdot \phi(d)$

The comparison between Bachelier and empirical prices is a core result. Where they diverge reveals the pricing impact of non-normality.

### Empirical Mispricing Pattern

At K = 0.02: Bachelier overprices by 17.6% (too much density in the center)
At K = 0.05: Roughly correct (+2.6%)
At K = 0.10: Bachelier underprices by 59.9% (misses the fat tails)

The crossover around K = 0.05 is where the normal density intersects the empirical leptokurtic density.

## 3. Distributional Analysis

### Fat Tails

With excess kurtosis of 5.46, the line-movement distribution has substantially heavier tails than a Gaussian. The Jarque-Bera statistic of 3.14 million (p = 0) makes this unambiguous. Practically, this means:
- A 3σ line movement (|Δp| > 0.158) occurs ~0.6% of the time — roughly 3× more often than under normality
- Extreme movements (|Δp| > 0.2) are driven by injury news, weather, lineup changes, and sharp-money surges

### Sport-Level Heterogeneity

Line movement properties vary dramatically by sport:

| Sport | σ | Kurtosis | Interpretation |
|-------|---|----------|----------------|
| Football (soccer) | 0.056 | 4.09 | Most normal — deep liquidity |
| Hockey | 0.040 | 10.28 | Low vol, very fat tails — thin markets |
| Floorball | 0.022 | 191.21 | Extreme tails — single sharp bettors move lines |
| Rugby union | 0.062 | 7.04 | High vol, moderate tails |
| American football | 0.050 | 3.93 | Near-normal — efficient US market |

The variation suggests that a single parametric model cannot capture all sports; sport-specific calibration is needed.

### Normality Testing

Three tests all reject normality at any conventional level:
- **Jarque-Bera**: Tests joint skewness = 0 and kurtosis = 3
- **Shapiro-Wilk**: W = 0.94, p = 1.21e-40
- **Anderson-Darling**: Statistic = 44,012 (far exceeds all critical values)

## 4. Derivative Instruments

### Tier 1: Vanilla Options

**Call on line movement:**
$$\text{Payoff}_{call} = \max(\Delta p - K, 0)$$

A call with $K = 0.02$ pays off when the closing implied probability exceeds the opening by more than 2 percentage points. Empirical price: 0.0102, ITM probability: 26.1%.

**Put on line movement:**
$$\text{Payoff}_{put} = \max(K - \Delta p, 0)$$

### Tier 2: Volatility Instruments

**Straddle (ATM):**
$$\text{Payoff}_{straddle} = |\Delta p|$$

Pure exposure to the magnitude of line movement. Empirical price: 0.0353, always ITM (any nonzero movement produces a payoff).

**Strangle:**
$$\text{Payoff}_{strangle} = \max(\Delta p - K_{call}, 0) + \max(K_{put} - \Delta p, 0)$$

Dead zone between strikes; cheaper but needs larger movements. The 2pp strangle (K = ±0.02) has 54.6% ITM probability.

**Butterfly:**
$$\text{Payoff}_{butterfly} = \max(\Delta p - K_1, 0) - 2\max(\Delta p - K_2, 0) + \max(\Delta p - K_3, 0)$$

Profits when lines don't move much — a short-volatility position.

### Tier 3: Variance Swaps

$$\text{Payoff}_{var} = N \cdot (\sigma^2_{realized} - \sigma^2_{strike})$$

Pure variance exposure. The strike is calibrated from historical data; the realized variance is computed over a rolling window.

## 5. Pricing

### Empirical (Nonparametric)

1. Estimate the density of $\Delta p$ using kernel density estimation (Silverman bandwidth)
2. Draw $M = 50{,}000$ samples from the KDE
3. Compute payoff for each draw
4. Price = mean payoff

$$\hat{V} = \frac{1}{M} \sum_{i=1}^{M} f(\Delta p_i), \quad \Delta p_i \sim \hat{g}(\Delta p)$$

where $\hat{g}$ is the KDE estimate.

### Implied Volatility

Given an empirical price, we can back out the Bachelier implied volatility via Newton-Raphson:

$$\sigma_{n+1} = \sigma_n - \frac{C^{Bach}(\sigma_n) - C^{emp}}{\nu(\sigma_n)}$$

where $\nu$ is the Bachelier vega. If implied vol exceeds historical vol at deep OTM strikes, this indicates the market "knows" about fat tails even under a normal model — an implied volatility smile/smirk.

## 6. Backtesting

### Rolling Window Protocol

1. **Training window** (5,000 observations): Estimate the $\Delta p$ distribution via KDE
2. **Price**: Set fair value from training-period Monte Carlo
3. **Test window** (500 observations): Observe realized $\Delta p$ values
4. **P&L**: Mean realized payoff − fair price
5. **Roll forward** and repeat

With 2.52M observations and step size 500, this produces 5,039 evaluation periods.

### Performance Metrics

- **Sharpe Ratio**: $SR = \frac{\bar{r}}{\sigma_r}$ (annualization not applicable in event-time)
- **Sortino Ratio**: $\frac{\bar{r}}{\sigma_{downside}}$
- **Maximum Drawdown**: Largest peak-to-trough decline in cumulative P&L
- **Hit Rate**: Fraction of positive-P&L periods
- **Bootstrap CI**: 5,000-resample confidence intervals on the Sharpe ratio

### Results

All standalone derivative strategies produce negative Sharpe ratios (best: call K=0.02 at −0.129), consistent with semi-strong efficiency. The market prices line-movement risk correctly on average; there is no systematic arbitrage from buying or selling synthetic derivatives.

## 7. Portfolio Optimization

### Mean-Variance (Markowitz)

$$\min_w \quad w'\Sigma w \quad \text{s.t.} \quad w'\mu \geq \mu^*, \quad \sum w_i = 1, \quad w_i \in [-0.5, 2.0]$$

### CVaR Optimization

$$\min_w \quad \text{CVaR}_\alpha(w'r) \quad \text{s.t.} \quad w'\mu \geq \mu^*, \quad \sum w_i = 1$$

where $\text{CVaR}_{0.05} = -E[r \mid r \leq \text{VaR}_{0.05}]$.

### Frontier Expansion Result

The derivative-augmented efficient frontier lies above the spot-only frontier:
- Spot-only optimal: Sharpe = 0.279
- With derivatives: Sharpe = 0.361 (+29.4%)
- Optimal allocation: long underdog spot (1.71), short straddle (−0.05), short strangle (−0.06), long call (0.05)

The value of derivatives is not standalone alpha but portfolio risk management — selling volatility (straddle/strangle shorts) while expressing directional views through spot positions and calls.

## 8. PASPA Regime Analysis

The 2018 repeal of PASPA provides a natural experiment in market structure:

| Parameter | Pre-PASPA | Post-PASPA (ex COVID) | Change |
|-----------|-----------|----------------------|--------|
| σ (volatility) | 0.041 | 0.057 | +36.5% |
| Kurtosis | 6.72 | 4.77 | −1.96 |
| Skewness | 0.115 | 0.197 | +0.082 |
| Straddle price | 0.0264 | 0.0396 | +50.0% |

**Interpretation**: US legalization increased the volume and velocity of information flowing into betting markets, raising baseline volatility. Simultaneously, the increased participation reduced the frequency of extreme mispricings (lower kurtosis). More liquidity → higher routine vol, fewer tail events.

**Derivative pricing implication**: The Bachelier model becomes more accurate post-PASPA. At K=0.05, mispricing drops from 18% to 4.3%. As markets become more efficient, parametric models become better approximations — the informational content of the non-normal tail shrinks.

## 9. Bookmaker Microstructure

Sharp books (Pinnacle, Bet-in-Asia) and recreational books (888sport, bet365) exhibit distinct line-movement signatures:

- **Sharp books**: Higher σ (0.058), lower kurtosis (4.25), negative mean drift (−0.004). Lines move more, more predictably, and systematically away from the home team as informed bettors correct public bias.
- **Recreational books**: Lower σ (0.051), higher kurtosis (6.11), near-zero drift (−0.001). Lines move less on average but with occasional extreme jumps — likely when accumulated sharp-money pressure finally forces corrections on stale lines.

This is consistent with Levitt (2004) and the broader market microstructure literature on informed vs. uninformed order flow.

## References

- Bachelier, L. (1900). "Théorie de la spéculation." Annales Scientifiques de l'École Normale Supérieure.
- Black, F. and Scholes, M. (1973). "The pricing of options and corporate liabilities." Journal of Political Economy.
- Levitt, S. (2004). "Why are gambling markets organised so differently from financial markets?" The Economic Journal.
- Shin, H.S. (1993). "Measuring the incidence of insider trading in a market for state-contingent claims." The Economic Journal.
