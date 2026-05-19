"""Pure-Python finance toolkit: returns, risk metrics, correlations,
positions parsing, and a minimal backtest harness.

Stdlib only, on purpose. vectorbt / backtrader are powerful but heavy --
this module is for quick sanity-check work the finance persona can drive
through the `thandv finance` CLI or via `run_bash`. For production
strategy research, users should run vectorbt or backtrader directly.

Honesty: nothing in this module looks at live market data, predicts
prices, or recommends positions. It computes descriptive numbers from
data the user provides. Per NOT_FINANCIAL_ADVICE.md.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PERIODS_PER_YEAR = 252  # US equity trading days


# --- Returns ------------------------------------------------------------

def returns_from_prices(prices: list[float]) -> list[float]:
    """Simple period-over-period returns. `returns[i] = prices[i+1]/prices[i] - 1`.

    Length = len(prices) - 1. Raises ValueError on prices <= 0 (would imply
    a delisting or data error, not a return we can compute).
    """
    if len(prices) < 2:
        return []
    out: list[float] = []
    for prev, curr in zip(prices, prices[1:]):
        if prev <= 0:
            raise ValueError(f"non-positive price in series: {prev}")
        out.append(curr / prev - 1.0)
    return out


# --- Risk metrics -------------------------------------------------------

def annualised_volatility(
    returns: list[float], periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    """Sample standard deviation of returns, scaled by sqrt(periods/year).

    Returns 0.0 for series with < 2 observations -- we can't compute a
    sample stdev from a single point and the alternative (raise) is
    annoying for callers that just want a "no signal" sentinel.
    """
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: list[float],
    rf: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualised Sharpe = (mean(returns) - rf_per_period) / stdev(returns)
    * sqrt(periods/year). `rf` is the annual risk-free rate.

    Returns 0.0 if stdev is 0 (constant series); raises ValueError on
    fewer than two observations -- a Sharpe from one return is nonsense
    and silent zeros there would mislead.
    """
    if len(returns) < 2:
        raise ValueError("need at least two returns for a Sharpe ratio")
    rf_per_period = rf / periods_per_year
    sd = statistics.stdev(returns)
    if sd == 0:
        return 0.0
    excess = statistics.fmean(returns) - rf_per_period
    return excess / sd * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: list[float],
    rf: float = 0.0,
    target: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualised Sortino = (mean(returns) - rf_per_period) / downside_dev
    * sqrt(periods/year).

    Downside deviation = sqrt(mean(min(r - target, 0)^2)) -- only
    sub-target periods contribute, but the *mean* is over the full
    series (a standard convention; some libs use only downside count).

    Returns +inf when there is excess return but no downside (a useful
    signal, not an error) and 0.0 when both excess and downside are
    zero. Raises ValueError on fewer than two observations.
    """
    if len(returns) < 2:
        raise ValueError("need at least two returns for a Sortino ratio")
    rf_per_period = rf / periods_per_year
    excess = statistics.fmean(returns) - rf_per_period
    downside_squared = [min(r - target, 0.0) ** 2 for r in returns]
    dd = math.sqrt(sum(downside_squared) / len(returns))
    if dd == 0:
        return 0.0 if excess == 0 else math.inf if excess > 0 else -math.inf
    return excess / dd * math.sqrt(periods_per_year)


def max_drawdown(prices_or_equity: list[float]) -> float:
    """Largest peak-to-trough decline on the input series.

    Returned as a non-positive float: 0.0 for a monotone-up series,
    -0.5 for a 50% drawdown. Accepts either prices or an equity curve;
    the math is the same.
    """
    if not prices_or_equity:
        return 0.0
    peak = prices_or_equity[0]
    worst = 0.0
    for v in prices_or_equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            if dd < worst:
                worst = dd
    return worst


def portfolio_metrics(
    returns: list[float],
    rf: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> dict:
    """Bundle of {sharpe, sortino, max_drawdown, volatility, total_return}.

    Caller passes per-period returns. `total_return` is the cumulative
    product minus one (geometric). `max_drawdown` is computed on the
    equity curve derived from these returns.
    """
    if len(returns) < 2:
        raise ValueError("need at least two returns for portfolio metrics")
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    return {
        "n": len(returns),
        "total_return": equity[-1] - 1.0,
        "sharpe": sharpe_ratio(returns, rf=rf, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, rf=rf, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "volatility": annualised_volatility(returns, periods_per_year=periods_per_year),
    }


# --- Correlation --------------------------------------------------------

def correlation_matrix(returns_by_asset: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Pearson correlation matrix between every pair of return series.

    All series must have the same length (we don't align by date here --
    that's the caller's job). Constant series produce NaN-equivalent 0.0
    rather than raising, so the matrix stays usable.
    """
    names = list(returns_by_asset)
    if not names:
        return {}
    n = len(next(iter(returns_by_asset.values())))
    for k, v in returns_by_asset.items():
        if len(v) != n:
            raise ValueError(
                f"series {k!r} has length {len(v)}, expected {n}; "
                "align inputs to common dates before calling"
            )
    out: dict[str, dict[str, float]] = {}
    for a in names:
        out[a] = {}
        for b in names:
            if a == b:
                out[a][b] = 1.0
                continue
            try:
                out[a][b] = statistics.correlation(returns_by_asset[a], returns_by_asset[b])
            except statistics.StatisticsError:
                out[a][b] = 0.0
    return out


# --- Positions / exposure -----------------------------------------------

@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: float
    asset_class: str = "equity"

    @property
    def market_value(self) -> float:
        """Notional at avg_price -- not mark-to-market; the user gives us
        cost basis, not live quotes."""
        return self.qty * self.avg_price


def parse_positions_csv(path: str | Path) -> list[Position]:
    """Read a positions CSV with columns: symbol, qty, avg_price[, asset_class].

    `asset_class` defaults to "equity". Unknown columns are ignored.
    Raises ValueError with a clear message on missing required columns
    so the user knows what to fix.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{p}: empty CSV (no header row)")
        missing = {"symbol", "qty", "avg_price"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{p}: missing required column(s): {sorted(missing)}"
            )
        out: list[Position] = []
        for i, row in enumerate(reader, 2):  # row 2 = first data row
            try:
                out.append(
                    Position(
                        symbol=row["symbol"].strip(),
                        qty=float(row["qty"]),
                        avg_price=float(row["avg_price"]),
                        asset_class=(row.get("asset_class") or "equity").strip(),
                    )
                )
            except (ValueError, KeyError) as e:
                raise ValueError(f"{p}: row {i}: {e}") from e
    return out


def exposure_summary(positions: list[Position]) -> dict:
    """Notional exposure grouped by asset_class plus a gross/net split.

    Gross = sum of |market_value|; net = sum (signed). Useful sanity
    check before reporting "I'm 60% in tech" -- ensures the user sees
    the actual breakdown rather than a model's narrative of it.
    """
    if not positions:
        return {
            "gross": 0.0,
            "net": 0.0,
            "by_asset_class": {},
            "by_symbol": {},
        }
    gross = sum(abs(p.market_value) for p in positions)
    net = sum(p.market_value for p in positions)
    by_class: dict[str, float] = {}
    by_sym: dict[str, float] = {}
    for p in positions:
        by_class[p.asset_class] = by_class.get(p.asset_class, 0.0) + p.market_value
        by_sym[p.symbol] = by_sym.get(p.symbol, 0.0) + p.market_value
    return {
        "gross": gross,
        "net": net,
        "by_asset_class": by_class,
        "by_symbol": by_sym,
    }


# --- Backtest harness ---------------------------------------------------

def _validate_signal(s: float) -> int:
    si = int(s)
    if si not in (-1, 0, 1):
        raise ValueError(f"signal must be -1, 0, or 1; got {s}")
    return si


def backtest(
    prices: list[float],
    signals: list[int],
    *,
    commission_bps: float = 0.0,
) -> dict:
    """Tiny backtest harness: signals[i] sets the position at the close of
    day i and is realised on the day i+1 close-to-close return.

    This 'signal-at-close, position-effective-next-day' convention is the
    standard guard against lookahead bias. If you want intraday execution
    realism, use vectorbt or backtrader directly.

    Inputs:
      - `prices`: list of close prices, length N (>= 2).
      - `signals`: list of -1 / 0 / +1, length N. Last signal is ignored
        (there's no "next day" to realise it on).
      - `commission_bps`: round-trip commission per turnover unit, in
        basis points. e.g. 5 = 0.05% per unit of position change.

    Returns: dict with equity_curve (list, length N), total_return,
    sharpe, max_drawdown, n_trades, turnover.
    """
    if len(prices) < 2:
        raise ValueError("backtest needs at least two prices")
    if len(signals) != len(prices):
        raise ValueError(
            f"signals length {len(signals)} must equal prices length {len(prices)}"
        )
    valid_signals = [_validate_signal(s) for s in signals]

    rets = returns_from_prices(prices)  # length N-1, rets[i] is day i->i+1
    # Position effective on day i+1's return is signal[i].
    positions = valid_signals[:-1]
    commission = commission_bps / 10_000.0

    # Turnover at day i+1's open: |positions[i] - positions[i-1]|; on day 0
    # we assume we started flat, so first turnover = |positions[0] - 0|.
    daily_pnl: list[float] = []
    n_trades = 0
    total_turnover = 0.0
    prev_pos = 0
    for i, (pos, r) in enumerate(zip(positions, rets)):
        turn = abs(pos - prev_pos)
        if turn != 0:
            n_trades += 1
            total_turnover += turn
        daily_pnl.append(pos * r - turn * commission)
        prev_pos = pos

    equity = [1.0]
    for pnl in daily_pnl:
        equity.append(equity[-1] * (1 + pnl))

    return {
        "equity_curve": equity,
        "total_return": equity[-1] - 1.0,
        "sharpe": sharpe_ratio(daily_pnl) if len(daily_pnl) >= 2 else 0.0,
        "max_drawdown": max_drawdown(equity),
        "n_trades": n_trades,
        "turnover": total_turnover,
    }


# --- Lightweight CSV readers used by the CLI ----------------------------

def read_prices_csv(path: str | Path, column: str = "price") -> list[float]:
    """Read a single price column from a CSV. Header is required.

    `column` defaults to 'price'. Empty lines are skipped. Non-numeric
    rows raise ValueError with the row number for easy debugging.
    """
    p = Path(path)
    with p.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{p}: missing column {column!r}")
        out: list[float] = []
        for i, row in enumerate(reader, 2):
            cell = (row.get(column) or "").strip()
            if not cell:
                continue
            try:
                out.append(float(cell))
            except ValueError as e:
                raise ValueError(f"{p}: row {i}: {e}") from e
    return out


def read_signals_csv(path: str | Path, column: str = "signal") -> list[int]:
    """Read a single integer-signal column from a CSV. Same shape as
    `read_prices_csv` -- header required, empty rows skipped, bad values
    raise with the row number."""
    p = Path(path)
    with p.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{p}: missing column {column!r}")
        out: list[int] = []
        for i, row in enumerate(reader, 2):
            cell = (row.get(column) or "").strip()
            if not cell:
                continue
            try:
                out.append(_validate_signal(float(cell)))
            except ValueError as e:
                raise ValueError(f"{p}: row {i}: {e}") from e
    return out
