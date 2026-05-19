from __future__ import annotations

import math

import pytest

from thandv import finance_tools as ft


# --- returns_from_prices --------------------------------------------------

def test_returns_from_prices_basic():
    out = ft.returns_from_prices([100.0, 110.0, 99.0])
    assert out == pytest.approx([0.10, -0.10])


def test_returns_from_prices_short():
    assert ft.returns_from_prices([]) == []
    assert ft.returns_from_prices([100.0]) == []


def test_returns_from_prices_rejects_non_positive():
    with pytest.raises(ValueError, match="non-positive price"):
        ft.returns_from_prices([100.0, 0.0, 50.0])


# --- annualised_volatility ------------------------------------------------

def test_annualised_volatility_known_value():
    # stdev of [0.01, -0.01, 0.01, -0.01] = 0.01154...; annualised x sqrt(252)
    rets = [0.01, -0.01, 0.01, -0.01]
    expected = ft.statistics.stdev(rets) * math.sqrt(252)
    assert ft.annualised_volatility(rets) == pytest.approx(expected)


def test_annualised_volatility_too_short_returns_zero():
    assert ft.annualised_volatility([]) == 0.0
    assert ft.annualised_volatility([0.01]) == 0.0


# --- sharpe_ratio ---------------------------------------------------------

def test_sharpe_ratio_positive_returns():
    rets = [0.01, 0.02, 0.015, 0.005, 0.01]
    sr = ft.sharpe_ratio(rets, rf=0.0)
    assert sr > 0  # all positive returns → positive Sharpe


def test_sharpe_ratio_constant_returns_zero():
    """A constant non-zero return has zero stdev → Sharpe collapses to 0
    rather than +inf, matching the documented behaviour."""
    rets = [0.01] * 10
    assert ft.sharpe_ratio(rets) == 0.0


def test_sharpe_ratio_short_input_raises():
    with pytest.raises(ValueError, match="at least two returns"):
        ft.sharpe_ratio([0.01])


def test_sharpe_ratio_rf_lowers_score():
    rets = [0.01, 0.02, 0.015, 0.005, 0.01]
    high = ft.sharpe_ratio(rets, rf=0.0)
    low = ft.sharpe_ratio(rets, rf=1.0)  # impossibly high rf
    assert low < high


# --- sortino_ratio --------------------------------------------------------

def test_sortino_ratio_no_downside_returns_inf():
    rets = [0.01, 0.02, 0.03]  # all positive vs target=0
    assert ft.sortino_ratio(rets) == math.inf


def test_sortino_ratio_mixed_returns_finite():
    rets = [0.01, -0.02, 0.03, -0.01, 0.02]
    s = ft.sortino_ratio(rets)
    assert math.isfinite(s)


def test_sortino_ratio_only_downside_negative():
    rets = [-0.01, -0.02, -0.03]
    s = ft.sortino_ratio(rets)
    assert s < 0


# --- max_drawdown --------------------------------------------------------

def test_max_drawdown_simple():
    # 100 -> 200 -> 50: dd = (50 - 200)/200 = -0.75
    assert ft.max_drawdown([100, 200, 50]) == pytest.approx(-0.75)


def test_max_drawdown_monotone_up():
    assert ft.max_drawdown([100, 101, 102, 103]) == 0.0


def test_max_drawdown_empty():
    assert ft.max_drawdown([]) == 0.0


# --- portfolio_metrics ---------------------------------------------------

def test_portfolio_metrics_bundle_shape():
    rets = [0.01, -0.005, 0.02, -0.01, 0.015]
    m = ft.portfolio_metrics(rets)
    assert set(m.keys()) == {
        "n", "total_return", "sharpe", "sortino", "max_drawdown", "volatility"
    }
    assert m["n"] == 5
    # Geometric total return matches manual computation
    cum = 1.0
    for r in rets:
        cum *= 1 + r
    assert m["total_return"] == pytest.approx(cum - 1)


def test_portfolio_metrics_short_input_raises():
    with pytest.raises(ValueError):
        ft.portfolio_metrics([0.01])


# --- correlation_matrix --------------------------------------------------

def test_correlation_matrix_perfect_positive():
    """Identical series → perfect positive correlation."""
    a = [0.01, 0.02, -0.01, 0.03]
    cm = ft.correlation_matrix({"x": a, "y": a})
    assert cm["x"]["y"] == pytest.approx(1.0)
    assert cm["x"]["x"] == 1.0


def test_correlation_matrix_perfect_negative():
    a = [0.01, 0.02, -0.01, 0.03]
    b = [-v for v in a]
    cm = ft.correlation_matrix({"x": a, "y": b})
    assert cm["x"]["y"] == pytest.approx(-1.0)


def test_correlation_matrix_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="align inputs"):
        ft.correlation_matrix({"x": [1.0, 2.0], "y": [1.0]})


def test_correlation_matrix_constant_series_yields_zero():
    """statistics.correlation raises on constant input; we swallow it as
    0.0 so the matrix stays usable."""
    cm = ft.correlation_matrix({"x": [0.01, -0.01, 0.02], "y": [1.0, 1.0, 1.0]})
    assert cm["x"]["y"] == 0.0


def test_correlation_matrix_empty():
    assert ft.correlation_matrix({}) == {}


# --- parse_positions_csv -------------------------------------------------

def test_parse_positions_csv_basic(tmp_path):
    csv_path = tmp_path / "pos.csv"
    csv_path.write_text(
        "symbol,qty,avg_price,asset_class\n"
        "AAPL,10,150.00,equity\n"
        "TLT,5,90.00,bond\n"
    )
    out = ft.parse_positions_csv(csv_path)
    assert len(out) == 2
    assert out[0].symbol == "AAPL"
    assert out[0].market_value == pytest.approx(1500.0)
    assert out[1].asset_class == "bond"


def test_parse_positions_csv_default_asset_class(tmp_path):
    csv_path = tmp_path / "pos.csv"
    csv_path.write_text("symbol,qty,avg_price\nAAPL,10,150.00\n")
    out = ft.parse_positions_csv(csv_path)
    assert out[0].asset_class == "equity"


def test_parse_positions_csv_missing_column_raises(tmp_path):
    csv_path = tmp_path / "pos.csv"
    csv_path.write_text("symbol,qty\nAAPL,10\n")
    with pytest.raises(ValueError, match="missing required column"):
        ft.parse_positions_csv(csv_path)


def test_parse_positions_csv_bad_row_points_to_line(tmp_path):
    csv_path = tmp_path / "pos.csv"
    csv_path.write_text(
        "symbol,qty,avg_price\n"
        "AAPL,not-a-number,150.00\n"
    )
    with pytest.raises(ValueError, match="row 2"):
        ft.parse_positions_csv(csv_path)


def test_parse_positions_csv_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ft.parse_positions_csv(tmp_path / "missing.csv")


# --- exposure_summary ----------------------------------------------------

def test_exposure_summary_empty():
    s = ft.exposure_summary([])
    assert s["gross"] == 0.0 and s["net"] == 0.0
    assert s["by_asset_class"] == {} and s["by_symbol"] == {}


def test_exposure_summary_long_and_short():
    positions = [
        ft.Position("AAPL", 10, 150.0, "equity"),    # +1500
        ft.Position("TLT", -5, 90.0, "bond"),        # -450
        ft.Position("AAPL", 5, 160.0, "equity"),     # +800 -- same symbol
    ]
    s = ft.exposure_summary(positions)
    assert s["gross"] == pytest.approx(2750.0)
    assert s["net"] == pytest.approx(1850.0)
    assert s["by_asset_class"]["equity"] == pytest.approx(2300.0)
    assert s["by_asset_class"]["bond"] == pytest.approx(-450.0)
    assert s["by_symbol"]["AAPL"] == pytest.approx(2300.0)
    assert s["by_symbol"]["TLT"] == pytest.approx(-450.0)


# --- backtest ------------------------------------------------------------

def test_backtest_long_only_uptrend_matches_buy_and_hold():
    """If we're long every day with no commissions, the backtest's total
    return must equal price[-1]/price[0] - 1."""
    prices = [100.0, 110.0, 121.0, 133.1]
    signals = [1, 1, 1, 1]  # last one ignored
    r = ft.backtest(prices, signals)
    expected = prices[-1] / prices[0] - 1
    assert r["total_return"] == pytest.approx(expected)
    assert r["n_trades"] == 1  # only the initial flat->long transition


def test_backtest_flat_zero_return():
    prices = [100.0, 110.0, 90.0, 100.0]
    signals = [0, 0, 0, 0]
    r = ft.backtest(prices, signals)
    assert r["total_return"] == 0.0
    assert r["n_trades"] == 0


def test_backtest_short_during_downtrend_is_profitable():
    prices = [100.0, 90.0, 80.0]
    signals = [-1, -1, -1]
    r = ft.backtest(prices, signals)
    assert r["total_return"] > 0


def test_backtest_lookahead_protection():
    """A signal on the last day must be ignored: there is no next-day
    return to apply it to. If we DID apply it, a perfect-foresight 'short
    just before a crash' would show non-zero return even with a 2-element
    series + a -1 on the second element."""
    prices = [100.0, 50.0]
    signals = [0, -1]  # short on the last day
    r = ft.backtest(prices, signals)
    # day 0 signal=0 → position=0 → no return.
    # day 1 signal=-1 is ignored (no day 2 to realise on).
    assert r["total_return"] == 0.0


def test_backtest_signal_validation():
    with pytest.raises(ValueError, match="signal must be"):
        ft.backtest([100.0, 110.0], [2, 0])


def test_backtest_length_mismatch():
    with pytest.raises(ValueError, match="signals length"):
        ft.backtest([100.0, 110.0, 120.0], [1, 1])


def test_backtest_commission_reduces_return():
    prices = [100.0, 110.0, 100.0, 110.0, 100.0]
    signals = [1, -1, 1, -1, 0]  # high turnover
    no_cost = ft.backtest(prices, signals, commission_bps=0.0)
    with_cost = ft.backtest(prices, signals, commission_bps=50.0)
    assert with_cost["total_return"] < no_cost["total_return"]
    assert with_cost["n_trades"] == no_cost["n_trades"]


# --- read_prices_csv / read_signals_csv ----------------------------------

def test_read_prices_csv_basic(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("date,price\n2024-01-01,100.0\n2024-01-02,101.0\n")
    out = ft.read_prices_csv(p)
    assert out == [100.0, 101.0]


def test_read_prices_csv_missing_column(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("date,close\n2024-01-01,100.0\n")
    with pytest.raises(ValueError, match="missing column"):
        ft.read_prices_csv(p)


def test_read_prices_csv_skips_empty(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("price\n100.0\n\n101.0\n")
    out = ft.read_prices_csv(p)
    assert out == [100.0, 101.0]


def test_read_signals_csv_basic(tmp_path):
    p = tmp_path / "sig.csv"
    p.write_text("date,signal\n2024-01-01,1\n2024-01-02,-1\n2024-01-03,0\n")
    out = ft.read_signals_csv(p)
    assert out == [1, -1, 0]


def test_read_signals_csv_rejects_out_of_range(tmp_path):
    p = tmp_path / "sig.csv"
    p.write_text("signal\n1\n2\n")
    with pytest.raises(ValueError, match="row 3"):
        ft.read_signals_csv(p)
