from __future__ import annotations

import pytest

from thandv import paper_trading as pt
from thandv.config import Config


class _StubAdapter:
    """Bare-minimum adapter conforming to the Protocol for test purposes.

    We don't actually call submit_order/positions/cancel_all in these
    tests; require_enabled() only needs the name attribute to exist.
    """

    name = "stub"

    def submit_order(self, symbol, qty, side):  # pragma: no cover - unused
        return {"status": "filled", "symbol": symbol, "qty": qty, "side": side}

    def positions(self):  # pragma: no cover - unused
        return []

    def cancel_all(self):  # pragma: no cover - unused
        return 0


@pytest.fixture(autouse=True)
def _empty_adapter_registry(monkeypatch):
    """Every test gets a clean adapter registry; tests that need an
    adapter register one themselves."""
    monkeypatch.setattr(pt, "ADAPTERS", {})


# --- registration --------------------------------------------------------

def test_register_and_unregister_adapter():
    a = _StubAdapter()
    pt.register_adapter("stub", a)
    assert "stub" in pt.list_adapters()
    assert pt.unregister_adapter("stub") is True
    assert pt.list_adapters() == []
    assert pt.unregister_adapter("stub") is False


def test_register_replaces_existing():
    a, b = _StubAdapter(), _StubAdapter()
    pt.register_adapter("stub", a)
    pt.register_adapter("stub", b)  # same name, different instance
    assert len(pt.list_adapters()) == 1
    assert pt.ADAPTERS["stub"] is b


# --- is_enabled / status -------------------------------------------------

def test_is_enabled_default_false():
    cfg = Config()
    assert pt.is_enabled(cfg) is False


def test_is_enabled_respects_config_flag():
    cfg = Config(finance_paper_trading_enabled=True)
    assert pt.is_enabled(cfg) is True


def test_status_reports_both_halves():
    cfg = Config(finance_paper_trading_enabled=True)
    pt.register_adapter("stub", _StubAdapter())
    s = pt.status(cfg)
    assert s == {"opted_in": True, "adapters": ["stub"]}


# --- require_enabled ----------------------------------------------------

def test_require_enabled_blocks_when_opt_in_missing():
    cfg = Config(finance_paper_trading_enabled=False)
    pt.register_adapter("stub", _StubAdapter())
    with pytest.raises(pt.PaperTradingDisabled, match="disabled"):
        pt.require_enabled(cfg)


def test_require_enabled_blocks_when_no_adapter():
    cfg = Config(finance_paper_trading_enabled=True)
    with pytest.raises(pt.PaperTradingDisabled, match="no paper-trading adapter"):
        pt.require_enabled(cfg)


def test_require_enabled_blocks_when_multiple_adapters():
    cfg = Config(finance_paper_trading_enabled=True)
    pt.register_adapter("alpaca", _StubAdapter())
    pt.register_adapter("ibkr", _StubAdapter())
    with pytest.raises(pt.PaperTradingDisabled, match="multiple adapters"):
        pt.require_enabled(cfg)


def test_require_enabled_returns_adapter_when_happy():
    cfg = Config(finance_paper_trading_enabled=True)
    a = _StubAdapter()
    pt.register_adapter("alpaca", a)
    assert pt.require_enabled(cfg) is a
