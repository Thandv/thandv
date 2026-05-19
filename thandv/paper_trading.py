"""Paper-trading harness contract and opt-in gate.

This module ships ZERO broker integrations on purpose. Paper trading
that connects to a real broker (even in sandbox mode) needs the user's
credentials, has rate limits, and -- critically -- gets confused with
live trading by people who didn't read the docs. So:

1. The harness is OFF by default. `Config.finance_paper_trading_enabled`
   must be set to True before any adapter call goes through.
2. No PaperTradingAdapter implementations are bundled. Users register
   their own (e.g. Alpaca, IBKR, Kraken) at import time. We document
   the contract; the trading code stays in user-land.
3. Every gated call returns a structured dict so the CLI / agent can
   surface a clear reason instead of a stack trace.

This keeps the project honest: we don't pretend to ship trading
infrastructure we haven't built.
"""

from __future__ import annotations

from typing import Protocol

from thandv.config import Config


class PaperTradingAdapter(Protocol):
    """Contract every user-supplied paper-trading adapter must satisfy.

    Adapters live entirely in user code. Register at runtime with
    `register_adapter("my-broker", MyAdapter())`. No bundled
    implementations -- shipping one would imply we've tested it against
    a live broker, which we haven't.
    """

    name: str  # short identifier shown to the user

    def submit_order(
        self, symbol: str, qty: float, side: str
    ) -> dict:
        """Submit a paper order. Returns broker-specific metadata as a
        dict; `side` is 'buy' or 'sell'."""
        ...

    def positions(self) -> list[dict]:
        """Return current paper positions as list of dicts with at least
        `symbol`, `qty`, `avg_price`."""
        ...

    def cancel_all(self) -> int:
        """Cancel every open paper order. Returns the count cancelled."""
        ...


ADAPTERS: dict[str, PaperTradingAdapter] = {}


def register_adapter(name: str, adapter: PaperTradingAdapter) -> None:
    """User-land hook: register a PaperTradingAdapter instance.

    Last-one-wins on duplicate names so user code can swap adapters
    during a single Python session (e.g. test harnesses).
    """
    ADAPTERS[name] = adapter


def unregister_adapter(name: str) -> bool:
    return ADAPTERS.pop(name, None) is not None


def list_adapters() -> list[str]:
    return sorted(ADAPTERS)


def is_enabled(cfg: Config | None = None) -> bool:
    """True iff the user has explicitly opted in via config.

    A `True` here does NOT mean an adapter is registered -- it only
    means the user has acknowledged the opt-in. `status()` reports both
    halves so callers can produce specific error messages.
    """
    cfg = cfg or Config.load()
    return bool(cfg.finance_paper_trading_enabled)


def status(cfg: Config | None = None) -> dict:
    """Snapshot of (opt_in flag, registered adapter count, names).

    Used by the CLI to print a single human-readable line. Doesn't
    raise, doesn't side-effect.
    """
    cfg = cfg or Config.load()
    return {
        "opted_in": bool(cfg.finance_paper_trading_enabled),
        "adapters": list_adapters(),
    }


# --- Gated call ---------------------------------------------------------

class PaperTradingDisabled(RuntimeError):
    """Raised when paper trading is invoked without opt-in or without
    a registered adapter."""


def require_enabled(cfg: Config | None = None) -> PaperTradingAdapter:
    """Return the (single) registered adapter, or raise with a clear
    reason. If the user has more than one adapter registered, this
    raises and asks them to pick -- we don't guess.

    Two failure modes:
      - opt-in not set: PaperTradingDisabled with config-set instructions
      - no adapters registered: PaperTradingDisabled pointing at the
        registration API + docs
    """
    cfg = cfg or Config.load()
    if not cfg.finance_paper_trading_enabled:
        raise PaperTradingDisabled(
            "paper trading is disabled. Set "
            "`finance_paper_trading_enabled=true` in your config "
            "(thandv config --set finance_paper_trading_enabled=true) "
            "after reading NOT_FINANCIAL_ADVICE.md."
        )
    if not ADAPTERS:
        raise PaperTradingDisabled(
            "no paper-trading adapter registered. Thandv does not bundle "
            "broker integrations. Register your own with "
            "paper_trading.register_adapter(name, adapter) -- see the "
            "PaperTradingAdapter Protocol for the contract."
        )
    if len(ADAPTERS) > 1:
        raise PaperTradingDisabled(
            f"multiple adapters registered ({list_adapters()}); pick one "
            "by unregistering the others. The harness does not guess."
        )
    return next(iter(ADAPTERS.values()))
