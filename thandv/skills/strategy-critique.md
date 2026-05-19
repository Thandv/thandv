A trading strategy is suspect until proven otherwise. Before reporting
returns, work through this checklist. If you cannot answer one, say so
plainly -- skipping a check is dishonest.

Lookahead bias
- Does the signal at time t use any data only known at t+1 or later?
- Resampled bars: does the resample window end after the bar you signal
  on? (it shouldn't.)
- Fundamental data: is the **filing date** used, or the **reporting
  period date**? Use filing date; the rest is lookahead.

Survivorship bias
- Does the price universe include delisted tickers? If not, expect
  3-5% annual upward bias in equities backtests.
- Index histories: point-in-time membership, not current-membership.
  S&P 500 today is not S&P 500 in 2005.

Overfitting
- How many tunable parameters? Each one needs many independent
  observations to justify.
- Was the parameter search run on the same window used to report
  results? If yes, the headline number is meaningless.
- "Out-of-sample" doesn't count if you peeked during the search.

Transaction costs
- Slippage, half-spread, commission -- pick numbers and state them.
- For mid-frequency strategies, realistic costs often eat the entire
  backtested edge.

Capacity
- Does the strategy assume infinite liquidity? Stress-test the size at
  which average daily volume of the traded names becomes a constraint.

Regime
- What window does the backtest cover? Is it dominated by one regime
  (e.g. 2010s low-vol bull, 2020 vol spike)?
- Report worst-decile-month return and longest drawdown duration, not
  just average Sharpe.

Statistical significance
- Number of independent trades, not number of days.
- Bootstrap or randomisation: what's the probability this Sharpe came
  from luck given the trade count?

Be specific. "The backtest assumes zero costs" is honest; "I optimised
this carefully" is not. When in doubt, default to the more conservative
number.
