# Not Financial Advice

Thandv includes a `finance` persona. This document is the canonical, plain-
language statement of what that persona is and isn't.

## What the finance persona can do

- Career: resumes, cover letters, freelance proposals, side-hustle research,
  contract review.
- Personal finance: budgeting, tax-document organisation and parsing
  (research, not advice), expense analysis from CSVs.
- Investment research: summarise 10-Ks, earnings calls, news articles;
  explain financial concepts (options, duration, beta, free cash flow, etc.);
  parse portfolio statements; compute portfolio metrics (Sharpe, Sortino,
  drawdown, exposure, correlation).
- Strategy work: write trading strategies in Python (vectorbt / backtrader);
  run backtests on historical data; critique strategies for assumptions,
  lookahead bias, survivorship bias, data snooping, and unrealistic
  transaction costs; scaffold paper-trading harnesses.

## What it does not do

- Recommend specific securities to buy or sell.
- Predict market direction.
- Provide investment advice as that term is meant by securities regulators.
- Claim or generate "alpha", "edge", or any forward-looking return forecast.

## Capability honesty

Thandv runs a local 7–30B parameter open-weights model. It has:

- No real-time market data.
- No quantitative or markets-specific training corpus.
- No statistical edge over markets, by construction.

If a strategy it writes happens to backtest well on historical data, that
is not evidence of forward returns; that is in-sample fitting plus
survivorship bias. The model knows enough to **say so** when asked, and
the persona prompt instructs it to.

## Legal frame

Output from the finance persona is **educational and analytical content**.
Nothing it produces is investment advice in your jurisdiction or any
other.

Investment advice is regulated in most countries (US: Investment Advisers
Act of 1940 + state rules; UK: FCA; EU: MiFID II; India: SEBI; etc.).
The line between "research and education" and "advice" varies by
jurisdiction. If you use Thandv to make your own financial decisions,
that is your decision and your responsibility. If you distribute or
modify Thandv such that it gives investment advice *to others* — to
clients, subscribers, or the public — you may be inside the perimeter of
your local securities regulator. Read your rules first.

This project is research scaffolding. The model is bad at markets. We say
so plainly, in the system prompt, here, and on every relevant reply.
