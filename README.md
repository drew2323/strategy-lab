# Notes for The Team

Entry points are in `strat_STRATEGYNAME` directories in `research`.

there are `SINGLE` and `MULTI` version
- `SINGLE` - just straight one pass meaning fetching data (local/remote), the data cleaning(only main session, no extended market) indicators and entry-points (in given time window) and backtest with SINGLE parameter values and strategy result. Used for basic strategy research.
- `MULTI` - basic strategy pass taken from step above but with hyperparameters testing, result for each strategy for each parameter combination in vbt portfolio dataset. To evaluate performance of each(best) parameter combination, maybe display as parallel coordinates plot.

I can imagine it can be used to create MVP (not sure about notebook format as it becomes clumsy with lots of data), hyperparameter testing should be refactored, walk forward optimization and cross validation should be added.

Maybe start exploration with `/research/strat_ORDER_IMBALANCE/v2_single.ipynb`

strategies I have worked on so far
- CANDLE_GAPS - candle gaps of various size within high resolution ohlcv (1s) as source for entry signals
- LINREG_MULTI - combination of linear regressions on different length and time resolutions
- SUPERTREND - supetrend on different resolutions
- ORDER_IMBALANCE - order imbalanced columns calculated from trade data as new columns on OHLCV bars
- TIME_ENTRIES - just idea for time based entries to leverage random movement of price

## Data preparation

Usually for strategies above,  1second OHLCV data are prepared by vectorized aggregator based on trade data for given period. Trade data are either loaded from local file or fetched from Alpaca (simple not funded registration required allowing to fetch data both crypto/stocks). Aggregator is in `prepare_aggregated_data.ipynb`. Alpaca secret and key is loaded from .env it contains just

```
ACCOUNT1_LIVE_API_KEY=_YOUR_ALPACA_API_KEY
ACCOUNT1_LIVE_SECRET_KEY=YOUR_ALPACA_SECRET_KEY
```

It tries to load trade data from local files, if they are not present they are fetched from Alpaca by `fetch_trades_parallel` by reusing components from my `v2realbot` platform which is imported as requirement. Then the trades are aggregated by `aggregate_trades` - it supports time based OHLCV or Volume Bars, Dollar Bars or Renko Bars.

I usually prepare few months of 1s data (as highest resolution) and store it as `.parquet` file for each symbol. Then during the research I just load these parquets and then resample it to lower frequencies for signals, but usually backtest on 1s frequency in order to get highest precision of entry/exit.

Hope it helps, feel free to contact me if you get stuck with something.

# Research for v2realbot

## Overview
Strategy research and development tracker. Serves as a central hub for strategizing, idea generation, and performance tracking of the stragies

## Purpose
This repository is established as an issue tracker to:
- Facilitate the proposal and discussion of new trading strategies.
- Track the progress and refinement of existing strategies.
- Document and share insights, research findings, and performance analyses.

## Getting Started
1. **Proposal of New Strategies**: To propose a new strategy, create a new issue with a clear and descriptive title. Outline your strategy, including its rationale, intended market conditions, and expected outcomes.

2. **Discussion and Feedback**: Use the issues section for ongoing discussions, feedback, and collaborative refinement of strategies.

3. **Strategy Documentation**: Each strategy should be documented in detail on the Wiki pages, including its parameters, implementation guidelines, and any relevant backtesting results. (Note: documentation either here or on [trading.mujdenik.eu](trading.mujdenik.eu) - we will see what's better in practice)

4. **Problem Reporting and Optimization**: Report any issues or suggest optimizations for existing strategies through the Issues section, providing relevant data and analysis to support your observations.

5. **Contribution Guidelines**: Please refer to `CONTRIBUTING.md` for detailed guidelines on how to contribute effectively to this repository.

## Collaboration Tools
- **Issues**: For proposing, discussing, and tracking strategies.
- **Wiki**: For detailed documentation and resource sharing, please se [trading.mujdenik.eu](trading.mujdenik.eu)
- **Research folder**: Notebook dedicated do each strategy.

## Integration with v2realbot
This repository operates in tandem with the [v2realbot](https://github.com/drew2323/v2trading) repository. Ensure all strategies are compatible and follow the guidelines for integration into the v2realbot Trading Platform.
