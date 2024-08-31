# Note for The Team

Entry points are in `strat_STRATEGYNAME` directories.

there are `SINGLE` and `MULTI` version
- `SINGLE` - just straight one pass meaning fetching data (local/remote), indicators and entry-points and backtest with SINGLE parameter values and strategy result. Used for basic strategy research.
- `MULTI` - basic strategy pass taken from step above but with hyperparameters testing, result for each strategy for each parameter combination in vbt portfolio dataset. To evaluate performance of each(best) parameter combination, maybe display as parallel coordinates plot.

I can imagine it can be used to create MVP (not use with notebooks as they become clumsy with lot of data), hyperparameter testing should be refactored, walk forward optimization and cross validation should be added.

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
This repository operates in tandem with the `v2realbot` repository. Ensure all strategies are compatible and follow the guidelines for integration into the v2realbot Trading Platform.
