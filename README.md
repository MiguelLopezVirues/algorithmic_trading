
This Readme has to be the introduction to the whole project. Here I have to explain:

Premise:
- Being able to establish RR based on prediction intervals.

Approach:
- ML for predictions. Using intervals to establish RR ratios and using those as dynamic TP/SL.
- Analysis of the strategy.
- Backtesting (not the same as ML Forecast Backtesting) of strategy.
- Put into production.
- Trading API connection


For presentation:
- Distinction between more training data in terms of time (time window) and in terms of examples (stocks)



Things to highlight:
- Customization of Skforecast differentiator.

Lessons learned:
- More exhaustive control of dataset and model versions

Next steps:
- Work with all days of the week, as potentially better entries might be found there.
- Experiment further with different configuration of dependent time series and direct models. Wait for upcoming Skforecast version that will bring that feature or customize it.
- Integrate custom quantile prediction or residuals correction to mimic the approach followed in this study, with enhanced performance.
- Extend this study with novel data architectures like Graph Databases for GNNs.