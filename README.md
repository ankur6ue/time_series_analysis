# time series data analysis

example 1: Uses a 2 layer LSTM to perform forecasting for a simple function without covariates (see example1.png for sample result)

example 2: Adds the time index as a covariate (see example2.png for sample result)

example 2_b: Performs forecasting for the pollution dataset (https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data). Only the time_index 
is used as a covariate.

Subclasses the Pytorch dataset and sampler classes to load batches starting with random time indices. The time indices in a 
given batch are sequential 

example 2_c: Forecasting on the pollution dataset with other variables also used as covariates.

example 4: The model outputs the parameters of a distribution (modeled as a Gaussian) rather than the predicted value. 

TBD: Implement adding hour/time of day as a covariate, add data loaders for other time series - parts, trips etc, add performance 
metrics..
