# code taken from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from pandas import read_csv
import pandas
from datetime import datetime
import os
from matplotlib import pyplot
# dataset source: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
path = os.path.dirname(os.path.realpath(__file__))
def dateparse(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
dataset = read_csv('LD2011_2014.txt', sep=";", decimal=",", header=None, skiprows=[0], parse_dates = [[0]], date_parser=dateparse)
dataset.set_index("0", inplace=True)
weekday = pandas.Series(dataset.index[:].weekday, dtype='int32')
hour = pandas.Series(dataset.index[:].hour, dtype='int32')
minute = pandas.Series(dataset.index[:].minute, dtype='int32')
# manually specify column names. Note we must insert values, not the series, otherwise we'll get index matching issues.
# see: https://stackoverflow.com/questions/42041092/python-adding-column-to-dataframe-causes-nan
# insert after all the time series columns. wday, hour and minute are the only covariates in this case
dataset.insert(len(dataset.columns), 'weekday', weekday.values)
dataset.insert(len(dataset.columns), 'hour', hour.values)
dataset.insert(len(dataset.columns), 'min', minute.values)
dataset.index.name = 'date'

# drop the first 35042 values because they are all zeros
dataset = dataset[35042:]
# summarize first 5 rows
print(dataset.head(5))

# plot some columns
values = dataset.values
groups = [0, 1, 2, 3, 4, 5, 6, 7]
i = 1
# plot each column
fig = pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[20000:24000, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
# for adjusting vertical spacing between subplots (hspace is for vertical spacing, go figure)
fig.subplots_adjust(hspace=0.5)
pyplot.show()
# save to file
dataset.to_csv('electricity.csv')