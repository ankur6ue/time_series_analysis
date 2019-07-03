# code taken from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from pandas import read_csv
import pandas
from datetime import datetime
import os
from matplotlib import pyplot
# dataset source: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
path = os.path.dirname(os.path.realpath(__file__))
# Running the example prints the first 5 rows of the transformed dataset and saves the dataset to “pollution.csv“
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('pollution_raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
weekday = pandas.Series(dataset.index[:].weekday, dtype='int32')
hour = pandas.Series(dataset.index[:].hour, dtype='int32')
# manually specify column names. Note we must insert values, not the series, otherwise we'll get index matching issues.
# see: https://stackoverflow.com/questions/42041092/python-adding-column-to-dataframe-causes-nan
dataset.insert(1, 'weekday', weekday.values)
dataset.insert(2, 'hour', hour.values)
dataset.columns = ['pollution', 'weekday', 'hour', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
#dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
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
	pyplot.plot(values[20000:-1, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
# for adjusting vertical spacing between subplots (hspace is for vertical spacing, go figure)
fig.subplots_adjust(hspace=0.5)
pyplot.show()
# save to file
dataset.to_csv('pollution.csv')