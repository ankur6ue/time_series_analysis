# code taken from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from pandas import read_csv
import pandas
from datetime import datetime
import os
from matplotlib import pyplot

path = os.path.dirname(os.path.realpath(__file__))
# Running the example prints the first 5 rows of the transformed dataset and saves the dataset to “pollution.csv“
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv(path + '\\..\\data\\pollution_raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
weekday = pandas.Series(dataset.index[:].weekday, dtype='int32')
hour = pandas.Series(dataset.index[:].hour, dtype='int32')
# manually specify column names
#dataset.insert(1, 'weekday', weekday)
#dataset.insert(2, 'hour', hour)
#dataset.columns = ['pollution', 'weekday', 'hour', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))

# plot some columns
values = dataset.values
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
# save to file
dataset.to_csv(path + '\\..\\data\\pollution.csv')