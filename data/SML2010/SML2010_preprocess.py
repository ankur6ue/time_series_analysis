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

dataset = read_csv('NEW-DATA-1.T15.txt', skiprows=[0], header=None, parse_dates = [[0,1]], delim_whitespace=True, dayfirst=True)
# drop columns: 24: day of week, enthalpic motors (cols 19,20,21 in original dataset), 16, 17, 18 - sunlight (east, south), irradiance,
# 13: sun dusk, 11: lighting (room), 9: Rel Humidity (room), 7: CO2 (room), 4: indoor temperature (room)
dataset.drop([23, 20, 19, 18, 17, 16, 15, 12, 10, 8, 6, 3], axis=1, inplace=True)
dataset.set_index("0_1", inplace=True)
weekday = pandas.Series(dataset.index[:].weekday, dtype='int32')
hour = pandas.Series(dataset.index[:].hour, dtype='int32')
min = pandas.Series(dataset.index[:].minute, dtype='int32')
# manually specify column names. Note we must insert values, not the series, otherwise we'll get index matching issues.
# see: https://stackoverflow.com/questions/42041092/python-adding-column-to-dataframe-causes-nan
dataset.insert(1, 'weekday', weekday.values)
dataset.insert(2, 'hour', hour.values)
dataset.insert(3, 'min', min.values)
# legend:
# IDT (Target): Indoor temperature (dinning-room), in ÂºC.
# WDAY: Weekday
# HOUR: hour
# MIN: minutes
# WFT: Weather forecast temperature, in ÂºC.
# CO2: Carbon dioxide in ppm (dinning room).
# REHU: Relative humidity (dinning room), in %.
# LGHT: Lighting (dinning room), in Lux.
# RAIN: Rain, the proportion of the last 15 minutes where rain was detected (a value in range [0,1]).
# WIND: Wind, in m/s.
# SLWF: Sun light in west facade, in Lux.
# ODTM: Outdoor temperature, in ÂºC.
# ODRH: Outdoor relative humidity, in %.
dataset.columns = ['IDT', 'WDAY', 'HOUR', 'MIN', 'WFT', 'CO2', 'REHU',
				   'LGHT', 'RAIN', 'WIND', 'SLWF', 'ODTM', 'ODRH']
dataset.index.name = 'date'
# summarize first 5 rows
print(dataset.head(5))

# plot some columns
values = dataset.values
groups = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
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
dataset.to_csv('SML2010.csv')