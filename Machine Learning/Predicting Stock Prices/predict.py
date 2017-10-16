import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

data = pd.read_csv('sphist.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

data['five_day_average'] = pd.Series(index=data.index)
data['five_day_sd'] = pd.Series(index=data.index)
data['thirty_day_average'] = pd.Series(index=data.index)
data['year_average'] = pd.Series(index=data.index)
data['year_sd'] = pd.Series(index=data.index)
data['five_year_average_ratio'] = pd.Series(index=data.index)
data['five_year_sd_ratio'] = pd.Series(index=data.index)
#Compute the averages and SDs - shifting each series forward one to account for 
#inclusion of current date in average
data['five_day_average'] = pd.rolling_mean(data['Close'], 5)
data['five_day_average'] = data['five_day_average'].shift(1)

data['five_day_sd'] = pd.rolling_std(data['Close'], 5)
data['five_day_sd'] = data['five_day_sd'].shift(1)

data['thirty_day_average'] = pd.rolling_mean(data['Close'], 30)
data['thirty_day_average'] = data['thirty_day_average'].shift(1)

data['year_average'] = pd.rolling_mean(data['Close'], 365)
data['year_average'] = data['year_average'].shift(1)

data['year_sd'] = pd.rolling_std(data['Close'], 365)
data['year_sd'] = data['year_sd'].shift(1)

#Calculate the ratio between the five day average and the year average
data['five_year_average_ratio'] = data['five_day_average']/data['year_average']
data['five_year_sd_ratio'] = data['five_day_sd']/data['year_sd']

#Remove any data pre 1951
data = data[data["Date"] > datetime(year=1951, month=1, day=2)]
#Because there isn't exactly 365 trading days in a year
#need to remove any remaining missing values
data.dropna(axis=0, inplace=True)

#Generate test and train data, wwill train on data older than 2013, to make predictions
#after 2013
train = data[data["Date"] < datetime(year=2013, month=1, day=1)]
test = data[data["Date"] > datetime(year=2013, month=1, day=1)]

#I shall use Mean Absolute Error as my error metric, which will grant me an
#intuition for how far 'off' the correct price I am
exclude_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date']
model = LinearRegression()
model.fit(train.drop(exclude_columns, axis=1), train['Close'])
predictions = model.predict(test.drop(exclude_columns, axis=1))
error = mae(predictions, test['Close'])
#print(data.head(10))
print(error)