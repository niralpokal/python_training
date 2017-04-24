import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/AMD')
df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]
df['daily_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] *100
df['hl_change'] = (df['Adj. High'] - df['Adj. Low'])/ df['Adj. Close'] *100                 

df = df[['Adj. Close', 'daily_change', 'hl_change', 'Adj. Volume']]

forcast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.001 * len(df)))

df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = svm.SVR()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
print(df.head())
