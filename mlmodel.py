import numpy as np
import pandas as pd  
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

fuel_data = pd.read_csv('./FuelConsumption.csv')
# print(fuel_data.head())

x = fuel_data[['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y = fuel_data['CO2EMISSIONS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

poly_features = PolynomialFeatures(degree=4, include_bias=False)
std_scaler = StandardScaler()
lin_reg = LinearRegression()
poly_reg = make_pipeline(poly_features, std_scaler, lin_reg)
poly_reg.fit(x_train, y_train)

# print('Correlation Train =', poly_reg.score(x_train, y_train))
# print('Correlation Test =', poly_reg.score(x_test, y_test))

predictions_test = poly_reg.predict(x_test)

import pickle

filename = 'model.pickle'
pickle.dump(poly_reg, open(filename, 'wb'))