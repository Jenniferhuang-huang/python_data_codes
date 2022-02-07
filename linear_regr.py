"""
Use both single and multiple linear regression models to predict CO2 emission.
Independent variables including engine size, cylinders of cars and fuel consumption
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# get an overview of data and clean up data
df = pd.read_csv("FuelConsumption.csv")
df.head()
df.describe()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
# plot each parameter
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
# plot fuel consumption vs emission
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("fuel consumption")
plt.ylabel("Emission")
plt.show()
# plot engine size vs emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
# plot cylinders vs emission
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()
# Linear regression model between engine size and emission
# use 80% data for training and 20% for testing
train_set = np.random.rand(len(df)) < 0.8
train_engine = cdf[train_set]
test_engine = cdf[~train_set]
# build model
regr_engine = linear_model.LinearRegression()
train_x_engine = np.asanyarray(train_engine[['ENGINESIZE']])
train_y_engine = np.asanyarray(train_engine[['CO2EMISSIONS']])
regr_engine.fit(train_x_engine, train_y_engine)
# The coefficients
print('Coefficients: ', regr_engine.coef_)
print('Intercept: ', regr_engine.intercept_)
# plot fit line
plt.scatter(train_engine.ENGINESIZE, train_engine.CO2EMISSIONS, color='blue')
plt.plot(train_x_engine, regr_engine.coef_[0][0] * train_x_engine + regr_engine.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
# use test data to evaluate model
test_x_engine = np.asanyarray(test_engine[['ENGINESIZE']])
test_y_engine = np.asanyarray(test_engine[['CO2EMISSIONS']])
y_hat_engine = regr_engine.predict(test_x_engine)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat_engine - test_y_engine)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat_engine - test_y_engine) ** 2))
print("Root mean square error: %.2f" % np.sqrt(np.mean((y_hat_engine - test_y_engine) ** 2)))
print("R2-score: %.2f" % r2_score(test_y_engine, y_hat_engine))

# Linear regression model between fuel consumption and emission
# use 80% data for training and 20% for testing
train_set_fuel = np.random.rand(len(df)) < 0.8
train_fuel = cdf[train_set_fuel]
test_fuel = cdf[~train_set_fuel]
# build model
regr_fuel = linear_model.LinearRegression()
train_x_fuel = np.asanyarray(train_fuel[['FUELCONSUMPTION_COMB']])
train_y_fuel = np.asanyarray(train_fuel[['CO2EMISSIONS']])
regr_fuel.fit(train_x_fuel, train_y_fuel)
# The coefficients
print('Coefficients: ', regr_fuel.coef_)
print('Intercept: ', regr_fuel.intercept_)
# plot fit line
plt.scatter(train_fuel.FUELCONSUMPTION_COMB, train_fuel.CO2EMISSIONS, color='blue')
plt.plot(train_x_fuel, regr_fuel.coef_[0][0] * train_x_fuel + regr_fuel.intercept_[0], '-r')
plt.xlabel("FEULCONSUMPTION")
plt.ylabel("Emission")
# use test data to evaluate model
test_x_fuel = np.asanyarray(test_fuel[['FUELCONSUMPTION_COMB']])
test_y_fuel = np.asanyarray(test_fuel[['CO2EMISSIONS']])
y_hat_fuel = regr_fuel.predict(test_x_fuel)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat_fuel - test_y_fuel)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat_fuel - test_y_fuel) ** 2))
print("Root mean sqare error: %.2f" % np.sqrt(np.mean((y_hat_fuel - test_y_fuel) ** 2)))
print("R2-score: %.2f" % r2_score(test_y_fuel, y_hat_fuel))

# Multiple linear regression model to predict CO2 emission using features: engine size, fuel consumption and cylinders
# use 80% data for training and 20% for testing
msk_multi = np.random.rand(len(df)) < 0.8
train_multi = cdf[msk_multi]
test_multi = cdf[~msk_multi]
# build model
regr_multi = linear_model.LinearRegression()
train_x_multi = np.asanyarray(train_multi[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y_multi = np.asanyarray(train_multi[['CO2EMISSIONS']])
regr_multi.fit(train_x_multi, train_y_multi)
# The coefficients
print('Coefficients: ', regr_multi.coef_)
# use test data to evaluate model
y_hat_multi = regr_multi.predict(test_multi[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_x_multi = np.asanyarray(test_multi[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y_multi = np.asanyarray(test_multi[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat_multi - test_y_multi) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr_multi.score(test_x_multi, test_y_multi))
