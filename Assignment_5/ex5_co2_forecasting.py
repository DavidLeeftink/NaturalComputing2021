import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

## load CO2 data
CO2_data = loadmat('data/CO2/CO2data.mat')

x, x_test = CO2_data['xtrain'], CO2_data['xtest']
y, y_test = CO2_data['ytrain'], CO2_data['ytest']


# ## Split train in train-validation for time series
tscv = TimeSeriesSplit()
estimators = np.arange(1, 252, step=10)
MSE = np.zeros((2, tscv.get_n_splits(x,y), len(estimators)))

for i, (train_idx, val_idx) in enumerate(tscv.split(x)):
    print(f'Split {i+1}/{tscv.get_n_splits(x,y)}')
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    rf_mse, ada_mse = 0, 0
    for j, n_estimators in enumerate(estimators):

        # Fit random forest
        random_forest = RandomForestRegressor(n_estimators=n_estimators, criterion='mse')
        random_forest.fit(x_train, y_train.reshape(-1))
        y_pred_rf = random_forest.predict(x_val)
        MSE[0,i,j] = mean_squared_error(y_val, y_pred_rf)

        # fit adaboost
        adaboost = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=n_estimators)
        adaboost.fit(x_train, y_train.reshape(-1))
        y_pred_ada = adaboost.predict(x_val)
        MSE[1,i,j] = mean_squared_error(y_val, y_pred_ada)


# Plot MSE as function of maximum depth for both classifiers
mean_mse_rf = np.mean(MSE[0,:,:], axis=0)
mean_mse_ada = np.mean(MSE[1,:,:], axis=0)
plt.figure()
plt.plot(estimators, mean_mse_rf, label='Random Forest')
plt.plot(estimators, mean_mse_ada, label='AdaBoost')
plt.ylabel('Mean Squared Error')
plt.xlabel('Number of estimators')
plt.legend()
plt.savefig('figures/a5_estimors_comparison.png')
plt.show()


# Fit random forest with optimal maximum
random_forest = RandomForestRegressor(n_estimators=20, criterion='mse')
random_forest.fit(x, y.reshape(-1))
y_train_pred_rf = random_forest.predict(x)
y_test_pred_rf = random_forest.predict(x_test)
y_test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)

# fit adaboost with optimal maximum depth
adaboost = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=20)
adaboost.fit(x, y.reshape(-1))
y_train_pred_ada = adaboost.predict(x)
y_test_pred_ada = adaboost.predict(x_test)
y_test_mse_ada = mean_squared_error(y_test, y_test_pred_ada)


## Evaluate testing data
plt.figure()
plt.plot(x, y, label='True data',alpha=0.6, color='gray')
plt.plot(x_test, y_test, label='Testing data',linestyle=':',alpha=0.6, color='gray')
plt.plot(x, y_train_pred_rf, color='r', label='Random forest train prediction')
plt.plot(x_test, y_test_pred_rf, color='r', linestyle=':', label='Random forest test prediction')
plt.plot(x, y_train_pred_ada, color='b', label='AdaBoost train prediction')
plt.plot(x_test, y_test_pred_ada, color='b', linestyle=':', label='Adaboost test prediction')
plt.legend()
plt.savefig('figures/a5_extrapolation_preds.png')
plt.show()
print(f'Random Forest MSE on test set: {round(y_test_mse_rf)}')
print(f'Ada MSE on test set: {round(y_test_mse_ada)}')
