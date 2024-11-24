import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold,cross_validate,train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import warnings
from sklearn.metrics import r2_score

# df = pd.read_excel(r'C:\Users\Administrator\Desktop\shuju\T80.xlsx', sheet_name='nip_chuli_4')
df = pd.read_excel(r'C:\Users\Administrator\Desktop\shuju\T80.xlsx', sheet_name='pin_chuli')
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1412)
cv = KFold(n_splits=10, shuffle=True, random_state=1412)
data_dmatrix = xgb.DMatrix(X, Y)

# Define cross-validation result function
def MSE(result, name):
    return abs(result[name].mean())

def RMSE(result, name):
    return abs(result[name].mean())

# GBDT
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 1000,  # Number of weak classifiers
          'max_depth': 10,        # Maximum depth of the weak classifiers (CART regression tree)
          'min_samples_split': 3, # Minimum number of samples required to split an internal node
          'learning_rate': 0.001, # Learning rate
          'loss': 'squared_error'} # Loss function: Mean squared error loss function
GBDTreg = GradientBoostingRegressor(**params, random_state=1412)
GBDTreg.fit(X_train, Y_train)

# Decision Tree
dtr = DTR(criterion='friedman_mse', splitter='best', random_state=42,
          max_depth=24, min_samples_split=2)
result_dtr = cross_validate(dtr, X, Y, cv=cv, n_jobs=-1,
                            scoring='neg_root_mean_squared_error',
                            verbose=True,
                            return_train_score=True)
clf = dtr.fit(X_train, Y_train)

# Random Forest
rfr = RFR(random_state=1412)
result_dtr = cross_validate(rfr, X, Y, cv=cv, n_jobs=-1,
                            scoring='neg_root_mean_squared_error',
                            verbose=True,
                            return_train_score=True)
clf = rfr.fit(X_train, Y_train)

# Plot cross-validation results
plt.figure(dpi=150)
plt.plot(range(1, 11), result_dtr['train_score'], 'b-', label='train')
plt.plot(range(1, 11), result_dtr['test_score'], 'r--', label='test')
plt.legend(loc=1)
plt.title('Random Forest Model RMSE: 538')
plt.show()
print(RMSE(result_dtr, 'train_score'))
print(RMSE(result_dtr, 'test_score'))

# XGBoost model
param_xgb = {'max_depth': 6, 'seed': 1412}
xgb_cv = xgb.cv(param_xgb, data_dmatrix, num_boost_round=20, nfold=5, seed=1412)
plt.figure(dpi=150)
plt.plot(xgb_cv.iloc[:, 0], 'r-', label='train rmse')
plt.plot(xgb_cv.iloc[:, 2], 'b--', label='test rmse')
plt.legend(loc=1)
plt.grid()
plt.title('XGBoost Initial Results')
plt.show()

# Plot learning curve
test = []
for i in range(1, 100, 2):
    params = {'max_depth': 6,
              'seed': 1412,
              'eta': 0.1,
              'nthread': 16}
    result = xgb.cv(params, data_dmatrix, num_boost_round=i, nfold=5, seed=1412)
    test.append(result.iloc[-1, 2])
plt.plot(range(1, 100, 2), test)
plt.title('num_boost_round')  # num_boost_round(20), lambda(10)
plt.show()

import hyperopt
from hyperopt import tpe, fmin, hp, Trials, partial
from hyperopt.early_stop import no_progress_loss

# Build the objective function
def hyperopt_xgboost(params):
    paramsforxgb = {
        'eta': params['eta'],
        'booster': 'dart',
        'colsample_bytree': params['colsample_bytree'],
        'colsample_bynode': params['colsample_bynode'],
        'gamma': params['gamma'],
        'lambda': params['lambda'],
        'min_child_weight': params['min_child_weight'],
        'max_depth': int(params['max_depth']),
        'subsample': params['subsample'],
        'objective': params['objective'],
        'rate_drop': params['rate_drop'],
        'nthread': 20,
        'verbosity': 0,
        'seed': 1412
    }
    result_xgb_cv = xgb.cv(paramsforxgb, data_dmatrix, nfold=5, seed=1412,
                           num_boost_round=int(params['num_boost_round']),
                           metrics=('rmse'))
    return result_xgb_cv.iloc[-1, 2]

# Define the parameter space
params_tpe_xgb = {
    'eta': hp.uniform('eta', 0.01, 3),
    'num_boost_round': hp.quniform('num_boost_round', 1, 50, 1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1, 0.1),
    'colsample_bynode': hp.quniform('colsample_bynode', 0.1, 1, 0.1),
    'gamma': hp.quniform('gamma', 0, 1, 0.001),
    'lambda': hp.quniform('lambda', 0, 1, 0.001),
    'min_child_weight': hp.quniform('min_child_weight', 0, 100, 1),
    'max_depth': hp.quniform('max_depth', 3, 35, 1),
    'subsample': hp.quniform('subsample', 0, 1, 0.05),
    'objective': hp.choice('objective', ['reg:squarederror', 'reg:squaredlogerror']),
    'rate_drop': hp.quniform('rate_drop', 0, 1, 0.05)
}

# Bayesian optimization
def params_hyperopt_xgb(max_evals=1000):
    trials = Trials()
    early_stop_fn = no_progress_loss(100)
    algo = partial(tpe.suggest, n_startup_jobs=50, n_EI_candidates=20)
    best_params = fmin(hyperopt_xgboost,
                       space=params_tpe_xgb,
                       algo=tpe.suggest,
                       max_evals=max_evals,
                       trials=trials,
                       verbose=True)
    print('Best params:', best_params)
    return best_params, trials

if __name__ == '__main__':
    params_hyperopt_xgb()

# Validate the optimization results
# pin
paramsforxgb_tpe = {'colsample_bynode': 0.9,
                    'colsample_bytree': 0.8,
                    'eta': 1.272406552587308,
                    'gamma': 0.845,
                    'lambda': 0.8180000000000001,
                    'max_depth': 20,
                    'min_child_weight': 0.0,
                    'num_boost_round': 40.0,
                    'objective': 'reg:squarederror',
                    'rate_drop': 0.45,
                    'subsample': 0.9,
                    # 'booster': 'dart',
                    'verbosity': 0,
                    'seed': 1412
    }
xgb_cv_tpe = xgb.cv(paramsforxgb_tpe, data_dmatrix, nfold=5, seed=1412,
                    num_boost_round=40,
                    metrics=('rmse'))

# nip
paramsforxgb_tpe = {'colsample_bynode': 0.6000000000000001,
                    'colsample_bytree': 0.6000000000000001,
                    'eta': 1.1504930337517472,
                    'gamma': 0.646,
                    'lambda': 0.105,
                    'max_depth': 34,
                    'min_child_weight': 0.0,
                    'num_boost_round': 50.0,
                    'objective': 'reg:squarederror',
                    'rate_drop': 0.55,
                    'subsample': 1.0,
                    # 'booster': 'dart',
                    'verbosity': 0,
                    'seed': 1412
    }
xgb_cv_tpe = xgb.cv(paramsforxgb_tpe, data_dmatrix, nfold=5, seed=1412,
                    num_boost_round=50,
                    metrics=('rmse'))

plt.figure(dpi=150)
plt.plot(range(1, 51), xgb_cv_tpe['train-rmse-std'], 'r-', label='train_score')
plt.plot(range(1, 51), xgb_cv_tpe['test-rmse-std'], 'b--', label='test_score')
plt.legend(loc=1)
print(xgb_cv_tpe)
print('*' * 30)
print(xgb_cv_tpe.iloc[-1, 2])
plt.show()

xgb_tpe = xgb.train(paramsforxgb_tpe, data_dmatrix,
                    num_boost_round=40)
fig, ax = plt.subplots(figsize=(10, 10), dpi=1000)
xgb.plot_tree(xgb_tpe, ax=ax)
xgb.plot_importance(xgb_tpe, grid=False, max_num_features=30)
plt.show()
print(xgb_tpe.get_fscore())

y_pred = xgb_tpe.predict(xgb.DMatrix(X_test))
data1 = pd.DataFrame(y_pred, Y_test)
data1.to_csv('predict.csv')
y_pred_train = xgb_tpe.predict(xgb.DMatrix(X_train))
data2 = pd.DataFrame(y_pred_train, Y_train)
data2.to_csv('predict_train.csv')

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
rmse1 = np.sqrt(mean_squared_error(Y_train, y_pred_train))
print("RMSE_test: %f" % (rmse))
print("RMSE_train: %f" % (rmse1))
r2 = r2_score(Y_test, y_pred)
print(r2)

# SHAP analysis
cols = X.columns
explainer = shap.TreeExplainer(GBDTreg)
shap_values = explainer.shap_values(X[cols], check_additivity=False)
y_base = explainer.expected_value
X['pred'] = xgb_tpe.predict(X[cols])
j = 10
player_explainer = pd.DataFrame()
player_explainer['feature'] = cols
player_explainer['feature_value'] = df[cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
print('y_base + sum_of_shap_values: %.2f'%(y_base + player_explainer['shap_value'].sum()))
print('y_pred: %.2f'%(X['pred'].iloc[j]))
shap.initjs()
shap.force_plot(y_base, shap_values[j], X[cols].iloc[j], matplotlib=True, link='identity')
shap.summary_plot(shap_values, X[cols], max_display=15)
shap.summary_plot(shap_values, X[cols], plot_type="bar", max_display=15)
shap.dependence_plot('HTL_additives_compounds', shap_values, X[cols], interaction_index=None, show=True)
shap_interaction_values = shap.TreeExplainer(xgb_tpe).shap_interaction_values(X[cols])
shap.summary_plot(shap_interaction_values, X[cols], max_display=10)