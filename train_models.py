from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from tqdm import tqdm
import xgboost as xgb

def estimate_func(X, Y):
    if round(mean([abs(x / y) if x > y else abs(y / x) for x, y in zip(X, Y)]), 3) < 0:
        return 100
    return round(mean([abs(x / y) if x > y else abs(y / x) for x, y in zip(X, Y)]), 3)


def BayesianRidg_func(optuna, X_train, y_train, X_test, y_test, train, best_params):
    if train:
        BayesianRidge_Params = {
             "max_iter" : optuna.suggest_int("max_iter", 25, 650, 25),
             "alpha_1" : optuna.suggest_float("alpha_1", 1e-7, 1e-5),
             "alpha_2" : optuna.suggest_float("alpha_2", 1e-7, 1e-5),
             "lambda_1" : optuna.suggest_float("lambda_1", 1e-7, 1e-5),
             "lambda_2" : optuna.suggest_float("lambda_2", 1e-7, 1e-5)
        }
    else:
        BayesianRidge_Params = {
            "max_iter": best_params['max_iter'],
            "alpha_1": best_params['alpha_1'],
            "alpha_2": best_params['alpha_2'],
            "lambda_1": best_params['lambda_1'],
            "lambda_2": best_params['lambda_2'],
        }

    reg_model = linear_model.BayesianRidge(**BayesianRidge_Params)
    reg_model.fit(X_train, y_train.ravel())
    result = estimate_func(reg_model.predict(X_test), y_test)

    return reg_model, result

def ElasticNet_func(optuna, X_train, y_train, X_test, y_test, train, best_params):
    if train:
        ElasticNet_Params = {
             "alpha" : optuna.suggest_float("alpha", 0.5, 3),
             "l1_ratio" : optuna.suggest_float("l1_ratio", 0, 1),
             "fit_intercept" : optuna.suggest_categorical("fit_intercept", [True, False]),
             "precompute" : optuna.suggest_categorical("precompute", [True, False]),
             "max_iter" : optuna.suggest_int("max_iter", 200, 5000, 200),
             "tol" : optuna.suggest_float("tol", 1e-4, 1e-2),
             "selection" : optuna.suggest_categorical("selection", ["cyclic", "random"]),
        }
    else:
        ElasticNet_Params = {
            "alpha": best_params['alpha'],
            "l1_ratio": best_params['l1_ratio'],
            "fit_intercept": best_params['fit_intercept'],
            "precompute": best_params['precompute'],
            "max_iter": best_params['max_iter'],
            "tol": best_params['tol'],
            "selection": best_params['selection']
        }

    reg_model = ElasticNet(random_state=0, **ElasticNet_Params)
    reg_model.fit(X_train, y_train)
    result = estimate_func(reg_model.predict(X_test), y_test)

    return reg_model, result

def random_forest_func(optuna, X_train, y_train, X_test, y_test, train, best_params):
    if train:
        random_forest_Params = {'bootstrap': optuna.suggest_categorical("bootstrap", [True, False]),
                              'max_depth': optuna.suggest_int('max_depth', 10, 350, 10),
                              'max_features': optuna.suggest_categorical("max_features", ['sqrt', 'log2']),
                              'min_samples_leaf': optuna.suggest_int('min_samples_leaf', 1, 6, 1),
                              'min_samples_split': optuna.suggest_int('min_samples_split', 2, 16, 2),
                              'n_estimators': optuna.suggest_int('n_estimators', 200, 7600, 200)}
    else:
        random_forest_Params = {'bootstrap': best_params['bootstrap'],
                                'max_depth': best_params['max_depth'],
                                'max_features': best_params['max_features'],
                                'min_samples_leaf': best_params['min_samples_leaf'],
                                'min_samples_split': best_params['min_samples_split'],
                                'n_estimators': best_params['n_estimators']}

    reg_model = RandomForestRegressor(random_state=0, **random_forest_Params)
    reg_model.fit(X_train, y_train.ravel())
    result = estimate_func(reg_model.predict(X_test), y_test)

    return reg_model, result

def XgBoost_func(optuna, X_train, y_train, X_test, y_test, train, best_params):
    if train:
        XGBoost_Params = {
            "objective": "reg:squarederror",
            "eval_metric" : "mape",
            "n_estimators": optuna.suggest_int("n_estimators", 100, 3000, 100),
            "verbosity": 0,
            "learning_rate": optuna.suggest_float("learning_rate", 1e-3, 0.1),
            "scale_pos_weight":optuna.suggest_int("scale_pos_weight", 1, 6),
            "max_depth": optuna.suggest_int("max_depth", 1, 15),
            "subsample": optuna.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": optuna.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": optuna.suggest_int("min_child_weight", 1, 20),
        }
    else:
        XGBoost_Params = {
            "objective": "reg:squarederror",
            "eval_metric": "mape",
            "n_estimators": best_params['n_estimators'],
            "verbosity": 0,
            "learning_rate": best_params['learning_rate'],
            "scale_pos_weight": best_params['scale_pos_weight'],
            "max_depth": best_params['max_depth'],
            "subsample": best_params['subsample'],
            "colsample_bytree": best_params['colsample_bytree'],
            "min_child_weight": best_params['min_child_weight'],
        }

    reg_model = xgb.XGBRegressor(random_state=0, **XGBoost_Params)
    reg_model.fit(X_train, y_train)
    result = estimate_func(reg_model.predict(X_test), y_test)

    return reg_model, result

def GradBoostRegr_func(optuna, X_train, y_train, X_test, y_test, train, best_params):
    if train:
        GradBoost_Params = {
            "loss":  optuna.suggest_categorical("loss", ["squared_error", "absolute_error", "huber", "quantile"]),
            "criterion" : optuna.suggest_categorical("criterion", ['friedman_mse', 'squared_error']),
            "min_samples_split" : optuna.suggest_int("min_samples_split", 2, 8, 1),
            "min_samples_leaf" : optuna.suggest_int("min_samples_leaf", 1, 6, 1),
            "alpha" : optuna.suggest_float("alpha", 0.1, 0.99),
            "n_estimators": optuna.suggest_int("n_estimators", 50, 2000, 10),
            "learning_rate": optuna.suggest_float("learning_rate", 1e-3, 0.25),
            "max_depth": optuna.suggest_int("max_depth", 1, 16, 1),
            "subsample": optuna.suggest_float("subsample", 0.05, 1.0)}
    else:
        GradBoost_Params = {
            "loss": best_params['loss'],
            "criterion": best_params['criterion'],
            "min_samples_split": best_params['min_samples_split'],
            "min_samples_leaf": best_params['min_samples_leaf'],
            "alpha": best_params['alpha'],
            "n_estimators": best_params['n_estimators'],
            "learning_rate": best_params['learning_rate'],
            "max_depth": best_params['max_depth'],
            "subsample": best_params['subsample']}

    reg_model = GradientBoostingRegressor(random_state=0, **GradBoost_Params)
    reg_model.fit(X_train, y_train.ravel())
    result = estimate_func(reg_model.predict(X_test), y_test)

    return reg_model, result

def CatBoostRegr_func(optuna, X_train, y_train, X_test, y_test, train, best_params):
    if train:
        CatBoost_Params = {
                "silent" : True,
                "eval_metric" : "MAPE",
                "n_estimators": optuna.suggest_int("n_estimators", 100, 4000, 100),
                "learning_rate": optuna.suggest_float("learning_rate", 1e-3, 0.1),
                "max_depth": optuna.suggest_int("max_depth", 1, 15),
                "subsample": optuna.suggest_float("subsample", 0.15, 1.0),
            }
    else:
        CatBoost_Params = {
                "silent": True,
                "eval_metric": "MAPE",
                "n_estimators": best_params['n_estimators'],
                "learning_rate": best_params['learning_rate'],
                "max_depth": best_params['max_depth'],
                "subsample": best_params['subsample'],
            }

    reg_model = CatBoostRegressor(random_state=0, **CatBoost_Params)
    reg_model.fit(X_train, y_train)
    result = estimate_func(reg_model.predict(X_test), y_test)

    return reg_model, result

def NeuralNetTorch_func(optuna, X_train, y_train, X_test, y_test, train, best_params):
    if train:
        num_of_epochs = optuna.suggest_int('num_of_epochs', 5000, 80000, 5000)
        lr = optuna.suggest_float('lr', 0.001, 0.1, step=0.001)
    else:
        num_of_epochs = best_params['num_of_epochs']
        lr = best_params['lr']

    class linearRegression(nn.Module):
        def __init__(self, input_dim):
            super(linearRegression, self).__init__()
            self.model = nn.Sequential(
                         nn.Linear(input_dim, 64),
                         nn.ReLU(),
                         nn.Linear(64, 40),
                         nn.ReLU(),
                         nn.Linear(40, 25),
                         nn.ReLU(),
                         nn.Linear(25, 1))
        def forward(self, d):
            out = self.model(d)
            return out

        def predict(self, input):
            input = torch.from_numpy(input).float()
            with torch.no_grad():
                out = self.model(input)
                return out

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()

    if type(y_test) != np.ndarray:
        y_test = y_test.to_numpy()

    y_test = torch.from_numpy(y_test).float()


    input_dim = X_train.shape[1]
    torch.manual_seed(42)
    reg_model = linearRegression(input_dim)

    loss = nn.MSELoss()
    optimizers = optim.Adam(params=reg_model.parameters(), lr=lr)

    num_of_epochs = num_of_epochs
    for i in tqdm(range(num_of_epochs)):

        y_train_prediction = reg_model(X_train)
        loss_value = loss(y_train_prediction.squeeze(), y_train)
        optimizers.zero_grad()
        loss_value.backward()
        optimizers.step()

    with torch.no_grad():
        result = estimate_func(reg_model.forward(X_test).numpy().flatten(), y_test.numpy().flatten())

    return reg_model, result
