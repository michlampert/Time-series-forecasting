import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class MultiStepForecaster:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size
        self.step_size_for_prediction = step_size
        self.models = [xgb.XGBRegressor() for i in range(step_size)]

    def fit(self, X, y):
        X_data = []
        y_data = []
        for i in range(len(y) - self.window_size - self.step_size):
            X_data.append(X[i: i + self.window_size])
            y_data.append(y[i + self.window_size: i +
                          self.window_size + self.step_size])

        X_windows = np.array(X_data)
        X_windows = X_windows.reshape((len(X_windows), -1))
        y_windows = np.array(y_data)
        y_windows = y_windows.reshape((len(y_windows), -1))

        for i, model in enumerate(self.models):
            model.fit(X_windows, y_windows[:, i])

        return X_windows, y_windows

    def error(self, predictions, y):
        return mean_squared_error(predictions, np.array(y).flatten()[-self.step_size_for_prediction:])

    def predict_one(self, X, y = None):

        X_windows = np.array([X[-self.window_size:]])
        X_windows = X_windows.reshape((len(X_windows), -1))

        predictions = np.array(list(map(lambda model: model.predict(X_windows), self.models))).flatten()
        
        if y is not None:
            print("MSR:", self.error(predictions, y))

        return predictions

    def predict(self, X, y = None): return self.predict_one(X, y)

    def predict_multiple(self, Xs, ys):
        result = [self.predict(X) for X in Xs]

        err = np.mean([self.error(prediction, y) for prediction, y in zip(result, ys)])
        return result, err

    def plot(self, X, y, real_data_name=None, *, df = None):
        if real_data_name and df is not None:
            df = df.copy()
            X = df
            y = df[real_data_name]
        else:
            df = y.loc[:]
        df["predicted"] = np.concatenate((np.empty((len(y) - self.step_size_for_prediction,)) * np.nan, self.predict(X[-self.window_size:], y)))
        if real_data_name:
            df[[real_data_name, "predicted"]][-self.step_size_for_prediction * 3:].plot()
        else:
            df[-self.step_size_for_prediction * 3:].plot()

class OneStepForecaster(MultiStepForecaster):
    def __init__(self, window_size, step_size):
        super().__init__(window_size, 1)
        self.step_size_for_prediction = step_size

    def predict(self, X, y = None):

        x = np.array(X).copy()

        predictions = np.array([])

        for i in range(self.step_size_for_prediction):
            X_windows = np.array([np.append(x, predictions)[-self.window_size:]])
            X_windows = X_windows.reshape((len(X_windows), -1))

            prediction = np.array(list(map(lambda model: model.predict(X_windows), self.models))).flatten()
            predictions = np.append(predictions, prediction)

        if y is not None:
            print("MSR:", self.error(predictions, y))

        return predictions