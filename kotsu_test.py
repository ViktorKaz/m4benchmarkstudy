from hashlib import sha1
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction
import numpy as np
from sktime.forecasting.compose import NetworkPipelineForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.datasets import load_shampoo_sales

from sktime.forecasting.compose._reduce import _sliding_window_transform
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
import kotsu 
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanSquaredPercentageError
from sktime.forecasting.model_evaluation import evaluate
import mlflow

# LOAD DATASET

dts_number = 0

daily_train = pd.read_csv('Dataset/Train/Daily-train.csv', skiprows=dts_number, nrows=1)
daily_test = pd.read_csv('Dataset/Test/Daily-test.csv', skiprows=dts_number, nrows=1)
info = pd.read_csv('Dataset/M4-info.csv', skiprows=dts_number, nrows=1)
y_train = daily_train.dropna(axis=1)
y_train  = pd.DataFrame(data=y_train.values[0][1:-1])

series_name = daily_train.iloc[0,0]
start_index = info.iloc[-1,-1]
index = pd.date_range(start_index,periods=y_train.shape[0],freq='D')
y_train.index = index
y_train= y_train.astype('float')

y_test = daily_test.dropna(axis=1)
y_test = pd.DataFrame(data=y_test.values[0][1:-1])
start_index = y_train.index[-1]  + pd.DateOffset(1)
index = pd.date_range(start_index,periods=y_test.shape[0],freq='D')
y_test.index = index
y_test= y_test.astype('float')

# DEFINE MODEL

regressor = RandomForestRegressor()
forecaster = make_reduction(regressor, window_length=15, strategy="recursive")

fh = ForecastingHorizon(np.arange(1,len(y_test)+1))


# Directional change reduced to forecasting
def converter(y1,y2):
    """
    Converts regression output to directional change output where 1 means up 0 means down
    Parameters
    ----------
    y1 : pd.Series
        Series preceeding y2. It is used only to calculate the value of the first item in y2 (up down relative to the last value of y1)
    y2 : pd.Series
    
    Returns
    -------
        pd.Series
    """
    concat_y1_y2 = pd.concat([y1[-2:-1],y2])
    dc = concat_y1_y2.shift(-1) > concat_y1_y2
    dc = dc[0:-1]
    dc[dc==True] =1
    dc[dc==False] =0
    dc = dc.astype('int')
    return dc

steps = ([
    ('forecaster',forecaster, {'fit':{'y': 'original_y', 'fh':'original_fh'},
                                'predict':{'fh':'original_fh'}
    }),
    ('converter', converter, {'fit':None,
                             'predict':{'y1':y_train,'y2':'forecaster'}})
])

# BENCHMARK WITH MLFLOW

model_registry_m4benchmarking = kotsu.registration.ModelRegistry()

model_registry_m4benchmarking.register(
    id="BenchmarkingPipe-v1",
    entry_point=NetworkPipelineForecaster,
    kwargs={'steps':steps}
)

validation_registry_m4benchmarking = kotsu.registration.ValidationRegistry()


def factory():
    """Factory for airline cross validation."""

    def airline_cross_validation(model):
        """Airline dataset cross validation."""
        error = MeanSquaredPercentageError()
        error(y_pred=y_test,y_true=y_test)
        return error

    return airline_cross_validation


validation_registry_m4benchmarking.register(
    id="first_dataset-v1",
    entry_point=factory,
)

kotsu.run.run(
    model_registry_m4benchmarking,
    validation_registry_m4benchmarking,
    "./kotsu_results_interval.csv",
)