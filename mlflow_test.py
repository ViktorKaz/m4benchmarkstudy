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
import mlflow 
from load_dataset import LoadM4Dataset

from sklearn.metrics import accuracy_score
dts_number = 0
data = LoadM4Dataset(main_dir_path="Dataset", dts_frequency='Daily')
y_train, y_test =data.load_dts_number(dts_number=dts_number)


regressor = RandomForestRegressor()
forecaster = make_reduction(regressor, window_length=15, strategy="recursive")

mlflow.set_tracking_uri("http://localhost:5000")


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

forecaster_pipe = NetworkPipelineForecaster([
    ('forecaster',forecaster, {'fit':{'y': 'original_y', 'fh':'original_fh'},
                                'predict':{'fh':'original_fh'}
    }),
    ('converter', converter, {'fit':None,
                             'predict':{'y1':y_train,'y2':'forecaster'}})
])

with mlflow.start_run():
    model_name="ForecastingPipeline"
    dataset_name = f"daily_{dts_number}"
    fh = ForecastingHorizon(np.arange(1,len(y_test)+1))
    y_true_dc = converter(y_train, y_test)
    forecaster_pipe.fit(y=y_train, fh=fh)
    out = forecaster_pipe.predict()
    acc = accuracy_score(y_true=y_true_dc.values, y_pred=out)
    mlflow.log_metric('accuracy', acc)
    mlflow.sklearn.log_model(sk_model=forecaster_pipe, artifact_path="models",registered_model_name=f"model:{model_name}_dataset:{dataset_name}")
    print(acc)


# #Directional change reduced to supervised classifcation

# def time_series_to_tabular(y,window_length, fh, return_value):
#     """
#     Converts a pd.Series to tabular format. Lables target variable 1 for up 0 for down
    
#     Parameters
#     ----------
#     y : pd.Series
#         input time series
#     window_length : int
#         number of features for each raw of X
#     fh : ForecastingHorizon
#         ForecastingHorizon object
#     stage : str 
#         must be `fit` or `predict`
#     return_value : string
#         acceptable values `X` and `y` only
    
#     Returns
#     -------
#         tuple of (numpy.ndarray, numpy.ndarray) where the first element are the target (y) variable and the second element are the features (X)
#     """
#     fh=fh
#     y_tmp,x_dc = _sliding_window_transform(y_train,window_length=window_length,fh=fh)
#     y_dc = np.zeros(len(y_tmp))
#     y_tmp = y_tmp.reshape(x_dc[:,-1].shape)
#     y_mask = (x_dc[:,-1] > y_tmp) #up observations
#     y_dc[y_mask]=1
#     if return_value == 'X':
#         return x_dc
#     if return_value == 'y':
#         return y_dc
    
# def concatenator(y_train, y_test, window_length):
#     return pd.concat([y_train[-window_length-1:-1],y_test])

# window_length = 5
# classifier_pipe = NetworkPipelineForecaster([
#     ('concatenator',concatenator, {'fit':None, 
#                                     'predict':{'y_train': y_train, 'y_test': y_test, 'window_length':window_length}}),
#     ('time_series_to_tabular_x', time_series_to_tabular,{
#             'fit':{'y':'original_y', 'window_length':window_length, 'fh': 'original_fh', 'return_value':'X'},
#             'predict': {'y':'concatenator', 'window_length':window_length, 'fh': 'original_fh', 'return_value':'X'} }),
#     ('time_series_to_tabular_y', time_series_to_tabular,{
#             'fit':{'y':'original_y', 'window_length':window_length, 'fh': 'original_fh', 'return_value':'y'},
#             'predict': None }),
#     ('classifier', RandomForestClassifier(), {'fit':{'X': 'time_series_to_tabular_x', 'y': 'time_series_to_tabular_y' },
#                                               'predict': {'X': 'time_series_to_tabular_x'} })
# ])

# fh=ForecastingHorizon([1])
# classifier_pipe.fit(y_train, fh=fh)
# out = classifier_pipe.predict()
# print(f'classifier{out}')


# #Reduce directional change to forecasting with exogenous variables
# fh = ForecastingHorizon(np.arange(1,len(y_test)+1))

# def reshape(y):
#     return y.values.reshape(-1,1)
# hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)

# def decoder(est,y):
#     X = est.decode(y.values.reshape(-1,1))[1]
#     X = pd.Series(X)
#     X.index = y.index
#     return X


# exogenous_pipe = NetworkPipelineForecaster([
#     ('reshape', reshape,  {'fit':{'y':'original_y'},
#                             'predict': {'y':y_test} }),
#     ('hmm',hmm_model, {'X':'reshape'}),
#     ('fitted_hmm', 'get_fitted_estimator', {'fit': None, 'predict':{'step_name':'hmm'}}), #discuss this interface for getting fitted models
#     ('decoder',decoder, {'fit':{'est':'hmm', 'y':'original_y'},
#                          'predict':{'est':'fitted_hmm', 'y':y_test}   }),
#     ('regressor', make_reduction(RandomForestRegressor(), window_length=5, strategy='recursive'), 
#                                                                         {'fit':{'y': 'original_y', 'X': 'decoder', 'fh':'original_fh'},
#                                                                          'predict':{'X': 'decoder'} } ),
#      ('converter', converter, {'fit':None,
#                              'predict':{'y1':y_train,'y2':'regressor'}})
# ])

# exogenous_pipe.fit(y_train, fh=fh)

# out = exogenous_pipe.predict()

# print(out)
