from sktime.forecasting.compose import NetworkPipelineForecaster
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from hmmlearn import hmm

from abc import ABC, abstractmethod

class BasePipe(ABC):
    def get_model_name(self):
        return self.model_name
    
    def get_model(self):
        return self.model
    
    @abstractmethod
    def get_fh(self):
        pass
    @abstractmethod
    def build(self):
        pass
# Directional change reduced to forecasting
class RegressorPipe(BasePipe):
    """
    Parameters
    ----------
    model_name : str
        name of model
    """
    def __init__(self, model_name):
        self.model_name = model_name

    def get_fh(self, **kwargs):
        return ForecastingHorizon(np.arange(1,len(kwargs['y_test'])+1))

    def build(self, **kwargs):
        regressor = RandomForestRegressor()
        forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
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
                                    'predict':{'y1':kwargs['y_train'],'y2':'forecaster'}})
        ])
        self.model = forecaster
        return forecaster


# with mlflow.start_run():
#     model_name="ForecastingPipeline"
#     dataset_name = f"daily_{dts_number}"
#     fh = ForecastingHorizon(np.arange(1,len(y_test)+1))
#     y_true_dc = converter(y_train, y_test)
#     forecaster_pipe.fit(y=y_train, fh=fh)
#     out = forecaster_pipe.predict()
#     acc = accuracy_score(y_true=y_true_dc.values, y_pred=out)
#     mlflow.log_metric('accuracy', acc)
#     mlflow.sklearn.log_model(sk_model=forecaster_pipe, artifact_path="models",registered_model_name=f"model:{model_name}_dataset:{dataset_name}")
#     print(acc)


#Directional change reduced to supervised classifcation
class SupervisedClassificationPipe(BasePipe):
    """
    Parameters
    ----------
    model_name : str
        name of model
    """
    def __init__(self, model_name):
        self.model_name = model_name
    def time_series_to_tabular(self,y,window_length, fh, return_value):
        """
        Converts a pd.Series to tabular format. Lables target variable 1 for up 0 for down
        
        Parameters
        ----------
        y : pd.Series
            input time series
        window_length : int
            number of features for each raw of X
        fh : ForecastingHorizon
            ForecastingHorizon object
        stage : str 
            must be `fit` or `predict`
        return_value : string
            acceptable values `X` and `y` only
        
        Returns
        -------
            tuple of (numpy.ndarray, numpy.ndarray) where the first element are the target (y) variable and the second element are the features (X)
        """
        fh=fh
        y_tmp,x_dc = _sliding_window_transform(y,window_length=window_length,fh=fh)
        y_dc = np.zeros(len(y_tmp))
        y_tmp = y_tmp.reshape(x_dc[:,-1].shape)
        y_mask = (x_dc[:,-1] > y_tmp) #up observations
        y_dc[y_mask]=1
        if return_value == 'X':
            return x_dc
        if return_value == 'y':
            return y_dc
        
    def concatenator(self,y_train, y_test, window_length):
        return pd.concat([y_train[-window_length-1:-1],y_test])
    def build(self, **kwargs):
        window_length = 5
        classifier_pipe = NetworkPipelineForecaster([
            ('concatenator',self.concatenator, {'fit':None, 
                                            'predict':{'y_train': kwargs['y_train'], 'y_test': kwargs['y_test'], 'window_length':window_length}}),
            ('time_series_to_tabular_x', self.time_series_to_tabular,{
                    'fit':{'y':'original_y', 'window_length':window_length, 'fh': 'original_fh', 'return_value':'X'},
                    'predict': {'y':'concatenator', 'window_length':window_length, 'fh': 'original_fh', 'return_value':'X'} }),
            ('time_series_to_tabular_y', self.time_series_to_tabular,{
                    'fit':{'y':'original_y', 'window_length':window_length, 'fh': 'original_fh', 'return_value':'y'},
                    'predict': None }),
            ('classifier', RandomForestClassifier(), {'fit':{'X': 'time_series_to_tabular_x', 'y': 'time_series_to_tabular_y' },
                                                    'predict': {'X': 'time_series_to_tabular_x'} })
        ])
        self.model = classifier_pipe
        return classifier_pipe
    def get_fh(self, **kwargs):
        return ForecastingHorizon([1])
# fh=ForecastingHorizon([1])
# classifier_pipe.fit(y_train, fh=fh)
# out = classifier_pipe.predict()
# print(f'classifier{out}')


# #Reduce directional change to forecasting with exogenous variables
# fh = ForecastingHorizon(np.arange(1,len(y_test)+1))

class ExogenousPipe(BasePipe):
    def __init__(self, model_name):
        self.model_name = model_name
    def reshape(self,y):
        return y.values.reshape(-1,1)
    
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


    def decoder(self,est,y):
        X = est.decode(y.values.reshape(-1,1))[1]
        X = pd.Series(X)
        X.index = y.index
        return X

    def build(self, **kwargs):
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)
        exogenous_pipe = NetworkPipelineForecaster([
            ('reshape', self.reshape,  {'fit':{'y':'original_y'},
                                    'predict': {'y':kwargs['y_test']} }),
            ('hmm',hmm_model, {'X':'reshape'}),
            ('fitted_hmm', 'get_fitted_estimator', {'fit': None, 'predict':{'step_name':'hmm'}}), #discuss this interface for getting fitted models
            ('decoder',self.decoder, {'fit':{'est':'hmm', 'y':'original_y'},
                                'predict':{'est':'fitted_hmm', 'y':kwargs['y_test']}   }),
            ('regressor', make_reduction(RandomForestRegressor(), window_length=5, strategy='recursive'), 
                                                                                {'fit':{'y': 'original_y', 'X': 'decoder', 'fh':'original_fh'},
                                                                                'predict':{'X': 'decoder'} } ),
            ('converter', self.converter, {'fit':None,
                                    'predict':{'y1':kwargs['y_train'],'y2':'regressor'}})
        ])
        self.model = exogenous_pipe
        return exogenous_pipe
    def get_fh(self, **kwargs):
        return ForecastingHorizon([1])
# exogenous_pipe.fit(y_train, fh=fh)

# out = exogenous_pipe.predict()

# print(out)
