from sktime.forecasting.compose import NetworkPipelineForecaster
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import _sliding_window_transform
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from hmmlearn import hmm
from utils import converter
from abc import ABC, abstractmethod
from sktime.classification.base import BaseClassifier
from sktime.base import BaseEstimator
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
    regressor : object
        sktime compatible regressor
    window_length : int
        parameter to sktime.forecasting.compose.make_reduction

    """
    def __init__(self, model_name, regressor, window_length):
        self.model_name = model_name
        self.regressor = regressor
        self.window_length = window_length

    def get_fh(self, **kwargs):
        return ForecastingHorizon(np.arange(1,len(kwargs['y_test'])+1))

    def build(self, **kwargs):
        forecaster = make_reduction(self.regressor, window_length=self.window_length, strategy="recursive")


        forecaster_pipe = NetworkPipelineForecaster([
            ('forecaster',forecaster, {'fit':{'y': 'original_y', 'fh':'original_fh'},
                                        'predict':{'fh':'original_fh'}
            }),
            ('converter', converter, {'fit':None,
                                    'predict':{'y1':kwargs['y_train'],'y2':'forecaster'}})
        ])
        self.model = forecaster_pipe
        return forecaster_pipe


########################################### REGRESSORS #######################################################
lasso_regression = RegressorPipe(model_name='lasso_regressor', regressor=linear_model.Lasso(), window_length=15)

random_forest_regressor_pipeline = RegressorPipe(model_name="random_forest_regressor",
                                                 regressor=RandomForestRegressor(),
                                                 window_length=15)
svm_regressor_pipeline = RegressorPipe(model_name='svm_regressor', regressor=svm.SVR(), window_length=15)

k_neighbours_regressor = RegressorPipe(model_name ='k_neighbours_regressor',regressor=KNeighborsRegressor(), window_length=15)

########################################### SUPERVISED CLASSIFICATION ##########################################



class SupervisedClassificationPipe(BasePipe):
    """
    Parameters
    ----------
    model_name : str
        name of model
    model : object
        sklearn classifier
    window_lenght : int
        For converting time series to tabular format.
    """
    def __init__(self, model_name, classifier, window_length):
        self.model_name = model_name
        self.classifier = classifier
        self.window_length = window_length
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
        classifier_pipe = NetworkPipelineForecaster([
            ('concatenator',self.concatenator, {'fit':None, 
                                            'predict':{'y_train': kwargs['y_train'], 'y_test': kwargs['y_test'], 'window_length':self.window_length}}),
            ('time_series_to_tabular_x', self.time_series_to_tabular,{
                    'fit':{'y':'original_y', 'window_length':self.window_length, 'fh': 'original_fh', 'return_value':'X'},
                    'predict': {'y':'concatenator', 'window_length':self.window_length, 'fh': 'original_fh', 'return_value':'X'} }),
            ('time_series_to_tabular_y', self.time_series_to_tabular,{
                    'fit':{'y':'original_y', 'window_length':self.window_length, 'fh': 'original_fh', 'return_value':'y'},
                    'predict': None }),
            ('classifier', self.classifier, {'fit':{'X': 'time_series_to_tabular_x', 'y': 'time_series_to_tabular_y' },
                                                    'predict': {'X': 'time_series_to_tabular_x'} })
        ])
        self.model = classifier_pipe
        return classifier_pipe
    def get_fh(self, **kwargs):
        return ForecastingHorizon([1])

logistic_regression = SupervisedClassificationPipe(model_name='LogisticRegression', classifier=linear_model.LogisticRegression(), window_length=15)
rf_classifier = SupervisedClassificationPipe(model_name='RandomForestClassifier', classifier=RandomForestClassifier(), window_length=15)
svm_classifier = SupervisedClassificationPipe(model_name='SVMClassifier', classifier=svm.SVC(), window_length=15)
k_neighbours_classifier = SupervisedClassificationPipe(model_name='K_NeighboursClassifier', classifier=KNeighborsClassifier(), window_length=15)

########################################### EXOGENOUS  ##########################################


class HHMExogenousPipeRegressor(BasePipe):
    """
    Parameters
    ----------
    model_name : str
        name of model
    regressor_model : object
        primary model
    exogenous_model : object
        secondary model
    """
    def __init__(self, model_name, regressor_model, exogenous_model, window_length):
        self.model_name = model_name
        self.exogenous_model = exogenous_model
        self.regressor_model = regressor_model
        self.window_length = window_length
    def reshape(self,y):
        return y.values.reshape(-1,1)
    

    def decoder(self,est,y):
        X = est.decode(y.values.reshape(-1,1))[1]
        X = pd.Series(X)
        X.index = y.index
        return X

    def build(self, **kwargs):
        exogenous_pipe = NetworkPipelineForecaster([
            ('reshape', self.reshape,  {'fit':{'y':'original_y'},
                                    'predict': {'y':kwargs['y_test']} }),
            ('exogenous_model',self.exogenous_model, {'X':'reshape'}),
            ('fitted_exogenous_model', 'get_fitted_estimator', {'fit': None, 'predict':{'step_name':'exogenous_model'}}), #discuss this interface for getting fitted models
            ('decoder',self.decoder, {'fit':{'est':'exogenous_model', 'y':'original_y'},
                                'predict':{'est':'fitted_exogenous_model', 'y':kwargs['y_test']}   }),
            ('regressor', make_reduction(self.regressor_model, window_length=5, strategy='recursive'), 
                                                                                {'fit':{'y': 'original_y', 'X': 'decoder', 'fh':'original_fh'},
                                                                                'predict':{'X': 'decoder'} } ),
            ('converter', converter, {'fit':None,
                                    'predict':{'y1':kwargs['y_train'],'y2':'regressor'}})
        ])
        self.model = exogenous_pipe
        return exogenous_pipe
    def get_fh(self, **kwargs):
        return ForecastingHorizon(np.arange(1,len(kwargs['y_test'])+1))

rf_hmm_exogenous = HHMExogenousPipeRegressor(model_name='RF_HMM_exogenous', regressor_model=RandomForestRegressor(),
                                          exogenous_model=hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000),
                                          window_length=15)


from mlfinlab.filters import cusum_filter


class CusumFilter:
    def __init__(self, filter=cusum_filter, search_range = np.arange(10,0.001, step=-0.01), pct_positive_annotations=0.1, threshold=None):
        self.filter = filter
        self.search_range = search_range
        self.pct_positive_annotations = pct_positive_annotations
        self.threshold=threshold

    def _find_threshold(self, X):
        thresholds = self.search_range
        sufficient_num_annotations = int(len(X) * self.pct_positive_annotations)
        for threshold in thresholds:
            num_annotations = len(self.filter(X, threshold))
            if num_annotations > sufficient_num_annotations:
                self.threshold = threshold
                return threshold
        
        self.threshold = thresholds[-1]
        return thresholds[-1]

    def _generate_predictions(self, X):
        filter_annotations = self.filter(X, threshold=self.threshold)
        predictions = pd.Series(data=np.zeros(len(X)), index=X.index)
        predictions.loc[filter_annotations] = 1
        return predictions
    def get_params(self, **kwargs):
        return {'threshold':self.threshold}

    def fit(self, X):
        self._find_threshold(X)
        return self
    def decode(self,X):
        return self._generate_predictions(X)
    def predict(self, X):
        return self._generate_predictions(X)



class CUSUMExogenousPipeRegressor(BasePipe):
    """
    Parameters
    ----------
    model_name : str
        name of model
    regressor_model : object
        primary model
    exogenous_model : object
        secondary model
    """
    def __init__(self, model_name, regressor_model, exogenous_model, window_length):
        self.model_name = model_name
        self.exogenous_model = exogenous_model
        self.regressor_model = regressor_model
        self.window_length = window_length
    def reshape(self,y):
        return y.values.reshape(-1,1)
    
    def decoder(self, est, y):
        return est._generate_predictions(y)

    def build(self, **kwargs):
        exogenous_pipe = NetworkPipelineForecaster([
            ('exogenous_model',self.exogenous_model, {'fit':{'X':'original_y'},
                                    'predict': None }),
            ('fitted_exogenous_model', 'get_fitted_estimator', {'fit': None, 'predict':{'step_name':'exogenous_model'}}),
            ('exogenous_model_decoder', self.decoder, {'fit': {'est': 'exogenous_model', 'y': 'original_y' }, 
                                                       'predict':{'est': 'fitted_exogenous_model', 'y': kwargs['y_test'] }}),

            ('regressor', make_reduction(self.regressor_model, window_length=5, strategy='recursive'), 
                                                                                {'fit':{'y': 'original_y', 'X': 'exogenous_model_decoder', 'fh':'original_fh'},
                                                                                'predict':{'X': 'exogenous_model_decoder'} } ),
            ('converter', converter, {'fit':None,
                                    'predict':{'y1':kwargs['y_train'],'y2':'regressor'}})
        ])
        self.model = exogenous_pipe
        return exogenous_pipe
    def get_fh(self, **kwargs):
        return ForecastingHorizon(np.arange(1,len(kwargs['y_test'])+1))


rf_cusum_exogenous = CUSUMExogenousPipeRegressor(model_name='RF_CUSUM_exogenous', regressor_model=RandomForestRegressor(),
                                          exogenous_model=CusumFilter(),
                                        window_length=15)
