from unicodedata import name
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sklearn.ensemble import RandomForestClassifier
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sktime.classification.base import BaseClassifier, BaseEstimator
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon





class DirectionalChangeClassifier(BaseEstimator):
    """Custom forecaster. todo: write docstring.
    todo: describe your custom forecaster here
    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    # tag values are "safe defaults" which can usually be left as-is
    _tags = {
            "X_inner_mtype": "pd.Series",
            "X_inner_mtype": "pd.DataFrame",  # which type do _fit/_predict accept, usually
            # this is either "numpy3D" or "nested_univ" (nested pd.DataFrame). Other
            # types are allowable, see datatypes/panel/_registry.py for options.
            "capability:multivariate": False,
            "capability:unequal_length": False,
            "capability:missing_values": False,
            "capability:train_estimate": False,
            "capability:contractable": False,
            "capability:multithreading": False,
            "python_version": None,  # PEP 440 python version specifier to limit versions
        }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, classifier, name, window_length=100):
        self.name=name
        self.classifier = cls
        self._window_length= window_length
        # todo: change "MyForecaster" to the name of the class
        super(DirectionalChangeClassifier, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory
    def _fit(self, y, X=None):
        """Fit forecaster to training data.
        private _fit containing the core logic, called from fit
        Writes to self:
            Sets fitted model attributes ending in "_".
        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.
        Returns
        -------
        self : reference to self
        """
        fh=ForecastingHorizon([1])
        y_tmp,x_dc = _sliding_window_transform(y,window_length=self._window_length,fh=fh)
        y_dc = np.zeros(len(y_tmp))
        y_tmp = y_tmp.reshape(x_dc[:,-1].shape)
        y_mask = (x_dc[:,-1] > y_tmp) #up observations
        y_dc[y_mask]=1
        y_dc = np.zeros(len(y_tmp))

        self.y = y
        self.classifier.fit(X=x_dc,y=y_dc)
        return self
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (y, X) or forecasting-horizon-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

    # todo: implement this, mandatory
    def _predict(self, X):
        """Forecast time series at future horizon.
        private _predict containing the core logic, called from predict
        State required:
            Requires state to be "fitted".
        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff
        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """

        # implement here
        fh=ForecastingHorizon([1])

        y_test_concat = pd.concat([self.y[-self._ini-1:-1],X])
        y_tmp_test,x_dc_test = _sliding_window_transform(y_test_concat,window_length=self._window_length,fh=fh)
        y_dc_test = np.zeros(len(y_tmp_test))
        y_tmp_test = y_tmp_test.reshape(x_dc_test[:,-1].shape)
        y_mask = (x_dc_test[:,-1] > y_tmp_test) #up observations
        y_dc_test[y_mask]=1
        y_pred = self.forecaster.predict(X)

        
        # concatenated = pd.concat([self.y[-1:0],y_pred])
        # predict_dc = concatenated.shift(-1) > concatenated
        # predict_dc = predict_dc[0:-1]
        # predict_dc[predict_dc == True] =1
        # predict_dc[predict_dc == False] =0
        # return predict_dc.astype(int)
        return y_pred

    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.
        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.
        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params
# class RandomForestGSCV(BaseForecaster):
#     def __init__(self, initial_window):
#         print('before')

#         param_grid = {"window_length": [7, 12, 15]}

#         regressor = RandomForestRegressor(n_jobs=12)
#         forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
#         cv = SlidingWindowSplitter(initial_window=initial_window, window_length=20)
#         super().__init__(forecaster, strategy="refit", cv=cv, param_grid=param_grid)

#         print('after')
#     def _fit(self,y, X=None, fh=None):
#         print('fit')
#         super().fit(y)



cls = RandomForestClassifier(n_jobs=12)
dc_rf = DirectionalChangeClassifier(classifier=RandomForestClassifier(n_jobs=12), name='RandomForestClassifier', window_length=100)
