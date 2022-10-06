from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
)
from sktime.forecasting.base import BaseForecaster
import pandas as pd




def convert_regression_forecasts(y_train,y_test,y_pred,y_true):
    pass





def convert_to_dc(y1,y2):
    """Converts y2 from pandas series of float to pandas series of int where 1 indicates 
    increase compared to the previous value in the series and 0 indicates a decrease
    Parameters
    ----------

    y1 : pandas series of float  
    y2 : pandas series of float

    Returns
    -------
        pandas series of int
    """
    concatenated = pd.concat([y1[-2:-1],y2])
    true_dc = concatenated.shift(-1) > concatenated
    true_dc = true_dc[0:-1]
    true_dc[true_dc==True] =1
    true_dc[true_dc==False] =0

    return true_dc.astype(int)



#random forest

class RandomForestGSCV(BaseForecaster):
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
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "univariate",
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # ignores-exogeneous-X = does estimator ignore the exogeneous X?
        "ignores-exogeneous-X": False,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": True,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, initial_window, name):
        name = name
        self.initial_window = initial_window
        param_grid = {"window_length": [7, 12, 15]}
        regressor = RandomForestRegressor(n_jobs=12)
        forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
        cv = SlidingWindowSplitter(initial_window=initial_window, window_length=20)
        self.forecaster = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
        # todo: change "MyForecaster" to the name of the class
        super(RandomForestGSCV, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory
    def _fit(self, y, X=None, fh=None):
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
        self.y = y
        self.forecaster.fit(y,fh=fh)
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
    def _predict(self, fh, X=None):
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
        # IMPORTANT: avoid side effects to X, fh
        y_pred = self.forecaster.predict(fh)

        
        # concatenated = pd.concat([self.y[-1:0],y_pred])
        # predict_dc = concatenated.shift(-1) > concatenated
        # predict_dc = predict_dc[0:-1]
        # predict_dc[predict_dc == True] =1
        # predict_dc[predict_dc == False] =0
        # return predict_dc.astype(int)
        return convert_to_dc(self.y, y_pred)

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

    def get_name(self):
        return self.name