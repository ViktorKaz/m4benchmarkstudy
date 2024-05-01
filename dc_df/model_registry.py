from dc_df.base import ExogenousDC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn import linear_model

class HMMExogenousDC(ExogenousDC):

    def __init__(self, estimator, y_train, y_test, name, n_components=3):
        ExogenousDC.__init__(self,estimator, y_train, y_test, name)
        from hmmlearn import hmm
        self.name = name
        self.exogenous_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)

