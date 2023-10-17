from dc_df.base import ExogenousDC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn import linear_model

class HMMExogenousDC(ExogenousDC):

    def __init__(self, estimator, y_train, y_test):
        ExogenousDC.__init__(self,estimator, y_train, y_test)
        from hmmlearn import hmm

        self.exogenous_model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)

# RF_HMM
class HMM_RF():
    def __init__(self):
        self.name = "RF_HMM"
        
    def build(self, y_train, y_test):
        regressor_rf = RandomForestRegressor(n_jobs=12)
        hmm_exogenous = HMMExogenousDC(estimator=regressor_rf, y_train=y_train, y_test=y_test)

        return hmm_exogenous