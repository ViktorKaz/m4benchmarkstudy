import pandas as pd
import numpy as np


from sktime.forecasting.compose import make_reduction
from evaluation_registry import DCEvaluator
from sktime.forecasting.model_selection import SlidingWindowSplitter


class DirectionalForecast():
    def __init__(self, estimator, y_train, y_test, name):
        self.estimator = estimator
        self.y_train = y_train
        self.y_test = y_test
        self.X = None #for exogenous models
        self.name = name
    def get_name(self):
        return self.name
    def fit_predict(self):
        """Returns predictions in pd.Series of float        
        """
        initial_window=int(len(self.y_train)*0.8)
        # fh = ForecastingHorizon(y_test.index, is_relative=False)
        
        estimator = make_reduction(self.estimator, window_length=15, strategy="recursive")
        # cv = SlidingWindowSplitter(initial_window=initial_window, window_length=20)
        # param_grid = {"window_length": [7, 12, 15]}
        # forecaster = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)

        for i in range(len(self.y_test)):
        
            date_index = pd.to_datetime([])
            y_pred = pd.Series(index=date_index)
            for i in range(len(self.y_test)):
                comb_y = pd.concat([self.y_train, self.y_test[0:i]])
                comb_y = comb_y.asfreq('D', method='bfill')
                estimator.fit(y=comb_y, X=None, fh=[1])
                pred = estimator.predict()
                y_pred[pred.index[0]]=pred.values[0]
            
            self.y_pred = y_pred
            return y_pred
    

    def evaluate(self):

        evaluator = DCEvaluator()
        # true_dc = self.convert_to_dc(y1=self.y_train, y2=self.y_test)
        # pred_dc = self.convert_to_dc(y1=self.y_train, y2=self.y_pred)
        if self.X is None:
            accuracy, f1,fpr, tpr, area_under_the_curve = evaluator.evaluate(self.y_train, self.y_test, self.y_pred, tag=self.tag)

            self._accuracy = accuracy
            self._f1 = f1
            self._fpr = fpr[1]
            self._tpr = tpr[1]
            self._auc = area_under_the_curve

            return accuracy, f1,fpr, tpr, area_under_the_curve
        else:
            X_idx_test = self.X.loc[self.y_test.iloc[0:-1].index].index
            X_idx_train = self.X.loc[self.y_train.index].index
            all_accuracy = []
            all_f1 = []
            all_fpr = []
            all_tpr = []
            all_area_under_the_curve = []
            for cls_to_skip in self.X.unique():
                #drop last prediction as we don't have exogenous prediction for it
                y_test = self.y_test.iloc[0:-1]
                y_pred = self.y_pred.iloc[0:-1]
                accuracy, f1,fpr, tpr, area_under_the_curve = evaluator.evaluate(
                                                                self.y_train.loc[X_idx_train != cls_to_skip], 
                                                                y_test.loc[X_idx_test != cls_to_skip], 
                                                                y_pred.loc[X_idx_test != cls_to_skip],
                                                                tag=self.tag)
                all_accuracy.append(accuracy)
                all_f1.append(f1)
                all_fpr.append(fpr)
                all_tpr.append(tpr)
                all_area_under_the_curve.append(area_under_the_curve)

            max_accuracy = max(all_accuracy)
            idx_max_accuracy = all_accuracy.index(max_accuracy)
            
            self._accuracy = all_accuracy[idx_max_accuracy]
            self._f1 = all_f1[idx_max_accuracy]
            self._fpr = all_fpr[idx_max_accuracy][1]
            self._tpr = all_tpr[idx_max_accuracy][1]
            self._auc = all_area_under_the_curve[idx_max_accuracy]

            return all_accuracy[idx_max_accuracy], \
                    all_f1[idx_max_accuracy], \
                    all_fpr[idx_max_accuracy], \
                    all_tpr[idx_max_accuracy], \
                    all_area_under_the_curve[idx_max_accuracy]
        
    def save_results(self):
        try:
            existing_data = pd.from_csv('results.csv')
        except:
            existing_data = pd.DataFrame(columns=['estimator','accuracy','f1','fpr','tpr','auc'])
        data = [self._accuracy, self._f1, self._tpr, self._fpr, self._auc]
        new_data = pd.DataFrame(columns=['estimator','accuracy','f1','fpr','tpr','auc'], data=data)
        concatednated_results = pd.concat([existing_data, new_data])
        concatednated_results.to_csv('results.csv')


class RegressorDF(DirectionalForecast):
    def __init__(self, *args, **kwargs):
        DirectionalForecast.__init__(self,*args, **kwargs)
        self.tag = 'regressor'

    def convert_to_dc(self,y1,y2):
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
    


class ExogenousDC(RegressorDF):
    def exogenous_predict(self):
        comb_y = pd.concat([self.y_train, self.y_test])
        predictions = []
        splitter = SlidingWindowSplitter(fh=[1], window_length=self.y_train.shape[0])
        for split in splitter.split(comb_y):
            y = comb_y.iloc[split[0]].values.reshape(-1,1)
            self.exogenous_model.fit(y)
            predictions.append(self.exogenous_model.decode(y)[1][-1])
        
        return pd.Series(index=self.y_test.index,data=predictions)
    
    def fit_predict(self):
        """Returns predictions in pd.Series of float        
        """
        initial_window=int(len(self.y_train)*0.8)
        # fh = ForecastingHorizon(y_test.index, is_relative=False)
        
        forecaster = make_reduction(self.estimator, window_length=15, strategy="recursive")
        # cv = SlidingWindowSplitter(initial_window=initial_window, window_length=20)
        # param_grid = {"window_length": [7, 12, 15]}
        # forecaster = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)

        for i in range(len(self.y_test)):
        
            date_index = pd.to_datetime([])
            y_pred = pd.Series(index=date_index)
            for i in range(len(self.y_test)):
                comb_y = pd.concat([self.y_train, self.y_test[0:i]])
                comb_y = comb_y.asfreq('D', method='bfill')
                X = self.exogenous_fit(y=comb_y)
                X_bfilled = X.asfreq('D',method='bfill')
                self.X = X_bfilled
                forecaster.fit(y=comb_y, X=X_bfilled, fh=[1])
                pred = forecaster.predict(X=X_bfilled)
                y_pred[pred.index[0]]=pred.values[0]
            
            self.y_pred = y_pred
            y_pred_dc = self.convert_to_dc(y1=self.y_train, y2=y_pred)
            return y_pred_dc, X_bfilled

    def exogenous_fit(self,y):
        """
        Simulates sequential arrival of the y_test data.
        Exogenous model makes 1 steap ahead predictions.
        Used to avoid leakage of the test data


        Retrurns
        --------
            pd Series
        """
            
        y_reshaped = y.values.reshape(-1,1)
        self.exogenous_model.fit(y_reshaped)
        decoded_y = self.exogenous_model.decode(y_reshaped)[1]

        return pd.Series(index=y.index, data=decoded_y)