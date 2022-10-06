from load_data import LoadM4Data

load_ts = LoadM4Data(train='Dataset/Train/Daily-train.csv', 
                    test='Dataset/Test/Daily-test.csv',
                    info='Dataset/M4-info.csv')
from dc_reduced_to_regression import RandomForestGSCV
from sktime.forecasting.base import ForecastingHorizon
from save_results import SaveResults

#estimators
from dc_reduced_to_supervised_classification import dc_rf

for i in range(0,2):
    print(i)
    y_train, y_test, series_name = load_ts.load(i)

    dc_rf.fit(y=y_train, X=y_train.to_frame())
    y_pred =dc_rf.predict(y=y_test)
    print(y_pred)
    strategy_name = dc_rf.get_name()
    to_save = SaveResults(series_name=series_name, strategy_name=strategy_name, y_true=y_test, y_pred=y_pred)
    to_save.save()