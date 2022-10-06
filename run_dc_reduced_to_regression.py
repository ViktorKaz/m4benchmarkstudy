from load_data import LoadM4Data

load_ts = LoadM4Data(train='Dataset/Train/Daily-train.csv', 
                    test='Dataset/Test/Daily-test.csv',
                    info='Dataset/M4-info.csv')
from dc_reduced_to_regression import RandomForestGSCV
from sktime.forecasting.base import ForecastingHorizon
from save_results import SaveResults
from dc_reduced_to_supervised_classification import dc_rf

for i in range(0,2):
    print(i)
    y_train, y_test, series_name = load_ts.load(i)
    initial_window=int(len(y_train)*0.8)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    rf = RandomForestGSCV(initial_window, name='RandomForestForecaster')
    rf.fit(y_train, fh=fh)
    y_pred =rf.predict(fh)
    print(y_pred)
    strategy_name = rf.get_name()
    to_save = SaveResults(series_name=series_name, strategy_name=strategy_name, y_true=y_test, y_pred=y_pred)
    to_save.save()