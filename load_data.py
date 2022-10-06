import pandas as pd

class LoadM4Data:
    """Class for loading M4 datasets
    Parameters
    ----------
    train : string 
        path to training dataset must be in csv format
    test : string
        path to testing dataset. Must be in csv format
    info : string
        path to info csv file
    time_series_to_load : int
        number

    Returns
    -------
    train : pandas Dataframe
        trainig set in pandas
    test : pandas DataFrame
        test set in pandas
    info:
    """
    def __init__(self, train, test, info):
        
        self._train = train
        self._test = test
        self._info = info

        with open(train) as f:

            self._total_nuber_of_rows = sum(1 for line in f) -1 #not count header row
        
    def load(self, series_num):
        """
        Loads a specific time series

        Parameters
        ----------
        series_num : int
            number of the time series in the database

        Returns
        -------
        """
        series_num +=1 # ignore header row

        train_df = pd.read_csv(self._train, skiprows=series_num, nrows=1, header=None)
        y_train = train_df.iloc[:,1:-1].dropna()
        y_train = y_train.squeeze()

        test_df  = pd.read_csv(self._test, skiprows=series_num, nrows=1,header=None)
        y_test = test_df.iloc[:,1:-1].dropna()
        y_test = y_test.squeeze()
        
        info = pd.read_csv(self._info)
        series_name = train_df[0].values[0]
        
        start_index_train = info[info['M4id']==series_name]['StartingDate'].iloc[0]
        index_train = pd.date_range(start_index_train, periods=len(y_train), freq='D')
        
        y_train.index = index_train
        y_train = y_train.astype(float)

        start_index_test = y_train.index[-1]  + pd.DateOffset(1)
        index_test = pd.date_range(start_index_test,periods=len(y_test),freq='D')
        y_test.index = index_test
        return y_train, y_test, series_name

    def get_total_number_of_timeseries(self):
        return self._total_nuber_of_rows


load_ts = LoadM4Data(train='Dataset/Train/Daily-train.csv', 
                    test='Dataset/Test/Daily-test.csv',
                    info='Dataset/M4-info.csv')
load_ts.load(4)