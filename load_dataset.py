import pandas as pd
import csv
class LoadM4Dataset:
    """Class for loading M4 datasets
    Parameters
    ----------
    main_dir_path : string 
        path to main directory with M4 Datasets
    dts_frequency : string
        Must follow M4 convention. Acceptable values: `Daily`, `Hourly`, `Monthly`, `Quarterly`, `Weekly`, `Yearly`


    Returns
    -------
    train : pandas Dataframe
        trainig set in pandas
    test : pandas DataFrame
        test set in pandas
    info:
    """
    def __init__(self, main_dir_path, dts_frequency):
        self.dts_train_path = f"{main_dir_path}/Train/{dts_frequency}-train.csv"
        self.dts_test_path = f"{main_dir_path}/Test/{dts_frequency}-test.csv"
        self.info_path = f"{main_dir_path}/M4-info.csv"

    def load_dts_number(self,dts_number):
        train = pd.read_csv(self.dts_train_path, skiprows=dts_number, nrows=1)
        test = pd.read_csv(self.dts_test_path, skiprows=dts_number, nrows=1)
        info = pd.read_csv(self.info_path, skiprows=dts_number, nrows=1)
        y_train = train.dropna(axis=1)
        y_train  = pd.DataFrame(data=y_train.values[0][1:-1])

        series_name = train.iloc[0,0]
        start_index = info.iloc[-1,-1]
        index = pd.date_range(start_index,periods=y_train.shape[0],freq='D')
        y_train.index = index
        y_train= y_train.astype('float')

        y_test = test.dropna(axis=1)
        y_test = pd.DataFrame(data=y_test.values[0][1:-1])
        start_index = y_train.index[-1]  + pd.DateOffset(1)
        index = pd.date_range(start_index,periods=y_test.shape[0],freq='D')
        y_test.index = index
        y_test= y_test.astype('float')

        return y_train, y_test
    
    def get_num_lines_in_dts(self):
        """
        Returns
        -------
            int
                number of lines in the dataset excluding the first line which is the header row
        """
        reader = csv.reader(open(self.dts_train_path))
        num_lines = 0
        for line in reader:
            num_lines+=1
        return num_lines -2

    def load_dts_sequentially(self):
        with open(self.dts_train_path, "r") as csv_train:
            with open(self.dts_test_path, "r") as csv_test:
                with open(self.info_path, "r") as csv_info:

                    csv_reader_train = csv.reader(csv_train, delimiter=',')
                    csv_reader_test = csv.reader(csv_test, delimiter=',')
                    csv_reader_info = csv.reader(csv_info, delimiter=',')
                    #skip first rows
                    next(csv_reader_train)
                    next(csv_reader_test)
                    next(csv_reader_info)
                    for train_line, test_line, info_line in zip(csv_reader_train, csv_reader_test, csv_reader_info):
                        y_train = pd.Series(train_line)
                        y_test = pd.Series(test_line)

                        y_train = y_train.dropna()
                        y_train  = y_train[1:-1]
                        start_index = info_line[-1]          
                        index = pd.date_range(start_index,periods=y_train.shape[0],freq='D')
                        y_train.index = index
                        y_train= y_train.astype('float')

                        y_test = y_test.dropna()
                        y_test = y_test[1:-1]
                        start_index = y_train.index[-1]  + pd.DateOffset(1)
                        index = pd.date_range(start_index,periods=y_test.shape[0],freq='D')
                        y_test.index = index
                        y_test= y_test.astype('float')

                        yield y_train, y_test