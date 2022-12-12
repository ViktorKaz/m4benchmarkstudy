import pandas as pd
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateutil.parser
import logging
logging.basicConfig(filename='errors.log', level=logging.DEBUG)

class LoadM4Dataset:
    """Class for loading M4 datasets
    Parameters
    ----------
    main_dir_path : string 
        path to main directory with M4 Datasets
    dts_frequency : list of string
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
        self.info_path = f"{main_dir_path}/M4-info.csv"
        self.dts_train_path = []
        self.dts_test_path = []
        self.dts_freq = []
        for dts_freq in dts_frequency:
            self.dts_train_path.append(f"{main_dir_path}/Train/{dts_freq}-train.csv")
            self.dts_test_path.append(f"{main_dir_path}/Test/{dts_freq}-test.csv")
            
            self.dts_freq.append(dts_freq)

    
    
    # def get_num_lines_in_dts(self, freq):
    #     """
    #     Parameters
    #     ----------
    #     freq : str
    #         Frequency of dataset
    #     Returns
    #     -------
    #         int
    #             number of lines in the dataset excluding the first line which is the header row
    #     """
    #     idx = self.dts_freq.index(freq)
    #     reader = csv.reader(open(self.dts_train_path[idx]))
    #     num_lines = 0
    #     for line in reader:
    #         num_lines+=1
    #     return num_lines -2

    def load_dts_sequentially(self):
        info = pd.read_csv(self.info_path)
        for train_dts_path, test_dts_path  in zip(self.dts_train_path, self.dts_test_path):
            with open(train_dts_path, "r") as csv_train:
                with open(test_dts_path, "r") as csv_test:
                        
                        csv_reader_train = csv.reader(csv_train, delimiter=',')
                        csv_reader_test = csv.reader(csv_test, delimiter=',')
                        #skip first rows
                        next(csv_reader_train)
                        next(csv_reader_test)

                        for train_line, test_line in zip(csv_reader_train, csv_reader_test):
                            y_train = pd.Series(train_line)
                            y_test = pd.Series(test_line)                            
                            dts_id = y_train[0]
                            print(f'############ LOADING: {dts_id}')
                            y_train  = y_train[1:-1]
                            y_train = pd.to_numeric(y_train).dropna()
                            start_index = info[info['M4id'] == dts_id]['StartingDate'].values[0]
                            start_index = dateutil.parser.parse(start_index)
                            if start_index > datetime.now():
                                start_index -= relativedelta(years=100)
                            dts_freq = dts_id[0]
                            try:
                                index = pd.date_range(start_index,periods=y_train.shape[0],freq=dts_freq)
                            except:
                                print(f'###### Error creating index for: {dts_id}')
                                logger=logging.getLogger(__name__)

                                logger.error(f'###### Error creating index for: {dts_id}')
                                continue

                            y_train.index = index
                            y_train= y_train.astype('float')

                            y_test = y_test[1:-1]
                            y_test = pd.to_numeric(y_test).dropna()
                            start_index = y_train.index[-1]  + pd.DateOffset(1)
                            try:
                                index = pd.date_range(start_index,periods=y_test.shape[0],freq=dts_freq)
                            except:
                                print(f'###### Error creating index for: {dts_id}')
                                logger=logging.getLogger(__name__)

                                logger.error(f'###### Error creating index for: {dts_id}')
                                continue
                            y_test.index = index
                            y_test= y_test.astype('float')

                            yield y_train, y_test, dts_id