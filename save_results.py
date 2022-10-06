import pandas as pd
class SaveResults:
    """
    Saves results to disk

    Parameters
    ----------
    series_name : string
    strategy_name : string
    y_true : pd.Series of int
    y_pred : pd.Series of int
    """
    SAVE_PATH = '/home/viktor/MPhil/m4benchmarkstudy/results/'

    def __init__(self, series_name, strategy_name, y_true, y_pred):
        self._series_name= series_name
        self._strategy_name = strategy_name
        self._y_true = y_true
        self._y_pred = y_pred

    def save(self):
        """Concatenate and save as dataframe"""
        df = pd.DataFrame.from_dict({'true': self._y_true.values, 'predict': self._y_pred.values})
        df.index = self._y_true.index
        file_name = f"{self.SAVE_PATH}{self._series_name}_{self._strategy_name}"
        df.to_csv(file_name)