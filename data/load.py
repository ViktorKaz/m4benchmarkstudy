import pandas as pd

def return_correlation_ts(idx_1, idx_2, corr_window=30, ma_window=30):
    df = pd.read_csv('data/res_10.csv')
    combined = pd.DataFrame()
    combined['DATE'] = df[df['ID'] == idx_1]['DATE']
    combined[idx_1] = df[df['ID'] == idx_1]['PX_LAST'].values
    combined[idx_2] = df[df['ID'] == idx_2]['PX_LAST'].values
    combined = combined[combined[idx_1].notna()]
    combined = combined[combined[idx_2].notna()]
    combined[f'corr_{idx_1}_{idx_2}'] = combined[idx_1].rolling(corr_window).corr(combined[idx_2])
    combined[f'MA_{ma_window}'] = combined[f'corr_{idx_1}_{idx_2}'].rolling(ma_window).mean()
    combined.index = pd.to_datetime(combined['DATE'].values)
    return combined

def split_ts(ts, training_fold=2/3):
    train = int(ts.dropna().shape[0]*training_fold)
    y_train = ts.iloc[0:train].dropna()
    y_test = ts.iloc[train+1:-1].dropna()
    y_train = y_train.asfreq('D', method='bfill')
    y_test = y_test.asfreq('D', method='bfill')

    return y_train, y_test

def get_y(idx_1, idx_2, corr_window=30, ma_window=30, training_fold=2/3):
    corr = return_correlation_ts(idx_1, idx_2, corr_window, ma_window)
    y_train, y_test = split_ts(corr[f'MA_{ma_window}'], training_fold)
    return y_train, y_test