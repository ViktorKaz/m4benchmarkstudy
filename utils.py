import pandas as pd

def converter(y1,y2):
    """
    Converts regression output to directional change output where 1 means up 0 means down
    Parameters
    ----------
    y1 : pd.Series
        Series preceeding y2. It is used only to calculate the value of the first item in y2 (up down relative to the last value of y1)
    y2 : pd.Series
    
    Returns
    -------
        pd.Series
    """
    concat_y1_y2 = pd.concat([y1[-2:-1],y2])
    dc = concat_y1_y2.shift(-1) > concat_y1_y2
    dc = dc[0:-1]
    dc[dc==True] =1
    dc[dc==False] =0
    dc = dc.astype('int')
    return dc
