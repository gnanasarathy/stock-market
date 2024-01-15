from datetime import datetime, timedelta
import pandas as pd

def roll_value(date, df, weeks):
    next_date_obj = date + timedelta(weeks = weeks)
    next_date = next_date_obj.strftime("%Y-%m-%d")

    if next_date in df.index:
        return df.loc[next_date]['Close']
    else:
        return None
    
def find_index_position(df):
    index_position = 0
    for i in range(len(df['Target'])):
        is_nan = pd.isna(df["Target"][i])
        if not is_nan:
            return index_position
        index_position += 1

def construct_date(date, weeks = 1):
    days = weeks * 7
    next_date_obj = date + timedelta(days=days)
    next_date = next_date_obj.strftime("%d/%m")
    return next_date

def date_number(date):
    date = date.strftime("%d%m")
    return int(date)
