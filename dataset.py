import pandas as pd
from constants import AGE_COLUMN, RACE_COLUMN, HEIGHT_COLUMN, WEIGHT_COLUMN

def read_data():
    df = pd.read_csv('data/warfarin.csv')
    return df

def impute_age_data(data):
    data_dropped = data.dropna(subset=[AGE_COLUMN])
    dropped_age_col = data_dropped[AGE_COLUMN].apply(lambda x: int(x[0]))
    mean = round(dropped_age_col.mean())
    return data[AGE_COLUMN].fillna(str(mean))

def impute_height_data(data):
    data_dropped = data.dropna(subset=[HEIGHT_COLUMN])
    mean = round(data_dropped[HEIGHT_COLUMN].mean())
    return data[HEIGHT_COLUMN].fillna(mean)

def impute_weight_data(data):
    data_dropped = data.dropna(subset=[WEIGHT_COLUMN])
    mean = round(data_dropped[WEIGHT_COLUMN].mean())
    return data[HEIGHT_COLUMN].fillna(mean)

def impute_race_data(data):
    return data[RACE_COLUMN].fillna('Unknown')

def impute_enzyme_column(column):
    return column.fillna(0)
