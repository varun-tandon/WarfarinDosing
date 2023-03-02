import pandas as pd
from constants import (
    AGE_COLUMN, 
    RACE_COLUMN, 
    HEIGHT_COLUMN, 
    WEIGHT_COLUMN, 
    ENZYME_INDUCER_COLUMNS, 
    DOSAGE_COLUMN,
)
pd.options.mode.chained_assignment = None  # default='warn'


def read_data():
    df = pd.read_csv('data/warfarin.csv')
    df = clean_data(df)
    return df

def clean_data(df):
    # first we drop the rows with nan values in Therapeutic Dose of Warfarin column
    df = df.dropna(subset=[DOSAGE_COLUMN])

    # Impute missing values, filling in 0 where relevant
    df[AGE_COLUMN] = impute_age_data(df)
    df[HEIGHT_COLUMN] = impute_height_data(df)
    df[WEIGHT_COLUMN] = impute_weight_data(df)
    df[RACE_COLUMN] = impute_race_data(df)
    df[ENZYME_INDUCER_COLUMNS] = df[ENZYME_INDUCER_COLUMNS].apply(impute_enzyme_column)
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
    # hehe u silly boi
    return data[WEIGHT_COLUMN].fillna(mean)

def impute_race_data(data):
    return data[RACE_COLUMN].fillna('Unknown')

def impute_enzyme_column(column):
    return column.fillna(0)
