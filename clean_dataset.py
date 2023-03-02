import pandas as pd
from constants import (
    AGE_COLUMN, 
    RACE_COLUMN, 
    HEIGHT_COLUMN, 
    WEIGHT_COLUMN, 
    ENZYME_INDUCER_COLUMNS, 
    DOSAGE_COLUMN,
    AMIODARONE_COLUMN,
    VKORC1_COLUMN,
    Features,
    Actions
)
pd.options.mode.chained_assignment = None  # default='warn'


def clean_data(df):
    # first we drop the rows with nan values in Therapeutic Dose of Warfarin column
    df = df.dropna(subset=[DOSAGE_COLUMN])

    # First impute missing values, filling in 0 where relevant
    df[AGE_COLUMN] = impute_age_data(df)
    df[HEIGHT_COLUMN] = impute_height_data(df)
    df[WEIGHT_COLUMN] = impute_weight_data(df)
    df[RACE_COLUMN] = impute_race_data(df)
    df[ENZYME_INDUCER_COLUMNS] = df[ENZYME_INDUCER_COLUMNS].apply(impute_enzyme_column)
    df[VKORC1_COLUMN] = impute_vkorc1_data(df)
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
    return data[WEIGHT_COLUMN].fillna(mean)

def impute_race_data(data):
    return data[RACE_COLUMN].fillna('Unknown')

def impute_enzyme_column(column):
    return column.fillna(0)

def impute_vkorc1_data(data):

    return data[VKORC1_COLUMN].fillna('G')

def augment_data(df):
    # Start by transforming the data with the features needed for the algo per S1f
    df[Features.AGE_IN_DECADES] = df[AGE_COLUMN].apply(lambda x: int(x[0]))
    df[Features.HEIGHT] = df[HEIGHT_COLUMN]
    df[Features.WEIGHT] = df[WEIGHT_COLUMN]

    df[Features.ENZYME_INDUCER_STATUS] = (df[ENZYME_INDUCER_COLUMNS].sum(axis=1) > 0).astype(int)
    df[Features.AMIODARONE_STATUS] = df[AMIODARONE_COLUMN].fillna(0)
    df[Features.ASIAN_RACE] = df[RACE_COLUMN].apply(lambda x: 1 if x == 'Asian' else 0)
    df[Features.BLACK_OR_AFRICAN_AMERICAN] = df[RACE_COLUMN].apply(lambda x: 1 if x == 'Black or African American' else 0)
    df[Features.MISSING_OR_MIXED_RACE] = df[RACE_COLUMN].apply(lambda x: 1 if x == 'Unknown' else 0)

    df[Features.VKORC1_G_A] = df['VKORC1 -1639 G>A'].apply(lambda x: 1 if x == 'G' else 0)

    # finally create the dosage column as a categorical
    df["dosage"] = df[DOSAGE_COLUMN].apply(lambda x: Actions.LOW if x < 21 else Actions.MEDIUM if x < 49 else Actions.HIGH)
    return df


if __name__ == "__main__":
    df = pd.read_csv('data/warfarin.csv')
    # impute missing data, remove rows with missing dosage
    df = clean_data(df)

    # augment data with features needed for the algorithms
    df = augment_data(df)
    # save only the needed columns
    df = df[[f.value for f in Features]]
    df.to_csv('data/warfarin_clean.csv', index=False)