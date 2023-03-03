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
    VKORC1_COLUMN_2255,
    VKORC1_COLUMN_1173,
    VKORC1_COLUMN_1542,
    CYP2C9_COLUMN,
    DOSAGE_BUCKET_COLUMN,
    Features,
    Actions
)
from utils import convert_dosage_to_action
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
    df = impute_vkorc1_data(df)
    df[CYP2C9_COLUMN] = df[CYP2C9_COLUMN].fillna('Unknown')
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

# note this logic must be applied in this ordering
def impute_vkorc1_data(data):
    for i, row in data.iterrows():
        if (row[RACE_COLUMN] != "Black or African American") or (row[RACE_COLUMN] != "Missing or Mixed Race"):
            if row[VKORC1_COLUMN_2255] == "C/C":
                data.at[i, VKORC1_COLUMN] = "G/G"
            elif row[VKORC1_COLUMN_2255] == "T/T":
                data.at[i, VKORC1_COLUMN] = "A/A"
            elif row[VKORC1_COLUMN_2255] == "C/T":
                data.at[i, VKORC1_COLUMN] = "A/G"
            elif pd.isna(row[VKORC1_COLUMN]):
                data.at[i, VKORC1_COLUMN] = "Missing"
        elif row[VKORC1_COLUMN_1173] == "C/C":
            data.at[i, VKORC1_COLUMN] = "G/G"
        elif row[VKORC1_COLUMN_1173] == "T/T":
            data.at[i, VKORC1_COLUMN] = "A/A"
        elif row[VKORC1_COLUMN_1173] == "C/T":
            data.at[i, VKORC1_COLUMN] = "A/G"
        elif (row[RACE_COLUMN] != "Black or African American") or (row[RACE_COLUMN] != "Missing or Mixed Race"):
            if row[VKORC1_COLUMN_1542] == "G/G":
                data.at[i, VKORC1_COLUMN] = "G/G"
            elif row[VKORC1_COLUMN_1542] == "C/C":
                data.at[i, VKORC1_COLUMN] = "A/A"
            elif row[VKORC1_COLUMN_1542] == "C/G":
                data.at[i, VKORC1_COLUMN] = "A/G"
            elif pd.isna(row[VKORC1_COLUMN]):
                data.at[i, VKORC1_COLUMN] = "Missing"
        elif pd.isna(row[VKORC1_COLUMN]):
                data.at[i, VKORC1_COLUMN] = "Missing"
    return data

def augment_data(df):
    # Start by transforming the data with the features needed for the algo per S1f
    df[Features.AGE_IN_DECADES.value] = df[AGE_COLUMN].apply(lambda x: int(x[0]))
    df[Features.HEIGHT.value] = df[HEIGHT_COLUMN]
    df[Features.WEIGHT.value] = df[WEIGHT_COLUMN]

    df[Features.ENZYME_INDUCER_STATUS.value] = (df[ENZYME_INDUCER_COLUMNS].sum(axis=1) > 0).astype(int)
    df[Features.AMIODARONE_STATUS.value] = df[AMIODARONE_COLUMN].fillna(0)
    df[Features.ASIAN_RACE.value] = df[RACE_COLUMN].apply(lambda x: 1 if x == 'Asian' else 0)
    df[Features.BLACK_OR_AFRICAN_AMERICAN.value] = df[RACE_COLUMN].apply(lambda x: 1 if x == 'Black or African American' else 0)
    df[Features.MISSING_OR_MIXED_RACE.value] = df[RACE_COLUMN].apply(lambda x: 1 if x == 'Unknown' else 0)

    df[Features.VKORC1_A_G.value] = df[VKORC1_COLUMN].apply(lambda x: 1 if x == 'A/G' else 0)
    df[Features.VKORC1_A_A.value] = df[VKORC1_COLUMN].apply(lambda x: 1 if x == 'A/A' else 0)
    df[Features.VKORC1_UNKOWN.value] = df[VKORC1_COLUMN].apply(lambda x: 1 if x == 'Missing' else 0)

    df[Features.CYP2C9_1_2.value] = df[CYP2C9_COLUMN].apply(lambda x: 1 if x == '*1/*2' else 0)
    df[Features.CYP2C9_1_3.value] = df[CYP2C9_COLUMN].apply(lambda x: 1 if x == '*1/*3' else 0)
    df[Features.CYP2C9_2_2.value] = df[CYP2C9_COLUMN].apply(lambda x: 1 if x == '*2/*2' else 0)
    df[Features.CYP2C9_2_3.value] = df[CYP2C9_COLUMN].apply(lambda x: 1 if x == '*2/*3' else 0)
    df[Features.CYP2C9_3_3.value] = df[CYP2C9_COLUMN].apply(lambda x: 1 if x == '*3/*3' else 0)
    df[Features.CYP2C9_UKNOWN.value] = df[CYP2C9_COLUMN].apply(lambda x: 1 if x == 'Unknown' else 0)

    # finally create the dosage column as a categorical
    df[DOSAGE_BUCKET_COLUMN] = df[DOSAGE_COLUMN].apply(convert_dosage_to_action)
    return df


if __name__ == "__main__":
    df = pd.read_csv('data/warfarin.csv')
    # impute missing data, remove rows with missing dosage
    df = clean_data(df)

    # augment data with features needed for the algorithms
    df = augment_data(df)
    # save only the needed columns
    selected_columns = [f.value for f in Features]
    selected_columns.append('dosage')
    df = df[selected_columns]
    df.to_csv('data/warfarin_clean.csv', index=False)