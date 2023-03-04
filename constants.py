from enum import Enum

# Potential actions an agent can take in applying doses
class Actions(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

# These are pulled from S1f of the paper
class Features(Enum):
    AGE_IN_DECADES = "age_in_decades"
    HEIGHT = "height"
    WEIGHT = "weight"
    
    ASIAN_RACE = "asian_race"
    BLACK_OR_AFRICAN_AMERICAN = "black_or_african_american"
    MISSING_OR_MIXED_RACE = "missing_or_mixed_race"
    ENZYME_INDUCER_STATUS = "enzyme_inducer_status"
    AMIODARONE_STATUS = "amiodarone_status"

    VKORC1_A_G = "vk0rc1_a_g"
    VKORC1_A_A = "vk0rc1_a_a"
    VKORC1_UNKOWN = "vk0rc1_unknown"
    CYP2C9_1_2 = "cyp2c9_1_2"
    CYP2C9_1_3 = "cyp2c9_1_3"
    CYP2C9_2_2 = "cyp2c9_2_2"
    CYP2C9_2_3 = "cyp2c9_2_3"
    CYP2C9_3_3 = "cyp2c9_3_3"
    CYP2C9_UKNOWN = "cyp2c9_unknown"

DOSAGE_BUCKET_COLUMN = "dosage"

# These are the features used by the computation of the clinical dosing agent
CLINICAL_DOSING_COLUMNS = [
    Features.AGE_IN_DECADES.value,
    Features.HEIGHT.value,
    Features.WEIGHT.value,
    Features.ASIAN_RACE.value,
    Features.BLACK_OR_AFRICAN_AMERICAN.value,
    Features.MISSING_OR_MIXED_RACE.value,
    Features.ENZYME_INDUCER_STATUS.value,
    Features.AMIODARONE_STATUS.value,
]

# These are the features used by the Linear Bandit agent
LINEAR_BANDIT_COLUMNS = [f.value for f in Features]

# Columns needed in order to do imputing or transformations
DOSAGE_COLUMN = 'Therapeutic Dose of Warfarin'
ENZYME_INDUCER_COLUMNS = [
    'Carbamazepine (Tegretol)',
    'Phenytoin (Dilantin)',
    'Rifampin or Rifampicin'
]
AGE_COLUMN = 'Age'
RACE_COLUMN = 'Race'
HEIGHT_COLUMN = 'Height (cm)'
WEIGHT_COLUMN = 'Weight (kg)'
AMIODARONE_COLUMN = 'Amiodarone (Cordarone)'
VKORC1_COLUMN = 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'
VKORC1_COLUMN_2255 = 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G'
VKORC1_COLUMN_1173 = 'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G'
VKORC1_COLUMN_1542 = 'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G'
CYP2C9_COLUMN = 'Cyp2C9 genotypes'

# Weights used by the Clinical Dosing baseline
CLINICAL_DOSING_BASELINE_WEIGHTS = [
    4.0376,
    -0.2546,
    0.0118,
    0.0134,
    -0.6752,
    0.4060,
    0.0443,
    1.2799,
    -0.5695
]