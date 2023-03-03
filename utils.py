from constants import DOSAGE_BUCKET_COLUMN, Actions

def get_reward(observation, action):
    return 0 if observation[DOSAGE_BUCKET_COLUMN] == action else -1

def convert_dosage_to_action(dosage):
    return Actions.LOW.value if dosage < 21 else Actions.MEDIUM.value if dosage < 49 else Actions.HIGH.value