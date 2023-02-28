import numpy as np

def match_dosages(dosages1, dosages2):
    # First condition: dosages1[i] and dosages2[i] are both less than 21
    mask1 = np.logical_and(dosages1 < 21, dosages2 < 21)
    # Second condition: dosages1[i] and dosages2[i] are both between 21 and 49
    mask2 = np.logical_and(dosages1 >= 21, dosages1 < 49)
    mask2 = np.logical_and(mask2, dosages2 >= 21)
    mask2 = np.logical_and(mask2, dosages2 < 49)
    # Third condition: dosages1[i] and dosages2[i] are both greater than 49
    mask3 = np.logical_and(dosages1 >= 49, dosages2 >= 49)
    # Or all three conditions together
    mask = np.logical_or(mask1, mask2)
    mask = np.logical_or(mask, mask3)
    return mask

def compute_accuracy(data, predictions):
    prediction_np = np.array(predictions)
    data_np = np.array(data)
    matching_dosages = match_dosages(prediction_np, data_np)
    accuracy = np.sum(matching_dosages) / len(matching_dosages)
    return accuracy
