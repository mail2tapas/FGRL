
import numpy as np

def aggregate(weight_list):
    if len(weight_list) == 0:
        return None
    return np.mean(weight_list, axis=0)
