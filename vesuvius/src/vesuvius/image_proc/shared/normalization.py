import numpy as np

def normalize_minmax(volume):
    minim = np.min(volume)
    maxim = np.max(volume)
    if maxim - minim == 0:
        return volume
    volume = volume - minim
    volume = volume / (maxim - minim)
    return volume