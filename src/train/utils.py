import numpy as np
from scipy.stats import wasserstein_distance


# adapted from https://github.com/ViniciusMikuni/SBUnfold
def GetEMD(ref, array, weights_arr=None, nboot=100):
    if weights_arr is None:
        weights_arr = np.ones(len(ref))
    ds = []
    for _ in range(nboot):
        # ref_boot = np.random.choice(ref,ref.shape[0])
        arr_idx = np.random.choice(range(array.shape[0]), array.shape[0])
        array_boot = array[arr_idx]
        w_boot = weights_arr[arr_idx]
        ds.append(10*wasserstein_distance(ref, array_boot, v_weights=w_boot))
    return np.mean(ds), np.std(ds)


# adapted from https://github.com/ViniciusMikuni/SBUnfold
def get_triangle_distance(true, predicted, bins, weights=None, nboot=100):
    x, _ = np.histogram(true, bins=bins, density=True)
    ds = []
    for _ in range(nboot):
        arr_idx = np.random.choice(range(predicted.shape[0]), predicted.shape[0])
        predicted_boot = predicted[arr_idx]
        y, _ = np.histogram(predicted, bins=bins, density=True, weights=weights)
        dist = 0
        w = bins[1:] - bins[:-1]
        for ib in range(len(x)):
            dist+=0.5*w[ib]*(x[ib] - y[ib])**2/(x[ib] + y[ib]) if x[ib] + y[ib] >0 else 0.0
        ds.append(dist*1e3)
    return np.mean(ds), np.std(ds)


# taken from https://github.com/ViniciusMikuni/SBUnfold
def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)
