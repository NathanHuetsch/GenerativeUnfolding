import numpy as np
from scipy.stats import wasserstein_distance


# taken from https://github.com/ViniciusMikuni/SBUnfold
def GetEMD(ref, array, binning, weights_arr=None, nboot=100):
    if weights_arr is None:
        weights_arr = np.ones(len(ref))
    ds = []
    hists = []
    for _ in range(nboot):
        # ref_boot = np.random.choice(ref,ref.shape[0])
        arr_idx = np.random.choice(range(array.shape[0]), array.shape[0])
        array_boot = array[arr_idx]
        w_boot = weights_arr[arr_idx]
        ds.append(10*wasserstein_distance(ref, array_boot, v_weights=w_boot))
        hists.append(np.histogram(array_boot, weights=w_boot, bins=binning, density=True)[0])
    unc = np.std(hists, 0)
    return np.mean(ds), np.std(ds), unc


# taken from https://github.com/ViniciusMikuni/SBUnfold
def get_triangle_distance(x,y,bins, make_hist=False):
    if make_hist:
        x, _ = np.histogram(x, bins=bins, density=True)
        y, _ = np.histogram(y, bins=bins, density=True)

    dist = 0
    w = bins[1:] - bins[:-1]
    for ib in range(len(x)):
        dist+=0.5*w[ib]*(x[ib] - y[ib])**2/(x[ib] + y[ib]) if x[ib] + y[ib] >0 else 0.0
    return dist*1e3


# taken from https://github.com/ViniciusMikuni/SBUnfold
def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)
