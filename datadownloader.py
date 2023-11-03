"""
Script to download the data used for the SBUnfold paper. Same data as Omnifold paper.
Downloads from here     https://zenodo.org/records/3548091

The first part of the script downloads all parts of the data in a lot of *.npz files.
The second part merges them into one file per dataset.

Need to set the save_dir variable at the top. The rest should run then.
The "while" and "try" loop looks a bit scuffed. The download sometimes fails, this restarts it at the last array.
"""



import energyflow
import numpy as np
import os

save_dir = '/remote/gpu07/huetsch/data'
datasets = ['Herwig', 'Pythia21', 'Pythia25', 'Pythia26']

for dataset in datasets:
    finished = False
    while not finished:
        try:
            energyflow.zjets_delphes.load(dataset, num_data=-1, pad=False, cache_dir=save_dir,
                                                   source='zenodo', which='all',
                                                   include_keys=None, exclude_keys=None)
            finished=True
        except:
            print("Failed", dataset)

data_dir = os.path.join(save_dir, 'datasets', 'ZjetsDelphes')
all_files = os.listdir(data_dir)
all_datasets = ['Herwig', 'Pythia21', 'Pythia25', 'Pythia26']

for dataset in all_datasets:
    out_dict = {}
    outfile = os.path.join(data_dir, dataset + "_full.npz")

    dataset_files = [os.path.join(data_dir, file) for file in all_files if dataset in file]
    assert len(dataset_files) == 17, dataset_files
    data = [np.load(file) for file in dataset_files]
    assert len(data) == 17

    keys = data[0].files

    for key in keys:
        if "particles" in key:
            continue
        placeholder = []
        for i in range(17):
            placeholder.append(data[i][key])
        out_dict[key] = np.concatenate(placeholder, axis=0)

    with open(outfile, "wb") as f:
        np.savez(f, out_dict)

