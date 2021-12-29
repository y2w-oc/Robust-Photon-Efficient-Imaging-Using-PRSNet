# The train, validation list make function
import numpy as np
from glob import glob
import re
import os.path
import sys

# specify dataset folder here that contains the output of SimulateTrainMeasurements.m
dataset_folder = os.path.abspath('../../data/TrainData/processed') + '/'

simulation_param_idx = 3    # 1 or 2 corresponding to that in SimulateTrainMeasurements.m

def intersect_files(train_files):
    intensity_train_files = []
    for t in train_files:
        intensity_train_files.append(glob(dataset_folder + t + 'intensity*.mat'))
    intensity_train_files = [file for sublist in intensity_train_files for file in sublist]

    spad_train_files = []
    if simulation_param_idx is not None:
        noise_param = [simulation_param_idx]
    else:
        print("Specify simulation_param_idx = 1 or 2")
        sys.exit("SIMULATION PARAMETER INDEX ERROR")
    for p in noise_param:
        spad_train_files.append([])
        for t in train_files:
            spad_train_files[-1].append(glob(dataset_folder + t + 'spad*p{}.mat'.format(p)))
        spad_train_files[-1] = [file for sublist in spad_train_files[-1] for file in sublist]
        spad_train_files[-1] = [re.sub(r'(.*)/spad_(.*)_p.*.mat', r'\1/intensity_\2.mat',
                                 file) for file in spad_train_files[-1]]

    intensity_train_files = set(intensity_train_files) 
    for idx, p in enumerate(noise_param):
        spad_train_files[idx] = set(spad_train_files[idx])
    intensity_train_files = intensity_train_files.intersection(*tuple(spad_train_files))

    return intensity_train_files

def main():

    with open('train.txt') as f:
        train_files = f.read().split()
    with open('val.txt') as f:
        val_files = f.read().split()

    print('Sorting training files')
    intensity_train_files = intersect_files(train_files)
    print('Sorting validation files')
    intensity_val_files = intersect_files(val_files)

    print('Writing training files')
    with open('train_intensity.txt', 'w') as f:
        for file in intensity_train_files:
            f.write(file + '\n')
    print('Writing validation files')
    with open('val_intensity.txt', 'w') as f:
        for file in intensity_val_files:
            f.write(file + '\n')

    print('Wrote {} train, {} validation files'.format(len(intensity_train_files),
                                                       len(intensity_val_files)))
    return

if __name__ == '__main__':
    main()





