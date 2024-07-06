'''
Process json data
'''

import json
from scipy.signal import butter, filtfilt
import numpy as np
from numpy import linalg as la
import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_trials():
    pass


def bpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalise(signal):
    # zero mean the signal
    return signal - np.mean(signal)



def csp(X0, X1, k):
    """
    Compute CSP for two classes of EEG data.

    X0, X1: EEG data for class 0 and class 1
            Shape: (n_trials, n_channels, n_samples)
            trials = number of individual recordings
            channels = 4 (C3,C4,PC3,PC4)
            samples = 768 datapoints
    k: number of top and bottom eigenvectors
    """
    # compute covariance matrices
    cov1 = np.mean([np.cov(trial) for trial in X0], axis=0)
    cov2 = np.mean([np.cov(trial) for trial in X1], axis=0)

    # get composite covariance + do eigendecomposition
    cov_comp = cov1 + cov2
    eig_values, eig_vectors = la.eigh(cov_comp)
    # sort eigs
    idx = np.argsort(eig_values)[::-1]
    eig_values, eig_vectors = eig_values[idx], eig_vectors[:, idx]

    # whitening transformation
    W = np.dot(np.diag(np.sqrt(1/(eig_values + 1e-6))), eig_vectors.T)
    print(W.shape)

    # calculate spatial filters
    S1 = np.dot(np.dot(cov1, W.T), W)
    S2 = np.dot(np.dot(cov2, W.T), W)

    # generalised eigendecomposition
    eig_values, eig_vectors = np.linalg.eigh(S1, S1 + S2)           # key line that maximises variances for one class and minimises for the other
    idx = np.argsort(eig_values)[::-1]
    eig_values, eig_vectors = eig_values[idx], eig_vectors[:, idx]

    # full CSP projection matrix
    W_csp = np.dot(eig_vectors.T, W)

    # Select top CSP components
    W_csp = np.concatenate((W_csp[:, :k], W_csp[:, -k:]), axis=1)

    return W_csp


def main():

    k = 3
    X0 = np.empty((0, 0, 0))                     # class 0 data
    X1 = np.empty((0, 0, 0))                     # class 1 data

    # pull trials from saved files
    folder = os.getcwd()
    for trial in os.listdir(folder):
        if trial == '.DS_Store':
            continue

        # class 0 = relaxed state, class 1 = motor imagery
        if 'raise right hand' in trial:
            print(os.path.splitext(trial)[0])
            with open(os.path.join(folder, trial), 'r') as f:
                data = json.load(f)

                # get individual trial data
                trial_data = []
                for key, value in data.items():
                    # extract channel data
                    filtered = normalise(bpass_filter(value, lowcut=7, highcut=30.0, fs=256, order=5))
                    trial_data.append(filtered)

                # 2D array of size (no. of channels, length of signal)
                trial_data = np.array(trial_data)

                # If X0 is empty, initialize it with the shape of the first trial
                if X0.size == 0:
                    X0 = trial_data.reshape(1, *trial_data.shape)
                else:
                    # Append the new trial along axis=0
                    X0 = np.append(X0, trial_data.reshape(1, *trial_data.shape), axis=0)

        if 'raise left hand' in trial:
            print(os.path.splitext(trial)[0])
            with open(os.path.join(folder, trial), 'r') as f:
                data = json.load(f)

                # get individual trial data
                trial_data = []
                for key, value in data.items():
                    # extract channel data
                    filtered = normalise(bpass_filter(value, lowcut=7, highcut=30.0, fs=256, order=5))
                    trial_data.append(filtered)

                trial_data = np.array(trial_data)

                if X1.size == 0:
                    X1 = trial_data.reshape(1, *trial_data.shape)
                else:
                    # Append the new trial along axis=0
                    X1 = np.append(X1, trial_data.reshape(1, *trial_data.shape), axis=0)

    # print(X0.shape)
    csp(X0=X0, X1=X1, k=k)

    return


if __name__ == '__main__':
    main()
