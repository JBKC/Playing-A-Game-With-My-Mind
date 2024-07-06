'''
Process json data
'''

import json
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.linalg as la
import os
import pandas as pd
import matplotlib.pyplot as plt

def bpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalise(signal):
    # zero mean the signal
    return signal - np.mean(signal)

def whiten(X1, X2):

    # get overall covariance matrix
    X = np.concatenate((X1, X2), axis=0)
    SX = np.mean([np.cov(trial) for trial in X], axis=0)        # both classes covariance matrix

    # eigendecomposition
    D, V = la.eigh(SX)                      # D,V = eigenvalues, eigenvectors
    idx = np.argsort(D)[::-1]               # sort D,V
    D = D[idx]
    V = V[:,idx]

    # return whitening matrix
    return np.dot(np.diag(np.sqrt(1 / (D + 1e-6))), V.T)

def compute_power(tensor):
    # take log of power across each CSP, resulting in shape (no. of trials, no. of CSP)
    return np.log(np.mean(np.abs(tensor)**2, axis=2) + 1e-10)

def csp(X1, X2, k):
    """
    Compute CSP for two classes of EEG data.

    X1, X2: EEG data for class 1 and class 2
            Shape: (n_trials, n_channels, n_samples)
            trials = number of individual recordings
            channels = 4 (C3,C4,PC3,PC4)
            samples = 768 datapoints
    k: number of top and bottom eigenvectors
    """
    # get covariance matrices across channels for each trial, then average across trials for each class
    S1 = np.mean([np.cov(trial) for trial in X1], axis=0)       # class 1 covariance matrix
    S2 = np.mean([np.cov(trial) for trial in X2], axis=0)       # class 2 covariance matrix

    # get whitening matrix
    W = whiten(X1,X2)

    # whiten data
    X1 = np.einsum('ij,kjl->kil', W.T, X1)
    X2 = np.einsum('ij,kjl->kil', W.T, X2)

    # whiten individual covariance matrices
    S1 = np.dot(np.dot(W.T, S1), W)
    S2 = np.dot(np.dot(W.T, S2), W)

    # solve generalised eigenvalue problem to get spatial filters
    D, V = la.eigh(S1, S1+S2)
    idx = np.argsort(D)[::-1]               # sort D,V
    V = V[:,idx]                            # eigenvectors = spatial filters

    # select k most important eigenvectors
    V_CSP = np.concatenate((V[:, :k], V[:, -k:]), axis=1).T

    # project data
    X1_CSP = np.array([np.dot(V_CSP, trial) for trial in X1])
    X2_CSP = np.array([np.dot(V_CSP, trial) for trial in X2])

    return X1_CSP, X2_CSP




def main():

    k = 3
    X1 = np.empty((0, 0, 0))                     # class 1 data
    X2 = np.empty((0, 0, 0))                     # class 2 data

    # pull trials from saved files
    folder = os.getcwd()
    for trial in os.listdir(folder):
        if trial == '.DS_Store':
            continue

        # class 1 = relaxed state, class 2 = motor imagery
        if 'raise right arm' in trial:
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

                # If X1 is empty, initialize it with the shape of the first trial
                if X1.size == 0:
                    X1 = trial_data.reshape(1, *trial_data.shape)
                else:
                    # Append the new trial along axis=0
                    X1 = np.append(X1, trial_data.reshape(1, *trial_data.shape), axis=0)

        if 'raise left arm' in trial:
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

                if X2.size == 0:
                    X2 = trial_data.reshape(1, *trial_data.shape)
                else:
                    # Append the new trial along axis=0
                    X2 = np.append(X2, trial_data.reshape(1, *trial_data.shape), axis=0)

    # pass through spatial filters
    X1, X2 = csp(X1=X1, X2=X2, k=k)

    # return power
    return compute_power(X1), compute_power(X2)


if __name__ == '__main__':
    main()
