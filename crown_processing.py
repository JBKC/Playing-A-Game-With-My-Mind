'''
Process json data
'''

import json
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.linalg as la
import os
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

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

def compute_power(tensor):
    '''

    :param tensor: takes in data shape (25,4,768)
    :return: power of each
    '''
    # take log of power across each CSP, resulting in shape (no. of trials, no. of CSP)
    return np.log(np.mean(np.abs(tensor)**2, axis=2) + 1e-10)

def whitening_matrix(X1, X2):

    # combine data from both classes
    X = np.concatenate((X1, X2), axis=0)

    # compress all features into a single vector giving shape (n_trials, n_channels * n_samples)
    X = X.reshape((-1, np.prod(X.shape[1:])))

    # centre data
    X = X - np.mean(X, axis=0)

    # get composite covariance matrix
    SX = np.dot(X.T, X) / X.shape[0]

    # eigendecomposition of composite covariance matrix
    U, D, _ = np.linalg.svd(SX)

    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(D + 1e-5)), U.T))               # ZCA
    # W = np.dot(np.diag(1.0 / np.sqrt(D + 1e-5)), U.T)                          # PCA

    # return whitening matrix
    return W


def whiten(X1,X2,W):

    # reshape data
    X1 = X1.reshape((-1, np.prod(X1.shape[1:])))
    X2 = X2.reshape((-1, np.prod(X2.shape[1:])))

    # whiten mean-centred data
    X1 = np.dot(X1 - np.mean(X1, axis=0), W.T)
    X2 = np.dot(X2 - np.mean(X2, axis=0), W.T)

    # revert shape
    X1 = X1.reshape(25, 4, 768)
    X2 = X2.reshape(25, 4, 768)

    return X1,X2


def csp(X1, X2, k):
    """
    Compute CSP for two classes of EEG data.

    X1, X2: Whitened EEG data for class 1 and class 2
            Shape: (n_trials, n_channels, n_samples)
            trials = number of individual recordings
            channels = 4 (C3,C4,PC3,PC4)
            samples = 768 datapoints
    W: whitening matrix
    k: number of top and bottom eigenvectors
    """

    # rehsape data
    X1 = X1.reshape((-1, np.prod(X1.shape[1:])))
    X2 = X2.reshape((-1, np.prod(X2.shape[1:])))

    # get covariance matrices
    S1 = np.dot(X1.T, X1) / X1.shape[0]
    S2 = np.dot(X2.T, X2) / X2.shape[0]
    S1 += 1e-6 * np.eye(S1.shape[0])
    S2 += 1e-6 * np.eye(S2.shape[0])

    # solve generalised eigenvalue problem to get spatial filters
    D, V = la.eigh(S1, S1+S2)
    idx = np.argsort(D)[::-1]
    V = V[:,idx]                            # eigenvectors == spatial filters

    # select k most important eigenvectors
    V_csp = np.concatenate((V[:, :k], V[:, -k:]), axis=1).T
    print(f'Spatial filter shape: {V_csp.shape}')

    # project data onto spatial filters
    X1_csp = np.dot(X1, V_csp.T)
    X2_csp = np.dot(X2, V_csp.T)

    X1_csp = X1_csp.reshape(25, 4, 768)
    X2_csp = X2_csp.reshape(25, 4, 768)

    return X1_csp, X2_csp


def scatter_plots(dict):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot 1
    for i in dict['W1']:
        ax1.scatter(i[0], i[1], color='red', label='class 1')
    for j in dict['W2']:
        ax1.scatter(j[0], j[1], color='blue', label='class 2')
    ax1.set_title('Raw data')

    # plot 2
    for i in dict['X1']:
        ax2.scatter(i[0], i[1], color='red', label='class 1')
    for j in dict['X2']:
        ax2.scatter(j[0], j[1], color='blue', label='class 2')
    ax2.set_title('Post CSP')

    plt.show()

def main():

    k = 1536
    X1 = np.empty((0, 0, 0))                     # class 1 data
    X2 = np.empty((0, 0, 0))                     # class 2 data

    # pull trials from saved files
    folder = "data_3 seconds"
    print(f'Pulling trials...')

    for trial in os.listdir(folder):
        if trial == '.DS_Store':
            continue
        # class 1 = relaxed state, class 2 = motor imagery
        if 'raise right arm' in trial:
            # print(os.path.splitext(trial)[0])
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
            # print(os.path.splitext(trial)[0])
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

    print(f'Number of class 1 trials: {X1.shape[0]}')
    print(f'Number of class 2 trials: {X2.shape[0]}')

    # get whitening matrix
    W = whitening_matrix(X1=X1, X2=X2)
    print(f'Whitening matrix shape: {W.shape}')

    # whiten data
    W1, W2 = whiten(X1=X1, X2=X2, W=W)

    # pass whitened data through spatial filters
    X1_csp, X2_csp = csp(X1=W1, X2=W2, k=k)

    print(f'Raw data shape: {X1.shape}')
    print(f'Filtered data shape: {X1_csp.shape}')

    # compute power of each
    P1 = compute_power(W1)
    P2 = compute_power(W2)
    P1_csp = compute_power(X1_csp)
    P2_csp = compute_power(X2_csp)

    data_dict = {
        'X1': X1,                   # raw data
        'X2': X2,
        'W1': W1,                   # whitened data
        'W2': W2,
        'X1 CSP': X1_csp,           # post-CSP data
        'X2 CSP': X2_csp,
        'P1': P1,                   # whitened power data
        'P2': P2,
        'P1 CSP': P1_csp,           # post-CSP power data
        'P2 CSP': P2_csp
    }

    scatter_plots(data_dict)



    # compute power
    return


if __name__ == '__main__':
    main()




