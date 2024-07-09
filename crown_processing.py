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

def compute_power(X1,X2):
    '''

    :param tensor: takes in data shape (25,4,768)
    :return: power of each signal, with dimension (25,4)
    '''

    P1 = np.log(np.mean(np.abs(X1) ** 2, axis=2) + 1e-10)
    P2 = np.log(np.mean(np.abs(X2) ** 2, axis=2) + 1e-10)
    #
    # total_mean = np.mean(np.concatenate((P1, P2), axis=0))
    #
    # P1 = P1 - total_mean
    # P2 = P2 - total_mean

    return P1, P2

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


def csp(X1, X2, W1, W2, W, k):
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
    W1 = W1.reshape((-1, np.prod(W1.shape[1:])))
    W2 = W2.reshape((-1, np.prod(W2.shape[1:])))

    # get covariance matrices
    S1 = np.dot(W1.T, W1) / W1.shape[0]
    S2 = np.dot(W2.T, W2) / W2.shape[0]
    S1 += 1e-6 * np.eye(S1.shape[0])
    S2 += 1e-6 * np.eye(S2.shape[0])

    # solve generalised eigenvalue problem to get spatial filters
    d, V = la.eigh(S1, S1+S2)

    # idx = np.argsort(d)[::-1]
    # V = V[:,idx]                            # eigenvectors == spatial filters
    # print(f'Eigenvectors shape: {V.shape}')


    # select k most important eigenvectors
    V_csp = V
    print(f'Spatial filter shape: {V_csp.shape}')

    # project data onto spatial filters
    X1_csp = np.dot(W1, V_csp.T)
    X2_csp = np.dot(W2, V_csp.T)

    X1_csp = X1_csp.reshape(25, 4, 768)
    X2_csp = X2_csp.reshape(25, 4, 768)

    # V_csp = np.concatenate((V[:, :k], V[:, -k:]), axis=1).T


    return X1_csp, X2_csp


def scatter_plots(dict):

    # channels: CP3, C3, C4, CP4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # plot 1
    # takes data (25, 4, 768) - plot first 2 channels for all 25 trials
    for i in dict['X1']:
        ax1.scatter(i[1], i[2], color='red', label='class 1')
    for j in dict['X2']:
        ax1.scatter(j[1], j[2], color='blue', label='class 2')

    for i in dict['X1 CSP']:
        ax2.scatter(i[1], i[2], color='red', label='class 1')
    for j in dict['X2 CSP']:
        ax2.scatter(j[1], j[2], color='blue', label='class 2')

    # plot 2
    # takes data (25, 6)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5))
    # ax1.scatter(dict['X1 CSP'][:, 3], dict['X1 CSP'][:, 1], color='red')      # plot CSP dimension 1 vs. 2
    # ax1.scatter(dict['X2 CSP'][:, 3], dict['X2 CSP'][:, 1], color='blue')
    # ax2.scatter(dict['X1 CSP'][:, 3], dict['X1 CSP'][:, 2], color='red')
    # ax2.scatter(dict['X2 CSP'][:, 3], dict['X2 CSP'][:, 2], color='blue')
    # ax3.scatter(dict['X1 CSP'][:, 3], dict['X1 CSP'][:, 0], color='red')
    # ax3.scatter(dict['X2 CSP'][:, 3], dict['X2 CSP'][:, 0], color='blue')
    # ax4.scatter(dict['X1 CSP'][:, 3], dict['X1 CSP'][:, 4], color='red')
    # ax4.scatter(dict['X2 CSP'][:, 3], dict['X2 CSP'][:, 4], color='blue')
    # ax5.scatter(dict['X1 CSP'][:, 3], dict['X1 CSP'][:, 5], color='red')
    # ax5.scatter(dict['X2 CSP'][:, 3], dict['X2 CSP'][:, 5], color='blue')


    plt.show()

def main():

    k = 3
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

    # pass data through spatial filters
    X1_csp, X2_csp = csp(X1=X1, X2=X2, W1=W1, W2=W2, W=W, k=k)

    print(f'Whitened data shape: {W1.shape}')
    print(f'Filtered data shape: {X1_csp.shape}')

    # compute power
    P1, P2 = compute_power(W1, W2)
    P1_csp, P2_csp = compute_power(X1_csp, X2_csp)
    print(f'Power shape: {P1.shape}')


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




