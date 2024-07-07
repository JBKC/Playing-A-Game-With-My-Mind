'''
Process json data
'''

import json
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.linalg as la
import os
from sklearn.covariance import MinCovDet
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
    # take log of power across each CSP, resulting in shape (no. of trials, no. of CSP)
    return np.log(np.mean(np.abs(tensor)**2, axis=2) + 1e-10)

def whitening_matrix(X1, X2):

    # combine data from both classes
    X = np.concatenate((X1, X2), axis=0)

    # compress all features into a single vector giving shape (n_trials, n_channels * n_samples)
    X = X.reshape((-1, np.prod(X.shape[1:])))
    print(X.shape)

    # centre data
    X = X - np.mean(X, axis=0)

    # get composite covariance matrix
    SX = np.dot(X.T, X) / X.shape[0]

    # eigendecomposition
    D, U = la.eig(SX)                       # D,V = eigenvalues, eigenvectors
    idx = np.argsort(D)[::-1]               # sort D,V
    D = D[idx]
    U = U[:,idx]

    # return whitening matrix
    return np.dot(U, np.dot(np.diag(1.0 / np.sqrt(D + 1e-5)), U.T))


def whiten(x,y,W):

    # reshape
    x = x.reshape((-1, np.prod(x.shape[1:])))
    y = y.reshape((-1, np.prod(y.shape[1:])))
    # whiten
    x = np.dot(x, W.T)
    y = np.dot(y, W.T)
    # revert shape
    x = x.reshape(25, 4, 768)
    y = y.reshape(25, 4, 768)

    return x,y




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

    X1 = X1.reshape((-1, np.prod(X1.shape[1:])))
    X2 = X2.reshape((-1, np.prod(X2.shape[1:])))

    # get covariance matrices for raw data
    epsilon = 1e-6
    S1 = np.dot(X1.T, X1) / X1.shape[0]
    S2 = np.dot(X2.T, X2) / X2.shape[0]
    S1 += epsilon * np.eye(S1.shape[0])
    S2 += epsilon * np.eye(S2.shape[0])

    # get whitening matrix
    W = whitening_matrix(X1,X2)
    print(f'Whitening matrix shape: {W.shape}')
    print(W)
    print(np.count_nonzero(W == 0))

    # whiten data
    X1,X2 = whiten(X1,X2,W)

    # whiten individual covariance matrices
    S1 = np.dot(S1, W.T)
    S2 = np.dot(S2, W.T)

    # solve generalised eigenvalue problem to get spatial filters
    D, V = la.eigh(S1, S1+S2)
    idx = np.argsort(D)[::-1]               # sort D,V
    V = V[:,idx]                            # eigenvectors = spatial filters

    # select k most important eigenvectors
    V_CSP = np.concatenate((V[:, :k], V[:, -k:]), axis=1).T

    # project data
    X1_CSP = np.array([np.dot(V_CSP, trial) for trial in X1])
    X2_CSP = np.array([np.dot(V_CSP, trial) for trial in X2])

    print(X1_CSP.shape)

    return X1, X2, X1_CSP, X2_CSP


def main():

    k = 3
    X1 = np.empty((0, 0, 0))                     # class 1 data
    X2 = np.empty((0, 0, 0))                     # class 2 data

    # pull trials from saved files
    folder = "data_3 seconds"
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
    W1, W2, X1_CSP, X2_CSP = csp(X1=X1, X2=X2, k=k)

    # return power
    return compute_power(X1), compute_power(X2), compute_power(W1), compute_power(W2), compute_power(X1_CSP), compute_power(X2_CSP)


if __name__ == '__main__':
    PX1, PX2, PW1, PW2, PX1CSP, PX2CSP = main()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot pre-CSP data
    for i in PX1:
        ax1.scatter(i[0],i[1],color='red',label='class 1')
    for j in PX2:
        ax1.scatter(j[0],j[1],color='blue',label='class 2')
    ax1.set_title('Raw data')

    # plot whitened data
    for i in PW1:
        ax2.scatter(i[0],i[1],color='red',label='class 1')
    for j in PW2:
        ax2.scatter(j[0],j[1],color='blue',label='class 2')
    ax2.set_title('Whitened')

    plt.show()


