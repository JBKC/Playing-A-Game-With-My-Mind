'''
Process json data
8 channels in order: CP3, C3, F5, PO3, PO4, F6, C4, CP4
'''

import json
import scipy.io
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
    signal = signal - np.mean(signal)
    # crop to get rid of edge effects (to update with more elegant method)
    # 100 samples = 490ms
    return signal[100:-100]

def compute_psd(tensor):
    '''
    :param takes in 3D tensor of shape (n_trials, n_channels, n_samples)
    :return:
    power spectral density of each shape (n_trials, n_channels, ceil(n_samples+1/2))
    freqs of shape ceil(n_samples+1/2))
    '''

    freqs, PSD = scipy.signal.welch(tensor, fs=265, axis=2, nperseg=tensor.shape[2])

    return np.array(freqs), np.array(PSD)

def logvar(PSD):
    '''
    Inputs PSD, returns a single power value for the plot as the log variance of the PSD
    :param PSD: shape (n_trials, n_channels, n_psd_points)
    :return: log variance of shape (n_trials, n_channels)
    '''

    return np.log(np.var(PSD, axis=2))

def plot_psd(freqs, P1, P2):

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # plot the first and last eigenvectors (the most extreme discrimination)
    ax1.plot(freqs, P1[:, 0, :].mean(axis=0), color='red', linewidth=1, label='right')
    ax1.plot(freqs, P2[:, 0, :].mean(axis=0), color='blue', linewidth=1, label='left')
    # ax1.set_title(f'PSD for {channel_names[1]} (controls right side)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_xlim(0, 30)
    ax1.set_ylim(1, 50)
    ax1.legend()

    ax2.plot(freqs, P1[:, -1, :].mean(axis=0), color='red', linewidth=1, label='right')
    ax2.plot(freqs, P2[:, -1, :].mean(axis=0), color='blue', linewidth=1, label='left')
    # ax2.set_title(f'PSD for {channel_names[6]} (controls left side)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim(0, 30)
    ax2.set_ylim(1, 50)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def scatter_logvar(L1, L2):

    # plot logvar data of most discriminative eigenvectors

    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

    ax1.scatter(L1[:, 0], L1[:, -1], color='red', linewidth=1, label='right')
    ax1.scatter(L1[:, 0], L2[:, -1], color='blue', linewidth=1, label='left')
    ax1.legend()

    plt.show()

def plot_logvar(L1, L2):

    # bar chart plot of post-CSP log variance

    plt.figure(figsize=(8,5))

    x0 = np.arange(L1.shape[1])
    x1 = np.arange(L1.shape[1]) + 0.4

    y0 = np.mean(L1, axis=0)
    y1 = np.mean(L2, axis=0)

    plt.bar(x0, y0, width=0.4, color='r')
    plt.bar(x1, y1, width=0.4, color='b')

    plt.xlim(-0.5, L1.shape[1] + 0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')

    plt.show()

    return



def whitening_matrix(sigma):

    # eigendecomposition of composite covariance matrix
    U, D, _ = np.linalg.svd(sigma)

    # W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(D + 1e-5)), U.T))               # ZCA
    # W = np.dot(np.diag(1.0 / np.sqrt(D + 1e-5)), U.T)                          # PCA

    return np.dot(np.diag(D ** -0.5), U.T)

def csp(X1, X2, k):
    """
    Compute CSP for two classes of EEG data.

    X1, X2: Shape = (n_trials, n_channels, n_samples)
    k: number of top and bottom eigenvectors
    """

    # calculate covariance matrices
    R1 = np.mean([np.dot(x, x.T) for x in X1], axis=0)
    R2 = np.mean([np.dot(x, x.T) for x in X2], axis=0)

    print(f'Cov shape: {R1.shape}')
    # should be (n_channels, n_channels)

    # get whitening matrix P from composite covariance matrix
    P = whitening_matrix(R1 + R2)
    print(f'Whitening matrix shape: {P.shape}')

    # whiten individual covariance matrices
    S1 = np.dot(np.dot(P, R1), P.T)
    S2 = np.dot(np.dot(P, R2), P.T)

    # print(S1)
    # print(S1+S2)
    # should == identity matrix

    D1, V1, _ = np.linalg.svd(S1)
    D2, V2, _ = np.linalg.svd(S2)

    # solve generalised eigenvalue problem to get spatial filters
    d, W = la.eigh(S1, S1+S2)
    idx = np.argsort(d)[::-1]
    W = W[:,idx]
    # eigenvectors == spatial filters == projection matrix
    print(f'Discriminative eigenvalues {d}')
    print(W)
    print(f'Spatial filter shape: {W.shape}')

    # W = np.column_stack((W[:, 0], W[:, -1]))

    # project data onto spatial filters
    X1_csp = np.stack([np.dot(W.T, trial) for trial in X1])
    X2_csp = np.stack([np.dot(W.T, trial) for trial in X2])

    print(f'X1_csp shape: {X1_csp.shape}')

    return X1_csp, X2_csp


def main():

    k = 1
    X1 = np.empty((0, 0, 0))                     # class 1 data
    X2 = np.empty((0, 0, 0))                     # class 2 data

    # pull trials from saved files
    folder = "data_4 seconds"
    print(f'Pulling trials...')

    for trial in os.listdir(folder):
        if trial == '.DS_Store':
            continue
        # class 1 = right, class 2 = left
        if 'right' in trial:
            # print(os.path.splitext(trial)[0])
            with open(os.path.join(folder, trial), 'r') as f:
                data = json.load(f)

                # get individual trial data
                trial_data = []
                for key, value in data.items():
                    # extract channel data
                    signal = normalise(bpass_filter(value, lowcut=5, highcut=15, fs=256, order=5))
                    # signal = normalise(value)
                    trial_data.append(signal)

                # 2D array of size (no. of channels, length of signal)
                trial_data = np.array(trial_data)

                # If X1 is empty, initialize it with the shape of the first trial
                if X1.size == 0:
                    X1 = trial_data.reshape(1, *trial_data.shape)
                else:
                    # Append the new trial along axis=0
                    X1 = np.append(X1, trial_data.reshape(1, *trial_data.shape), axis=0)

        if 'left' in trial:
            # print(os.path.splitext(trial)[0])
            with open(os.path.join(folder, trial), 'r') as f:
                data = json.load(f)

                # get individual trial data
                trial_data = []
                for key, value in data.items():
                    # extract channel data
                    signal = normalise(bpass_filter(value, lowcut=5, highcut=15, fs=256, order=5))
                    # signal = normalise(value)
                    trial_data.append(signal)

                trial_data = np.array(trial_data)

                if X2.size == 0:
                    X2 = trial_data.reshape(1, *trial_data.shape)
                else:
                    # Append the new trial along axis=0
                    X2 = np.append(X2, trial_data.reshape(1, *trial_data.shape), axis=0)

    print(f'Number of class 1 trials: {X1.shape[0]}')
    print(f'Number of class 2 trials: {X2.shape[0]}')
    print(f'Input data shape: {X1.shape}')

    # pass data through spatial filters
    X1, X2 = csp(X1=X1, X2=X2, k=k)

    # get Power Spectral Densities
    freqs, P1 = compute_psd(X1)
    _, P2 = compute_psd(X2)
    plot_psd(freqs, P1, P2)

    # get Log Variance
    L1 = logvar(P1)
    L2 = logvar(P2)
    scatter_logvar(L1,L2)

    return


if __name__ == '__main__':
    main()




