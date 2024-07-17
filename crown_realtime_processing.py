'''
Process realtime data sent in windows by async
Takes in data of shape (512,8)
'''

import json
import scipy.io
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.linalg
import os
import matplotlib.pyplot as plt
import joblib


def plot_psd(freqs, P1, P2):
    '''
    :params P1, P2: shape (n_trials, n_channels, n_samples/2 + 1)
    '''

    channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # plot the first and last eigenvectors (the most extreme discrimination)
    ax1.plot(freqs, P1[:, 0, :].mean(axis=0), color='red', linewidth=1, label='right')
    ax1.plot(freqs, P2[:, 0, :].mean(axis=0), color='blue', linewidth=1, label='left')
    # ax1.set_title(f'PSD for {channel_names[1]} (controls right side)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 1)
    ax1.legend()

    ax2.plot(freqs, P1[:, -1, :].mean(axis=0), color='red', linewidth=1, label='right')
    ax2.plot(freqs, P2[:, -1, :].mean(axis=0), color='blue', linewidth=1, label='left')
    # ax2.set_title(f'PSD for {channel_names[6]} (controls left side)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def bpass_filter(data, lowcut, highcut, fs, order=5):

    # signal params
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, data, axis=2)

def normalise(signal):
    # zero mean the signal
    return signal - np.mean(signal)

def compute_psd(tensor):
    '''
    :param takes in 3D tensor of shape (n_trials, n_channels, n_samples)
    :return:
    power spectral density of each shape (n_trials, n_channels, n_samples/2 + 1)
    freqs of shape ceil(n_samples+1/2))
    '''

    freqs, PSD = scipy.signal.welch(tensor, fs=265, axis=2, nperseg=tensor.shape[2])

    return np.array(freqs), np.array(PSD)

def spatial_filter(X1, X2):

    """
    Compute CSP for two classes of EEG data.

    X1, X2: Shape = (n_trials, n_channels, n_samples)
    """

    def cov(X):
        '''
        Take in 3D tensor X of shape (n_trials, n_channels, n_samples)
        Calculates the covariance matrix of shape (n_channels, n_channels)
        Normalise the covs by dividing by number of samples
        Returns average of covs over all trials
        '''
        n_samples = X.shape[2]
        covs = [np.dot(trial, trial.T) / n_samples for trial in X]

        return np.mean(covs, axis=0)

    def whitening_matrix(sigma):
        # eigendecomposition of composite covariance matrix
        D, U = np.linalg.eigh(sigma)

        # use PCA whitening (without transformation back into original space)
        return np.dot(np.diag(D ** -0.5), U.T)

    # calculate covariance matrices
    R1 = cov(X1)
    R2 = cov(X2)
    print(f'Covariance matrix  shape: {R1.shape}')                    # shape (n_channels, n_channels)

    # get whitening matrix P from composite covariance matrix
    P = whitening_matrix(R1 + R2)
    print(f'Whitening matrix shape: {P.shape}')                       # shape (n_channels, n_channels)

    # whiten covariance matrices
    S1 = np.dot(np.dot(P, R1), P.T)
    S2 = np.dot(np.dot(P, R2), P.T)

    # solve generalised eigenvalue problem
    D, U = scipy.linalg.eigh(S1, S1+S2)

    # sort eigenvalues in descending order
    idx = np.argsort(D)[::-1]
    D = D[idx]
    W = U[:, idx]               # W == spatial filters
    print(f'Discriminative eigenvalues: {D}')

    # transform spatial filters back into original space
    W = np.dot(P.T ,W)
    # keep top eigenvectors
    W = np.column_stack((W[:, :1], W[:, -1:]))

    # project spatial filters onto data
    X1_csp = np.stack([np.dot(W.T, trial) for trial in X1])
    X2_csp = np.stack([np.dot(W.T, trial) for trial in X2])

    return X1_csp, X2_csp

def main(window):

    # load saved model & spatial filters (move this to pre-recording in async file)
    model_file = joblib.load('models/lda_2024-07-17 18:18:10.377098.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']
    print(W.shape)
    print(W)

    window = np.array(window)
    print(window.shape)

    # pass window through preprocessing steps
    window = normalise(bpass_filter(window, 8, 15, 256))

    #### # MODEL EXPECTS SHAPE (1,2)

    model_file = joblib.load('models/lda_2024-07-17 18/18/10.377098.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']
    print(W.shape)
    print(W)


    #######

    # pass data through spatial filters using CSP
    X1, X2 = spatial_filter(X1=X1, X2=X2)


    # get Power Spectral Densities from spatially filtered data
    freqs, P1 = compute_psd(X1)
    _, P2 = compute_psd(X2)

    # get Log Variance
    L1 = logvar(X1)
    L2 = logvar(X2)

    # plots
    # plot_psd(freqs, P1, P2)
    # bar_logvar(L1,L2)
    # scatter_logvar(L1,L2)

    print(f'Input data shape: {L1.shape}')

    # return log variance as input into models
    return L1, L2


if __name__ == '__main__':
    main()





