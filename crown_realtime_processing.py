'''
Classify realtime data sent in windows by async
Takes in data of shape (512,8)
'''

import scipy.io
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.linalg
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

def logvar(PSD):
    '''
    Inputs PSD, returns a single power value for the plot as the log variance of the PSD
    :param PSD: shape (n_trials, n_channels, n_psd_points)
    :return: log variance of shape (n_trials, n_channels)
    '''

    return np.log(np.var(PSD, axis=2))

def compute_psd(tensor):
    '''
    :param takes in 3D tensor of shape (n_trials, n_channels, n_samples)
    :return:
    power spectral density of each shape (n_trials, n_channels, n_samples/2 + 1)
    freqs of shape ceil(n_samples+1/2))
    '''

    freqs, PSD = scipy.signal.welch(tensor, fs=265, axis=2, nperseg=tensor.shape[2])

    return np.array(freqs), np.array(PSD)

def main(window):

    # load saved model & spatial filters (move this to pre-recording in async file)
    model_file = joblib.load('models/lda_2024-07-17 18:18:10.377098.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    X = np.transpose(np.array(window), (1, 0))             # shape (n_channels, n_samples)
    print(f'Input window shape: {X.shape}')

    # pass window through preprocessing steps
    # normalise + bandpass
    X = normalise(bpass_filter(X, 8, 15, 256))

    # apply spatial filters
    X_csp = np.dot(W.T, X)
    print(f'X_csp shape: {X_csp.shape}')            # should be (n_CSP_components, n_samples)

    # get Power Spectral Density (optional visualisation)
    # freqs, P = compute_psd(X_csp)
    # plot_psd(freqs, P)

    # get Log Variance
    L = logvar(X_csp)

    print(f'Ineference shape: {L.shape}')          # should be (1, n_CSP_components)

    y_pred = model.predict(L)

    # print predicted class here

    return


if __name__ == '__main__':
    main()





