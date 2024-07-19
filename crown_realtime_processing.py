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


def plot_psd(freqs, P):
    '''
    :params: P (power) shape: (1, n_CSP_components, n_samples/2 + 1)
    '''

    channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # plot the first and last eigenvectors (the most extreme discrimination)
    ax1.plot(freqs, P[:, 0, :].mean(axis=0), color='red', linewidth=1, label='first eigenvalue')
    ax1.plot(freqs, P[:, -1, :].mean(axis=0), color='blue', linewidth=1, label='last eigenvalue')
    # ax1.set_title(f'PSD for {channel_names[1]} (controls right side)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 1)
    ax1.legend()

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
    # Z-score normalisation
    return (signal - np.mean(signal)) / np.std(signal)

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
    model_file = joblib.load('/Users/jamborghini/Documents/PYTHON/neurosity_multicontrol/models/lda_2024-07-19 22:37:28.677998.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    print(f'Spatial Filters shape: {W.shape}')

    X = np.transpose(np.array(window), (1, 0))             # shape (1, n_channels, n_samples)
    X = np.expand_dims(X, axis=0)
    print(f'Input window shape: {X.shape}')

    # pass window through preprocessing steps
    # normalise + bandpass
    X = normalise(bpass_filter(X, 8, 15, 256))

    # apply spatial filters to single trial
    X_csp = np.stack([np.dot(W.T, trial) for trial in X])
    print(f'X_csp shape: {X_csp.shape}')            # should be (1, n_CSP_components, n_samples)

    # get Power Spectral Density (optional visualisation)
    # freqs, P = compute_psd(X_csp)
    # plot_psd(freqs, P)

    # get Log Variance
    L = logvar(X_csp)

    print(f'Inference shape: {L.shape}')          # should be (1, n_CSP_components)

    # predict class probabilities
    probs = model.predict_proba(L)
    print(probs)

    return


if __name__ == '__main__':
    main()





