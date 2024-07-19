'''
Process saved json data for ML training
8 channels in order: CP3, C3, F5, PO3, PO4, F6, C4, CP4
'''

import json
import scipy.io
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.linalg
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_psd(freqs, P1, P2):
    '''
    :params P1, P2: shape (n_trials, n_channels, n_samples/2 + 1)
    '''

    channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # plot the first and last eigenvectors (the most extreme discrimination)
    ax1.plot(freqs, P1[:, 1, :].mean(axis=0), color='red', linewidth=1, label='right')
    ax1.plot(freqs, P2[:, 1, :].mean(axis=0), color='blue', linewidth=1, label='left')
    ax1.set_title(f'PSD for {channel_names[1]} (controls right side)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 100)
    ax1.legend()

    ax2.plot(freqs, P1[:, 6, :].mean(axis=0), color='red', linewidth=1, label='right')
    ax2.plot(freqs, P2[:, 6, :].mean(axis=0), color='blue', linewidth=1, label='left')
    ax2.set_title(f'PSD for {channel_names[6]} (controls left side)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 100)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def logvar(PSD):
    '''
    Inputs PSD, returns a single power value for the plot as the log variance of the PSD
    :param PSD: shape (n_trials, n_channels, n_psd_points)
    :return: log variance of shape (n_trials, n_channels)
    '''

    return np.log(np.var(PSD, axis=2))

def bar_logvar(L1, L2):
    '''
    Plot logvar bar chart of all channels to compare variance of each channel
    :param L1: shape (n_trials, n_channels)
    '''

    plt.figure(figsize=(8, 5))

    x0 = np.arange(L1.shape[1])                     # x axis = number of channels
    x1 = np.arange(L1.shape[1]) + 0.4

    y0 = np.mean(L1, axis=0)                        # y axis = log variance averaged over trials
    y1 = np.mean(L2, axis=0)

    plt.bar(x0, y0, width=0.4, color='red', label='right')
    plt.bar(x1, y1, width=0.4, color='blue', label='left')

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each CSP component')
    plt.xlabel('CSP components')
    plt.ylabel('log-var')
    plt.legend()

    plt.show()

    return

def scatter_logvar(L1, L2):
    '''
    Plot logvar scatter data of channels of interest OR most disriminative eigenvectors (1 point = 1 trial)
    :params L1,L2: shape (n_trials, n_channels)
    '''

    plt.figure(figsize=(8, 5))

    plt.scatter(L1[:, 0], L1[:, -1], color='red', linewidth=1, label='right')
    plt.scatter(L2[:, 0], L2[:, -1], color='blue', linewidth=1, label='left')
    plt.legend()

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
    # W = np.column_stack((W[:, :1], W[:, -1:]))

    # project spatial filters onto data
    X1_csp = np.stack([np.dot(W.T, trial) for trial in X1])
    X2_csp = np.stack([np.dot(W.T, trial) for trial in X2])

    return X1_csp, X2_csp, W

def main():
    '''
    Main function for extracting and processing data for input into model
    :return: 2D matrix of shape (n_trials, n_CSP_components)
    '''

    def interpolate(stream):
        '''
        Custom function for interpolating Crown time output to get continuous timestamps
        Required because each datapacket of 16 points all share the same timestamp
        :param output: 2D array of whole session
        :return: edited 2D array with interpolated times
        '''

        # timestamps in the first column
        times = stream[:,0]

        # window in segments of 16+1
        step = 16
        n_windows = int(len(times) / step)
        new_times = []

        # interpolate between timestamps of datapackets
        for i in range(n_windows):
            start_idx = i * step
            end_idx = start_idx + step
            start = times[start_idx]

            if i == n_windows - 1:          # if it's the last window
                end = start + 1000/16       # 16 packets = 1000ms
            else:
                end = times[end_idx]

            correction = end - (end - start) * 1 / step     # correct to end of packet i (rather than beginning of packet i+1)
            new_times.extend(np.linspace(start, correction, step))

        stream[:, 0] = new_times

        return stream

    def get_data(session):
        '''
        Extracts, reformats and preprocesses raw signal data from JSON file
        Contains list of dictionaries of length n_iterations (1 iteration = 16 datapoints for each of 8 channels)
        :param: session: directory of eeg_stream json file
        :return: 2D array of shape (n_samples, (time channel + n_EEG_channels))
        '''

        with open(session, 'r') as f:
            data = json.load(f)

            stream = []                 # array for individual streams to be appended to. of shape (n_channels, n_total_samples)

            # pull individual dictionaries (datapackets)
            for packet in data:
                # channels = packet['info']['channelNames']
                signals = np.array(packet['data'])
                time = packet['info']['startTime'] / 1000      # convert to seconds to match label data

                # append each row to stream
                for i in range(signals.shape[1]):
                    row = [time] + signals[:, i].tolist()
                    stream.append(row)

            # interpolate time values (smooth increments)
            stream = np.array(stream)
            stream = interpolate(stream)

            return stream

    def assign_trials(stream, onsets, window_length):
        '''
        Slice up and append trials using timestamps from label data
        :params:
            stream: 2D array of session
            onsets: list of class onset timestamps
            window_length: time in seconds of the period to take after the onset
            full_window: flag to choose whether to take full 4s period after onset or 0.5-2.5s period post-onset
        :return: 3D tensor of shape (n_trials, n_channels, n_samples)
        '''

        window = window_length * 256
        times = stream[:,0]
        time_idx = []                               # list of signal times that lie within a trial window

        # iterate through onset times
        for trial in onsets:
            start_idx = np.where(times >= trial)[0][0]                        # get index of start of trial in signal
            end_idx = min(start_idx + window, len(times))                     # end index of trial in signal
            time_idx.append(np.arange(start_idx,end_idx))                     # range of trial indices

        # create training tensor
        trials = stream[:,1:]
        tensor = np.stack([trials[idx,:] for idx in time_idx])

        # output tensor is shape (n_trials, n_channels, n_samples)
        return np.transpose(tensor, (0, 2, 1))

    ##########################################
    # create list of session tensors
    X1 = None                     # class 1
    X2 = None                     # class 2

    # pull saved files from distinct folders each containing a single session
    root = "training_data"

    for folder in os.listdir(root):
        if folder == ".DS_Store":
            continue  # Skip .DS_Store files

        folder_path = os.path.join(root, folder)

        for session_path in os.listdir(folder_path):
            session = os.path.join(folder_path, session_path)

            # pull continuous signal data
            if 'eeg_stream' in session:
                # extract list of dictionaries (datapackets) into one array
                stream = get_data(session)

                # print(f'Single class data shape: {stream.shape}')

            # pull trial labels
            if 'labels' in session:
                with open(session, 'r') as f:
                    data = json.load(f)

                    # pair onsets with class label
                    labels = [(i[0], i[-1]) for i in data]

                # get class onsets
                onsets_1 = [(i[0]) for i in labels if i[-1] == -1]      # right data
                onsets_2 = [(i[0]) for i in labels if i[-1] == 1]       # left data

        # print(f'Number of class 1 trials: {len(onsets_1)}')
        # print(f'Number of class 2 trials: {len(onsets_2)}')

        # match up class onsets with signal array
        X1_session = assign_trials(stream, onsets_1, window_length=2)                                # X1 = right
        X2_session = assign_trials(stream, onsets_2, window_length=2)                                # X2 = left

        # Concatenate the session data
        if X1 is None:
            X1 = X1_session
        else:
            X1 = np.concatenate((X1, X1_session), axis=0)

        if X2 is None:
            X2 = X2_session
        else:
            X2 = np.concatenate((X2, X2_session), axis=0)

    print(f'Each class tensor shape: {X1.shape}')

    # X1 = normalise(X1)
    # X2 = normalise(X2)

    # raw power plot
    freqs_raw, P1_raw = compute_psd(X1)
    _, P2_raw = compute_psd(X2)
    plot_psd(freqs_raw, P1_raw, P2_raw)


    # bandpass filter & normalise
    # X1_filt = normalise(bpass_filter(X1, 8, 15, 256))
    # X2_filt = normalise(bpass_filter(X2, 8, 15, 256))
    #
    # # pass data through spatial filters using CSP
    # X1_csp, X2_csp, W = spatial_filter(X1=X1_filt, X2=X2_filt)
    #
    # # get Power Spectral Densities from spatially filtered data
    # freqs, P1 = compute_psd(X1_csp)
    # _, P2 = compute_psd(X2_csp)
    #
    # # get Log Variance
    # L1 = logvar(X1_csp)
    # L2 = logvar(X2_csp)
    #
    # ## plots
    # plot_psd(freqs, P1, P2)
    # bar_logvar(L1,L2)
    # scatter_logvar(L1,L2)


    print(f'Each class model input shape: {L1.shape}')

    # return log variance and spatial filters as input into models
    return L1, L2, W


if __name__ == '__main__':
    main()





