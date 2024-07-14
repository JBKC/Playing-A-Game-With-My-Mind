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
    # can crop to get rid of edge effects (to update with more elegant method)
    # 100 samples = 490ms
    return signal

def compute_psd(tensor):
    '''
    :param takes in 3D tensor of shape (n_trials, n_channels, n_samples)
    :return:
    power spectral density of each shape (n_trials, n_channels, n_samples/2 + 1)
    freqs of shape ceil(n_samples+1/2))
    '''

    freqs, PSD = scipy.signal.welch(tensor, fs=265, axis=2, nperseg=tensor.shape[2])
    print(PSD.shape)

    return np.array(freqs), np.array(PSD)

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
    ax1.set_title(f'PSD for {channel_names[1]} (controls right side)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_xlim(0, 30)
    ax1.set_ylim(1, 20)
    ax1.legend()

    ax2.plot(freqs, P1[:, -1, :].mean(axis=0), color='red', linewidth=1, label='right')
    ax2.plot(freqs, P2[:, -1, :].mean(axis=0), color='blue', linewidth=1, label='left')
    ax2.set_title(f'PSD for {channel_names[6]} (controls left side)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim(0, 30)
    ax2.set_ylim(1, 20)
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
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
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


def spatial_filter(X1, X2):
    """
    Compute CSP for two classes of EEG data.

    X1, X2: Shape = (n_trials, n_channels, n_samples)
    """

    def whitening_matrix(sigma):
        # eigendecomposition of composite covariance matrix
        U, D, _ = np.linalg.svd(sigma)

        # W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(D + 1e-5)), U.T))               # ZCA

        # W output using PCA method
        return np.dot(np.diag(D ** -0.5), U.T)

    # calculate covariance matrices
    R1 = np.mean([np.dot(X, X.T) for X in X1], axis=0)
    R2 = np.mean([np.dot(X, X.T) for X in X2], axis=0)

    print(f'Cov shape: {R1.shape}')                                 # shape (n_channels, n_channels)

    # get whitening matrix P from composite covariance matrix
    P = whitening_matrix(R1 + R2)
    print(f'Whitening matrix shape: {P.shape}')                     # shape (n_channels, n_channels)

    # whiten individual covariance matrices
    S1 = np.dot(np.dot(P, R1), P.T)
    S2 = np.dot(np.dot(P, R2), P.T)

    # print(S1+S2)
    # == identity matrix

    # D1, V1, _ = np.linalg.svd(S1)
    # D2, V2, _ = np.linalg.svd(S2)

    # solve generalised eigenvalue problem to get spatial filters
    d, W = la.eigh(S1, S1+S2)

    # sort eigenvalues in descending order
    # idx = np.argsort(d)[::-1]
    # d = d[idx]
    # W = W[idx, :]

    # keep first and last eigenvectors
    # W = np.vstack([W[0, :], W[-1, :]])
    print(f'Spatial filter: {W}')

    # eigenvectors == spatial filters == projection matrix
    print(f'Discriminative eigenvalues {d}')
    print(f'Spatial filter shape: {W.shape}')

    # project data onto spatial filters
    X1_csp = np.stack([np.dot(W, trial) for trial in X1])
    X2_csp = np.stack([np.dot(W, trial) for trial in X2])

    print(f'Data after spatial filtering shape: {X1_csp.shape}')

    return X1_csp, X2_csp

def main():

    def interpolate(output):
        '''
        Custom function for interpolating Crown time output to get continuous timestamps
        Required because each datapacket of 16 points all share the same timestamp
        :param output: DataFrame of session
        :return: edited DataFrame with interpolated times
        '''

        times = output['time']
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

        output['time'] = new_times
        # output.to_json('output.json', orient='records', lines=True)

        return output

    def get_dataframe():
        '''
        Extracts, processes and reformats raw signal data from JSON file
        Contains list of dictionaries of length n_iterations (1 iteration = 16 datapoints for each of 8 channels)
        :return: pandas Dataframe of shape (n_samples, n_channels)
        '''

        with open(os.path.join(folder, session), 'r') as f:
            data = json.load(f)

            df = pd.DataFrame()         # output dataframe for holding entire session's data (all trials)
            stream = {}                 # individual streams within output dataframe

            # pull individual dictionaries (datapackets)
            for packet in data:
                channels = packet['info']['channelNames']
                signals = packet['data']
                stream['time'] = packet['info']['startTime'] / 1000      # convert to seconds to match label data

                # reformat channel data
                for i, channel in enumerate(channels):
                    stream[channel] = signals[i]

                # add stream into output dataframe
                stream = pd.DataFrame(stream)
                df = pd.concat([df, stream], ignore_index=True)

            # interpolate time values (smooth increments)
            df = interpolate(df)

            # apply filtering + normalisation to channel signals
            for channel in channels:
                # df[channel] = normalise(df[channel].values)     # no bandpass (for plotting)
                df[channel] = normalise(bpass_filter(df[channel].values, lowcut=7, highcut=20, fs=256, order=5))

            return df

    def assign_trials(df, onsets):
        '''
        Slice up and append trials using timestamps from label data
        :param df: DataFrame of session; onsets: list of class onset timestamps
        :return: tensor of shape (n_trials, n_channels, n_samples)
        '''

        window = 1024                               # prompt length (4 seconds) * fs (256 Hz)
        signal_times = np.array(df['time'])
        data = []                                   # list of dataframes

        # iterate through onset times
        for trial in onsets:
            start_idx = np.where(signal_times >= trial)[0][0]           # get index of start of trial in signal
            end_idx = min(start_idx + window, len(signal_times))        # end index of trial
            idx = np.arange(start_idx,end_idx)
            data.append(df.iloc[idx])

        # create training tensor
        tensor = np.stack([df.drop(columns=['time']).values for df in data])

        # output tensor is shape (n_trials, n_channels, n_samples)
        return np.transpose(tensor, (0, 2, 1))

    ##########################################
    # pull saved files
    folder = "test data"
    for session in os.listdir(folder):

        # pull continuous signal data
        if 'eeg_stream' in session:
            # extract list of dictionaries (datapackets) into one dataframe
            df = get_dataframe()

            print(f'All data shape: {df.shape}')

        # pull trial labels
        if 'labels' in session:
            with open(os.path.join(folder, session), 'r') as f:
                data = json.load(f)
                # pair onsets with class label
                labels = [(i[0], i[-1]) for i in data]

            # get class onsets
            onsets_1 = [(i[0]) for i in labels if i[-1] == -1]      # right data
            onsets_2 = [(i[0]) for i in labels if i[-1] == 1]       # left data

    print(f'Number of class 1 trials: {len(onsets_1)}')
    print(f'Number of class 2 trials: {len(onsets_2)}')

    # match up class onsets with signal data in DataFrame
    X1 = assign_trials(df, onsets_1)
    X2 = assign_trials(df, onsets_2)

    print(f'Input data shape: {X1.shape}')

    # pass data through spatial filters using CSP
    X1, X2 = spatial_filter(X1=X1, X2=X2)

    # get Power Spectral Densities
    freqs, P1 = compute_psd(X1)
    _, P2 = compute_psd(X2)
    plot_psd(freqs, P1, P2)

    # get Log Variance
    L1 = logvar(P1)
    L2 = logvar(P2)
    bar_logvar(L1,L2)
    scatter_logvar(L1,L2)

    return


if __name__ == '__main__':
    main()





