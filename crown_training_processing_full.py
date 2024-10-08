'''
version 2 - Full feature breakdown. Uses crown_full_feature_extraction file
Process saved json data for ML training
8 channels in order: CP3, C3, F5, PO3, PO4, F6, C4, CP4
'''

import json
import scipy.io
import numpy as np
import scipy.linalg
import os
import pandas as pd
import matplotlib.pyplot as plt
import crown_artifacts
from scipy import interpolate
import crown_full_feature_extraction

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

    def assign_trials(stream, onsets, window_length, delay):
        '''
        Slice up and append trials using timestamps from label data
        :params:
            stream: 2D array of session
            onsets: list of class onset timestamps
            window_length: time in seconds of the period to take after the onset
            delay: time in seconds from which to start window after the onset
        :return: 3D tensor of shape (n_trials, n_channels, n_samples)
        '''

        window = int(window_length * 256)
        delay = int(delay * 256)
        times = stream[:,0]
        time_idx = []                               # list of signal times that lie within a trial window

        # iterate through onset times
        for trial in onsets:
            start_idx = np.where(times >= trial)[0][0] + delay                # get index of start of trial in signal
            end_idx = min(start_idx + window, len(times))                     # end index of trial in signal
            time_idx.append(np.arange(start_idx,end_idx))                     # range of trial indices

        # create training tensor
        trials = stream[:,1:]
        tensor = np.stack([trials[idx,:] for idx in time_idx])

        # output tensor is shape (n_trials, n_channels, n_samples)
        return np.transpose(tensor, (0, 2, 1))

    def artifacts(X, method='discard'):
        '''
        2 ways of dealing with artifacts (in the form of big spikes in the raw signal data) -
        discard the whole signal or correct for them
        :params: X = 3D tensor of shape (n_trials, n_channels, n_samples)
        :return: 3D tensor of shape (n_trials, n_channels, n_samples)
        '''


        artifact_trials = []

        # get standard deviation of each trial and mean stdev over all trials in each channel
        for channel in range(X.shape[1]):
            trial_stds = np.std(X[:, channel, :], axis=1)
            mean_std = np.mean(trial_stds)

            print(f'channel: {channel}; mean std: {mean_std}; trial stds: {trial_stds}')

            for i, std in enumerate(trial_stds):
                # set criteria for detecting artifacts
                if (std > 2 * mean_std) and (std > 100):
                    artifact_trials.append(i)           # save index of trial
                    print(f'Trials removed: {artifact_trials}')

        if method=='discard':
            # remove all trials with artifacts
            X = X[[i for i in range(X.shape[0]) if i not in artifact_trials]]

        if method=='correct':
            # incomplete option
            crown_artifacts.main()

        return X

    ##########################################
    # create list of session tensors
    X1 = None                     # class 1
    X2 = None                     # class 2

    # pull saved files from distinct folders each containing a single session
    root = "./training_data"

    for folder in os.listdir(root):
        if folder == ".DS_Store":
            continue  # Skip .DS_Store files

        folder_path = os.path.join(root, folder)

        for session_path in os.listdir(folder_path):
            session = os.path.join(folder_path, session_path)
            print(session)

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

        # match up class onsets with signal array

        window_length = 3.5         # how many seconds of data to take (must be sub 4 seconds)
        delay = 0                   # how many seconds after prompt to start taking data from

        X1_session = assign_trials(stream, onsets_1, window_length=window_length, delay=delay)               # X1 = right
        X2_session = assign_trials(stream, onsets_2, window_length=window_length, delay=delay)               # X2 = left

        ## handle artifacts
        X1_session = artifacts(X1_session, method='discard')
        X2_session = artifacts(X2_session, method='discard')

        # Concatenate the session data
        if X1 is None:
            X1 = X1_session
        else:
            X1 = np.concatenate((X1, X1_session), axis=0)

        if X2 is None:
            X2 = X2_session
        else:
            X2 = np.concatenate((X2, X2_session), axis=0)

    ### now have signals labelled by class (left vs right)

    ## send to crown_full_feature_extraction for filters & feature extraction
    crown_full_feature_extraction.main(X1, X2)


if __name__ == '__main__':
    main()






