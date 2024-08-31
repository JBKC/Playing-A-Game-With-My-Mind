'''
8 channels in order: CP3, C3, F5, PO3, PO4, F6, C4, CP4
'''

import numpy as np
import matplotlib.pyplot as plt
import emd
import seaborn as sns
import scipy.signal
import pandas as pd


def plot_imfs(X, imf_dict, fs):
    '''
    :params: imf_dict of format [band][trials][channels](samples, imfs)
    '''

    # take arbitrary signal
    band = f'16.0-32.0Hz'
    trial = 12
    channel = 2

    imf = imf_dict[band][trial][channel]

    n_imfs = 2
    fig, axs = plt.subplots(n_imfs + 1, 2, figsize=(10, 1.5 * (n_imfs + 1)))
    time = np.arange(X.shape[-1]) / fs

    # Plot original signal
    axs[0, 0].plot(time, X[trial, channel, :])
    axs[0, 0].set_title(f'Original Signal: {band}')

    # Plot original signal's PSD
    f, psd = scipy.signal.welch(X[trial, channel, :], fs, nperseg=X.shape[-1], window='hann')
    axs[0, 1].plot(f, psd)
    axs[0, 1].set_title('Power Spectral Density')
    axs[0, 1].set_ylabel('PSD')
    axs[0, 1].set_xlim(0, 64)

    for i in range(n_imfs):
        # Plot imfs
        axs[i+1, 0].plot(time, imf[:, i])
        axs[i+1, 0].set_title(f'IMF {i+1}')

        # Plot filtered signal's PSD
        f, psd = scipy.signal.welch(imf[:, i], fs, nperseg=1024)
        axs[i+1, 1].plot(f, psd)
        axs[i+1, 1].set_xlim(0, 64)

    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

def emd_sift(X, dict, fs):
    '''
    Use EMD to break signal into IMFs
    '''

    # parameters for emd.sift.sift()
    sd_thresh = 0.05
    max_imfs = 2

    imf_dict = {}

    for k, X in dict.items():
        imf_dict[k] = {}                # sub-dictionary

        for trial in range(X.shape[0]):
            imf_dict[k][trial] = {}

            for channel in range(X.shape[1]):
                imf = emd.sift.sift(X[trial,channel,:], imf_opts={'sd_thresh': sd_thresh})
                imf_dict[k][trial][channel] = imf[:, :max_imfs]

    # plot_imfs(X, imf_dict, fs)

    return imf_dict

def imf_power(X, dict, band):
    '''
    :params: dict of format [band][trials][channels](n_samples, n_imfs)
    :returns: dict of format [channel](n_imfs)
    '''

    n_trials, _, _ = X.shape
    channels = [1,7]            # C3, C4

    av_power = {ch: np.zeros(2) for ch in channels}

    for _, trial_data in dict[band].items():
        for channel in channels:
            if channel in trial_data:

                # extract imf data from channels of interest
                imf = trial_data[channel]
                # Compute power for each IMF
                power = np.mean(np.square(imf), axis=0)  # average power for each IMF
                # accumulate power over all trials
                av_power[channel] += power

    print(av_power[1].shape)

    # average power over trials
    # power = {ch: power / n_trials for ch, power in av_power.items()}

    # return dictionary
    return av_power


def prepare_heatmap_data(power_dict, band):
    '''
    Prepare data for heatmap from power dictionary.

    :param power_dict: Dictionary containing average power
    :param band: Band of interest
    :returns: DataFrame suitable for heatmap
    '''

    def plot_heatmap():
        '''
        Plot heatmap for the power data.

        :param data: DataFrame containing power values
        '''

        plt.figure(figsize=(8, 4))
        sns.heatmap(data, annot=True, cmap='YlGnBu', cbar=True, fmt=".2f")
        plt.title('Power of First 2 IMFs')
        plt.xlabel('IMF')
        plt.ylabel('Channel')
        plt.show()


    # Extract power values for C3 and C4
    c3_power = power_dict[1]  # C3
    c4_power = power_dict[7]  # C4

    # Create a DataFrame
    data = pd.DataFrame({
        'IMF 1': [c3_power[0], c4_power[0]],
        'IMF 2': [c3_power[1], c4_power[1]],
    }, index=['C3', 'C4'])

    plot_heatmap()

    return

def main(X, dict, fs):

    band = f'16.0-32.0Hz'

    imf_dict = emd_sift(X, dict, fs)
    power_dict = imf_power(X, imf_dict, band)
    prepare_heatmap_data(power_dict, band)

    return


if __name__ == '__main__':
    main()
