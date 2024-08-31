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
    :returns: dict of format {[band][trial][channel](imfs)}
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
    Calculate normalised power of first two IMFs for C3 and C4 channels across all trials
    :param X: Original data array
    :param dict: Dictionary of format {[band][trials][channels](n_samples, n_imfs)}
    :param band: The frequency band to analyze
    :returns: Dictionary of format {channel: [imf1_power, imf2_power]}
    '''

    channels = [1, 6]  # C3, C4
    total_power = {ch: np.zeros(2) for ch in channels}  # Initialize for 2 IMFs

    # calculate total power across all trials for each channel
    for _, trial_data in dict[band].items():
        for channel in channels:
            if channel in trial_data:
                imf = trial_data[channel]
                power = np.mean(np.square(imf[:, :2]), axis=0)
                total_power[channel] += power

    # normalise power across all trials for each IMF
    normalised_power = {ch: np.zeros(2) for ch in channels}
    for imf in range(2):
        total_imf_power = sum(total_power[ch][imf] for ch in channels)
        for ch in channels:
            normalised_power[ch][imf] = total_power[ch][imf] / total_imf_power

    return normalised_power, channels

def prepare_heatmap_data(power_dict, channels):
    '''
    Prepare data for heatmap from power dictionary
    :param power_dict: Dictionary containing average power
    :param band: Band of interest
    :returns: DataFrame suitable for heatmap
    '''

    def plot_heatmap():
        '''
        Plot simple 2x2 grid for the power data
        :param data: DataFrame containing power values
        '''

        plt.figure(figsize=(8, 4))
        sns.heatmap(data.T, annot=True, cmap='YlGnBu', cbar=True, fmt=".2f",
                    vmin=0, vmax=1, cbar_kws={'label': 'Normalized Power'})
        plt.title('Normalized Power of First 2 IMFs')
        plt.xlabel('Channel')
        plt.ylabel('IMF')
        plt.tight_layout()
        plt.show()

    # Extract power values for 2 channels
    c3_power = power_dict[channels[0]]      # C3
    c4_power = power_dict[channels[-1]]     # C4

    # Create a DataFrame
    data = pd.DataFrame({
        'IMF 1': [c3_power[0], c4_power[0]],
        'IMF 2': [c3_power[1], c4_power[1]],
    }, index=['C3', 'C4'])

    plot_heatmap()

    return

def main(X, dict, fs):

    # band = f'8.0-16.0Hz'
    # band = f'16.0-32.0Hz'
    band = f'32.0-64.0Hz'

    # get imfs
    imf_dict = emd_sift(X, dict, fs)
    # get power for each imf
    power_dict, channels = imf_power(X, imf_dict, band)
    # plot power distribution
    prepare_heatmap_data(power_dict, channels)

    return


if __name__ == '__main__':
    main()
