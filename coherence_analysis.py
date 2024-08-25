'''
Exploring connectivity / coherence between EEG channels during motor imagery
8 channels in order: 0:CP3, 1:C3, 2:F5, 3:PO3, 4:PO4, 5:F6, 6:C4, 7:CP4
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from mne.channels import make_standard_montage


def phase_lag_index(tensor):
    '''
    Calculates the PLI over trials
    '''

    def plot_matrix():
        plt.figure(figsize=(6, 6))  # Set the figure size
        plt.imshow(pli_matrix, cmap='coolwarm', interpolation='nearest')  # Plot the array
        plt.colorbar()  # Add a color bar to indicate the intensity scale

    def plot_over_trials():

        ch2 = 5         # comparison channel
        plt.figure(figsize=(12, 8))

        for ch1 in range(n_channels):
            plt.plot(plis[ch1, ch2, :], label=f'{ch1 + 1}, {ch2 + 1}')

        plt.xlabel('Time')
        plt.ylabel('PLI')
        plt.title('Phase Lag Index (PLI) Over Time for All Channel Pairs')
        plt.legend(loc='upper right')
        plt.show()

    # calculate instantaneous phase angle using HT
    n_trials, n_channels, n_samples = tensor.shape
    # PLI arrays for each trial
    plis = np.zeros((n_channels, n_channels, n_samples))

    # get instantaneous phase
    analytic = scipy.signal.hilbert(tensor, axis=-1)
    phase = np.angle(analytic)

    # to vectorize using broadcasting
    for ch1 in range(n_channels):
        for ch2 in range(n_channels):
            # get differences at each timepoint for each trial - shape (n_trials, n_samples)
            diffs = phase[:, ch1, :] - phase[:, ch2, :]
            # calculate the PLI over trials - shape (n_samples,)
            pli = np.abs(np.mean(np.sign(np.imag(np.exp(1j * diffs))), axis=0))
            plis[ch1, ch2, :] = pli

            # additional - average PLI over time
            pli_matrix = np.mean(plis, axis=-1)

    plot_matrix()
    plot_over_trials()

def surface_laplacian(tensor, channels, fs):
    '''
    :params:
    tensor: shape (n_trials, n_channels, n_samples)
    channels: list of channel names
    fs: sampling frequency

    :return:
    laplacian_tensor: array of same shape as input,
    '''

    

    pass


def main(dict, band, fs):
    '''
    Takes in dict where dict[band] has shape (n_trials, n_channels, n_samples)
    '''

    channels = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    phase_lag_index(dict[band])
    surface_laplacian(dict[band], channels, fs)


if __name__ == '__main__':
    main()