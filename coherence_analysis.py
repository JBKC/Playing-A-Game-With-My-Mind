'''
Exploring connectivity / coherence between EEG channels during motor imagery
8 channels in order: 0:CP3, 1:C3, 2:F5, 3:PO3, 4:PO4, 5:F6, 6:C4, 7:CP4
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import stats
import mne
from mne.channels import make_standard_montage


def plot_matrix(matrix, channels):
    plt.figure(figsize=(6, 6))  # Set the figure size
    plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')  # Plot the array
    plt.xticks(range(len(channels)), channels, rotation=45, ha='right')
    plt.yticks(range(len(channels)), channels)
    plt.colorbar()  # Add a color bar to indicate the intensity scale
    plt.show()

def phase_lag_index(tensor, channels):
    '''
    Calculates the PLI over trials
    '''

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

    plot_matrix(pli_matrix, channels)
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
    n_trials, n_channels, n_samples = tensor.shape
    laplacian_tensor = np.zeros_like(tensor)

    # create mne info object
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')
    montage = make_standard_montage('standard_1020')
    info.set_montage(montage)

    for trial in range(n_trials):

        raw = mne.io.RawArray(tensor[trial], info)

        # Apply Surface Laplacian
        laplacian = mne.preprocessing.compute_current_source_density(raw)

        # Extract data and store in the new tensor
        laplacian_tensor[trial] = laplacian.get_data()

    return laplacian_tensor

def common_av_reference(tensor):

    # average over all channels for each trial
    avgs = np.mean(tensor, axis=1, keepdims=True)

    # subtract from original tensor
    return tensor - avgs


def topology_viz(tensor_a, tensor_b, channels, fs, time_divisions=5):
    '''
    Compare trial-averaged topological EEG maps over time for two datasets
    '''
    n_trials, n_channels, n_samples = tensor_a.shape

    # Create MNE info object
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types=['eeg'] * n_channels)

    # Define spatial layout of electrodes
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Process both tensors
    evokeds = []
    for tensor in [tensor_a, tensor_b]:
        avg_data = np.mean(tensor, axis=0)
        raw = mne.io.RawArray(avg_data, info)
        events = np.array([[0, 0, 1]])
        epoch = mne.Epochs(raw, events, tmin=0, tmax=raw.times[-1], baseline=None)
        evokeds.append(epoch.average())

    # plot comparison
    times = np.linspace(0, evokeds[0].times[-1], time_divisions)
    fig, axes = plt.subplots(2, time_divisions + 1, figsize=(18, 6))

    for tensor, evoked in enumerate(evokeds):
        for idx, time in enumerate(times):
            evoked.plot_topomap(times=time, axes=axes[tensor, idx], show=False,
                                extrapolate='head', time_unit='s', colorbar=False, show_names=True)

    # Add colorbars
    for row in range(2):
        cbar_ax = axes[row, -1]
        fig.colorbar(axes[row, 0].images[0], cax=cbar_ax)

    plt.tight_layout()
    plt.show()

    # # Create an animated topomap
    # anim = evoked.animate_topomap(frame_rate=5, time_unit='s', extrapolate='head')
    # plt.show()

def ae_correlation(tensor, channels):
    '''
    amplitude envelope correlation
    '''

    n_trials, n_channels, n_samples = tensor.shape

    # get instantaneous amplitude
    analytic = scipy.signal.hilbert(tensor, axis=-1)
    ae = np.abs(analytic)
    # average across trials
    ae = np.mean(ae, axis=0)            # shape (n_channels, n_samples)
    print(ae.shape)

    ae_matrix = np.zeros((n_channels, n_channels))

    # cycle through channel combinations
    for ch1 in range(n_channels):
        for ch2 in range(n_channels):
            ae_matrix[ch1,ch2], _ = stats.spearmanr(ae[ch1], ae[ch2])

    np.fill_diagonal(ae_matrix, 0)

    plot_matrix(ae_matrix, channels)



def main(dict, band, fs):
    '''
    Takes in dict where dict[band] has shape (n_trials, n_channels, n_samples)
    '''

    channels = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # car_tensor = common_av_reference(dict[band])
    # laplacian_tensor = surface_laplacian(dict[band], channels, fs)
    # topology_viz(dict[band], laplacian_tensor, channels, fs)

    # phase_lag_index(dict[band], channels)
    ae_correlation(dict[band], channels)

if __name__ == '__main__':
    main()