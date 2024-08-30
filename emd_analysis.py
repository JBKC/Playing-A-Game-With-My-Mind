import numpy as np
import matplotlib.pyplot as plt
import emd
import scipy.signal

def plot_imfs(X, imf_dict, fs):
    '''
    :params: imf_dict of format [band][trials][channels](samples, imfs)
    '''

    # take arbitrary signal
    band = f'16.0-32.0Hz'
    trial = 12
    channel = 2

    imf = imf_dict[band][trial][channel]

    n_imfs = 3
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
    max_imfs = 5

    imf_dict = {}

    for k, X in dict.items():
        imf_dict[k] = {}                # sub-dictionary

        for trial in range(X.shape[0]):
            imf_dict[k][trial] = {}

            for channel in range(X.shape[1]):
                imf = emd.sift.sift(X[trial,channel,:], imf_opts={'sd_thresh': sd_thresh})
                imf_dict[k][trial][channel] = imf

    plot_imfs(X, imf_dict, fs)


def main(X, dict, fs):

    emd_sift(X, dict, fs)

    return


if __name__ == '__main__':
    main()
