
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import scipy.io



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

    freqs, PSD = scipy.signal.welch(tensor, fs=265, axis=0, nperseg=tensor.shape[0])

    return np.array(freqs), np.array(PSD)

def plot_frequency_domain(freqs, P1):

    channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # # Create subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # plot frequency domain
    plt.plot(freqs, P1, color='black', linewidth=1)
    # plt.set_ylim(0, 1)
    # plt.set_xlim(0, 30)

    plt.tight_layout()
    plt.show()

def main(X1, X2):
    '''
    X1, X2 shape = (no. trials, no. channels, no. samples = no. seconds * fs)
    '''

    print(f'Class 1 trials: {X1.shape[0]}')
    print(f'Class 2 trials: {X2.shape[0]}')

    X1 = X1[0][0]

    # break out frequency components
    freqs_raw, P1 = compute_psd(X1)
    print(P1.shape)
    plot_frequency_domain(freqs_raw, P1)



    # bandpass filter & normalise
    # X1_filt = normalise(bpass_filter(X1, 8, 15, 256))
    # X2_filt = normalise(bpass_filter(X2, 8, 15, 256))


    return


if __name__ == '__main__':
    main()

