
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

    return filtfilt(b, a, data)

# def notch_filter(data, )

def normalise(signal):
    # Z-score normalisation
    return (signal - np.mean(signal)) / np.std(signal)

def notch_filter(data, fs, f0, quality_factor):
    b, a = scipy.signal.iirnotch(f0, quality_factor, fs)

    return filtfilt(b, a, data)


def compute_psd(tensor):

    freqs, PSD = scipy.signal.welch(tensor, fs=265, axis=0, nperseg=tensor.shape[0])

    return np.array(freqs), np.array(PSD)

def plot_frequency_domain(freqs, P1):

    channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # plot frequency domain
    plt.plot(freqs, P1, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()

def main(X1, X2):
    '''
    X1, X2 shape = (no. trials, no. channels, no. samples = no. seconds * fs)
    '''

    print(f'Class 1 trials: {X1.shape[0]}')
    print(f'Class 2 trials: {X2.shape[0]}')

    X1 = X1[0][1]

    plt.plot(X1)
    plt.show()


    # apply notch filter for power noise
    fs = 256
    f0 = 50                         # frequency to be removed
    quality_factor = 30

    X1 = notch_filter(X1, fs, f0, quality_factor)

    # X1 = bpass_filter(X1, 10, 30, fs=256, order=5)

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

