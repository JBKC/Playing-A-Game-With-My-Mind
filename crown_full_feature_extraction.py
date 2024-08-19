
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import scipy.io


def normalise(signal):
    # Z-score normalisation
    return (signal - np.mean(signal)) / np.std(signal)

def compute_psd(tensor):

    freqs, PSD = scipy.signal.welch(tensor, fs=265, axis=0, nperseg=tensor.shape[0])

    return np.array(freqs), np.array(PSD)

def plot_freq_response(coeffs,fs):

    # plot the filter response (frequency vs amplitude)
    w, h = scipy.signal.freqz(coeffs, worN=8000)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.title('Band-pass FIR Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

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

    X1 = X1[0][1]           # take example signal

    plt.plot(X1)
    plt.show()

    # apply notch FIR filter
    fs = 256
    low = 50
    high = 60
    cutoffs = [2*low / fs, 2*high / fs]          # normalise to nyquist
    numtaps = 11

    # get coefficients
    coeffs = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoffs, window='hamming')
    X1_fir = scipy.signal.lfilter(coeffs, 1.0, X1)

    plot_freq_response(coeffs, fs)

    freqs_raw, P1 = compute_psd(X1_fir)
    print(P1.shape)
    plot_frequency_domain(freqs_raw, P1)



    # bandpass filter & normalise
    # X1_filt = normalise(bpass_filter(X1, 8, 15, 256))
    # X2_filt = normalise(bpass_filter(X2, 8, 15, 256))


    return


if __name__ == '__main__':
    main()

