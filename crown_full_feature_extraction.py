
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import scipy.io


def normalise(signal):
    # Z-score normalisation
    # return (signal - np.mean(signal)) / np.std(signal)
    return signal - np.mean(signal)

def compute_psd(dict, band):

    signal = dict[band][0,0,:]          # pull out a single signal
    # print(signal.shape)

    freqs, PSD = scipy.signal.welch(signal, fs=265, axis=0, nperseg=signal.shape[-1], window='blackmanharris')

    return np.array(freqs), np.array(PSD)

def compute_fft(dict, band, fs):

    signal = dict[band][0,0,:]          # pull out a single signal
    N = len(signal)

    result = scipy.fft.fft(signal)[:N//2]
    freqs = scipy.fft.fftfreq(N, 1/fs)[:N//2]

    return freqs, result


def plot_freq_response(coeffs,fs):

    # plot the filter response (frequency vs amplitude)
    w, h = scipy.signal.freqz(coeffs, worN=8000)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.title('Band-pass FIR Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def plot_phase_response(coeffs, fs):

    # plot phase response
    w, h = scipy.signal.freqz(coeffs, worN=8000)
    plt.plot(0.5 * fs * w / np.pi, np.angle(h), 'r')
    plt.title('Filter Phase Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_psd(freqs, P1):

    channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # plot frequency domain
    plt.plot(freqs, P1, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()


def plot_fft(freqs, fft):

    plt.plot(freqs, fft, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()

def main(X1, X2):
    '''
    X1 = signals for right side motor imagery
    X2 = signals for left side motor imagery
    X1, X2 shape = (no. trials, no. channels, no. samples = no. seconds * fs)
    3rd dimension is the individual signal data
    '''

    print(f'Class 1 trials: {X1.shape[0]}')
    print(f'Class 2 trials: {X2.shape[0]}')

    # apply FIR filters to split into frequency buckets
    fs = 256
    numtaps = 101

    bands = {
        'delta': [2, 10],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 50],
    }

    # create dictionary of filtered tensors for each class
    dict_X1 = {}
    dict_X2 = {}

    for k, v in bands.items():
        # normalise cutoffs to nyquist frequencies
        cutoffs = [2 * v[0] / fs, 2 * v[1] / fs]         # normalise to nyquist
        # get FIR filter coefficients
        coeffs = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoffs, pass_zero=False, window='blackmanharris')

        X1_fir = np.zeros_like(X1)
        X2_fir = np.zeros_like(X2)

        # apply filter to X1 and X2
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                # normalise signal
                X1[i, j, :] = normalise(X1[i, j, :])
                # apply filter
                X1_fir[i, j, :] = scipy.signal.lfilter(coeffs, 1.0, X1[i, j, :])

        for i in range(X2.shape[0]):
            for j in range(X2.shape[1]):
                X2[i, j, :] = normalise(X2[i, j, :])
                X2_fir[i, j, :] = scipy.signal.lfilter(coeffs, 1.0, X2[i, j, :])

        # add to dictionary of tensors
        dict_X1[k] = X1_fir
        dict_X2[k] = X2_fir

        # plot_freq_response(coeffs, fs)
        # plot_phase_response(coeffs, fs)

    focus_band = 'alpha'

    # FFT
    # freqs, fft = compute_fft(dict_X1, focus_band, fs)
    # plot_fft(freqs, fft)

    # PSD
    freqs_raw, P1 = compute_psd(dict_X1, focus_band)
    plot_psd(freqs_raw, P1)

    return


if __name__ == '__main__':
    main()

