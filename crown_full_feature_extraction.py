
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import scipy.io
import pywt
import coherence_analysis
import mne
from mne.preprocessing import compute_current_source_density
import torch
from kymatio.torch import Scattering1D


def normalise(signal):
    # Z-score normalisation
    # return (signal - np.mean(signal)) / np.std(signal)
    return signal - np.mean(signal)

def compute_psd(dict, band):

    signal = dict[band][0,0,:]          # pull out a single signal (test visualisation)
    # print(signal.shape)

    # use windowing to smooth FFT response
    freqs, PSD = scipy.signal.welch(signal, fs=265, axis=0, nperseg=signal.shape[-1], window='hann')

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

def wavelet_transform(X, fs, wavelet='db4'):
    '''
    Decompose signals into EEG frequency bands using DWT
    '''

    n_trials, n_channels, n_samples = X.shape

    dwt_coeffs = []

    # Calculate the maximum decomposition level if not provided
    level = pywt.dwt_max_level(X.shape[-1], wavelet)

    # Perform the DWT
    for trial in range(n_trials):
        for channel in range(n_channels):
            coeffs = pywt.wavedec(X[trial, channel, :], wavelet, level=level)
            dwt_coeffs.append(coeffs)

    print(dwt_coeffs)
    print(level)

    return dwt_coeffs

    

def main(X1, X2):
    '''
    X1 = signals for right side motor imagery
    X2 = signals for left side motor imagery
    X1, X2 shape = (no. trials, no. channels, no. samples = no. seconds * fs)
    3rd dimension is the individual signal data
    '''

    def fir_method(X, numtaps=101):
        '''
        initial attempt - use FIR filter to split signals into frequency bands
        :params: input tensor X for a single class, of shape (n_trials, n_channels, n_samples)
        :returns:
        '''

        # create dictionary of filtered tensor
        dict = {}

        for k, v in bands.items():
            # normalise cutoffs to nyquist frequencies
            cutoffs = [2 * v[0] / fs, 2 * v[1] / fs]         # normalise to nyquist
            # get FIR filter coefficients using window method
            coeffs = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoffs, pass_zero=False, window='hann')

            X_fir = np.zeros_like(X)

            # apply filter to X
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    # normalise signal
                    X[i, j, :] = normalise(X[i, j, :])
                    # apply filter
                    X_fir[i, j, :] = scipy.signal.lfilter(coeffs, 1.0, X[i, j, :])


            # add to dictionary of tensors
            dict[k] = X_fir

            # plot_freq_response(coeffs, fs)
            # plot_phase_response(coeffs, fs)

        return dict

    print(f'Class 1 trials: {X1.shape[0]}')
    print(f'Class 2 trials: {X2.shape[0]}')

    fs = 256

    bands = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 50],
    }

    ### methods for extracting EEG frequency bands
    wavelet_transform(X=X1, fs=fs, wavelet='db4')

    # dict_X1 = fir_method(X1)
    # dict_X2 = fir_method(X2)

    focus_band = 'gamma'

    ##### calculate coherence between different channels for each class
    # X1_coh = coherence_analysis.main(dict=dict_X1, band=focus_band, fs=fs)
    # X2_coh = coherence_analysis.main(dict=dict_X2, band=focus_band, fs=fs)

    # PSD
    # freqs_raw, P1 = compute_psd(dict_X1, focus_band)
    # plot_psd(freqs_raw, P1)

    # FFT
    # freqs, fft = compute_fft(dict_X1, focus_band, fs)
    # plot_fft(freqs, fft)


    return


if __name__ == '__main__':
    main()

