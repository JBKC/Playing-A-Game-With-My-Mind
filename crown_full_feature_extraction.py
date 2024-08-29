
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import scipy.io
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

def wavelet_scattering(X, bands, fs):
    '''
    Perform continuous wavelet transform to filter signals into frequency bands
    '''

    # dictionary of scattering coefficients for each band
    band_coeffs = {band: [] for band in bands}

    n_samples, n_channels, n_samples = X.shape

    # find max and min frequency in bands
    max_freq = np.max([v for v in bands.values()])
    min_freq = np.min([v for v in bands.values()])

    # set number of decomposition octaves (J) and frequency resolution (Q)
    J = int(np.log2(fs / min_freq))
    Q = 8

    scattering = Scattering1D(J, n_samples, Q)      # create scattering object

    # apply scattering to each channel
    for channel in range(n_channels):
        coeffs = scattering(X[:, channel, :].unsqueeze(1))

        # Get the center frequencies of the wavelets
        freqs = scattering.meta()['xi']
        freqs_hz = freqs * fs / (2 * np.pi)

        # Extract coefficients for each band
        for band, freq_range in bands.items():
            band_mask = (freqs_hz >= freq_range[0]) & (freqs_hz <= freq_range[1])
            band_coeffs = coeffs[:, band_mask, :]
            band_coeffs[band].append(band_coeffs)

    # Convert lists to tensors
    for band in bands:
        band_coeffs[band] = torch.cat(band_coeffs[band], dim=1)

    return band_coeffs


    return

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
    wavelet_scattering(X=X1, bands=bands, fs=fs)

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

