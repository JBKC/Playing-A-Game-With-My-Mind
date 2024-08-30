
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


def plot_wavelet(X, filt_dict, bands, fs):

    # plot arbitrary signal
    trial = 12
    channel = 2

    n_bands = len(bands)
    fig, axs = plt.subplots(n_bands + 1, 2, figsize=(10, (n_bands + 1)))

    time = np.arange(X.shape[2]) / fs

    # Plot original signal
    axs[0, 0].plot(time, X[trial, channel, :])
    axs[0, 0].set_title('Original Signal')
    axs[0, 0].set_ylabel('Amplitude')

    # Plot original signal's PSD
    f, psd = scipy.signal.welch(X[trial, channel, :], fs, nperseg=X.shape[-1], window='hann')
    axs[0, 1].plot(f, psd)
    axs[0, 1].set_title('Power Spectral Density')
    axs[0, 1].set_ylabel('PSD')
    axs[0, 1].set_xlim(0, 50)

    for i, (band_name, band_data) in enumerate(filt_dict.items(), start=1):
        # Plot filtered signal
        axs[i, 0].plot(time, band_data[trial, channel, :])
        axs[i, 0].set_title(f'{band_name.capitalize()} Band ({bands[band_name][0]}-{bands[band_name][1]} Hz)')
        axs[i, 0].set_ylabel('Amplitude')

        # Plot filtered signal's PSD
        f, psd = scipy.signal.welch(band_data[trial, channel, :], fs, nperseg=1024)
        axs[i, 1].plot(f, psd)
        axs[i, 1].set_ylabel('PSD')
        axs[i, 1].set_xlim(0, 50)

    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

def wavelet_transform_dwt(X, fs, bands, wavelet='db4'):
    '''
    Decompose signals into EEG frequency bands using DWT
    '''

    def reconstruct(coeffs, freqs, octaves):
        '''
        Reconstruct signal into its bands
        '''

        # Create a copy of coefficients and set all to zero
        new_coeffs = [np.zeros_like(c) for c in coeffs]

        # Find levels that correspond to the desired frequency band
        for i, (low, high) in enumerate(octaves):
            if (freqs[0] <= high and freqs[-1] >= low):
                new_coeffs[level - i] = coeffs[level - i]

        # Reconstruct the signal
        reconstructed = pywt.waverec(new_coeffs, wavelet=wavelet)

        return reconstructed

    n_trials, n_channels, n_samples = X.shape

    # create dictionary to store filtered signals for each band
    filt_dict = {}

    # get max decomposition level using minimum frequency we're interested in
    min_freq = np.min([v for v in bands.values()])
    level = int(np.log2(fs / 2 / min_freq))

    # get frequency range of each octave
    octaves = [(fs / (2 ** (j + 1)), fs / (2 ** j)) for j in range(level + 1)]
    print(octaves)

    for band, freqs in bands.items():
        filt_dict[band] = np.zeros_like(X)

        for trial in range(n_trials):
            for channel in range(n_channels):
                signal = X[trial, channel, :]
                coeffs = pywt.wavedec(signal, wavelet, level=level)     # returns list of lists
                filtered_signal = reconstruct(coeffs=coeffs, freqs=freqs, octaves=octaves)
                filt_dict[band][trial, channel, :] = filtered_signal[:n_samples]

    plot_wavelet(X=X, filt_dict=filt_dict, fs=fs, bands=bands)

    return

def wavelet_transform_cwt(X, fs, bands, wavelet='morl'):
    '''
    Decompose signals into EEG frequency bands using CWT
    '''
    n_trials, n_channels, n_samples = X.shape
    filt_dict = {}              # dictionary for saving the filtered signals

    for band_name, (low_freq, high_freq) in bands.items():
        print(f"Processing {band_name} band...")

        # Calculate scales for this frequency band
        freqs = np.linspace(low_freq, high_freq, num=50)
        scales = (fs * pywt.central_frequency(wavelet)) / freqs

        filt_dict[band_name] = np.zeros((n_trials, n_channels, n_samples))

        for trial in range(n_trials):
            for channel in range(n_channels):
                signal = X[trial, channel, :]
                # perform CWT
                coef, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
                # Reconstruct the signal for this band
                filt_dict[band_name][trial, channel, :] = np.real(np.sum(coef, axis=0))

    plot_wavelet(X, filt_dict, bands, fs)

    return filt_dict
    

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
    # 1. DWT / CWT
    wavelet_transform_dwt(X=X1, fs=fs, bands=bands)
    # wavelet_transform_cwt(X=X1, fs=fs, bands=bands)


    # 2. FIR filter
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

