
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import scipy.io
import pywt
import emd_analysis
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

def iir_notch(dict, fs, freq, index, Q=30):
    '''
    Infinite impulse response filter
    '''

    # select band of interest
    k, X = list(dict.items())[index]
    n_trials, n_channels, _ = X.shape

    w0 = freq / (fs/2)                          # frequency to remove
    b, a = scipy.signal.iirnotch(w0, Q)

    X_filt = np.zeros_like(X)

    for i in range(n_trials):
        for j in range(n_channels):
            X_filt[i, j, :] = scipy.signal.filtfilt(b, a, X[i, j, :])

    dict[k] = X_filt

    return dict

def fir_method(X, fs, bands, numtaps=101):
    '''
    initial attempt - use FIR filter to split signals into frequency bands
    :params: input tensor X for a single class, of shape (n_trials, n_channels, n_samples)
    :returns: dictionary of tensors containing each frequency band
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


def plot_wavelet(X, filt_dict, fs):

    # plot arbitrary signal
    trial = 12
    channel = 2

    n_bands = len(filt_dict)
    fig, axs = plt.subplots(n_bands + 1, 2, figsize=(10, 1.5 * (n_bands + 1)))

    time = np.arange(X.shape[2]) / fs

    # Plot original signal
    axs[0, 0].plot(time, X[trial, channel, :])
    axs[0, 0].set_title('Original Signal')

    # Plot original signal's PSD
    f, psd = scipy.signal.welch(X[trial, channel, :], fs, nperseg=X.shape[-1], window='hann')
    axs[0, 1].plot(f, psd)
    axs[0, 1].set_title('Power Spectral Density')
    axs[0, 1].set_ylabel('PSD')
    axs[0, 1].set_xlim(0, 64)

    for i, (band_name, band_data) in enumerate(filt_dict.items(), start=1):
        # Plot filtered signal
        axs[i, 0].plot(time, band_data[trial, channel, :])
        axs[i, 0].set_title(f'{band_name}')

        # Plot filtered signal's PSD
        f, psd = scipy.signal.welch(band_data[trial, channel, :], fs, nperseg=1024)
        axs[i, 1].plot(f, psd)
        axs[i, 1].set_xlim(0, 64)

    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

def wavelet_transform_dwt(X, fs, wavelet='db4'):
    '''
    Decompose signals into EEG frequency bands (octaves) using DWT
    '''

    def reconstruct(coeffs, level):
        '''
        Reconstruct signal into its bands
        :params: list of DWT coefficients for each level, coefficients level of interest
        '''

        # create array of new_coefficients
        new_coeffs = [np.zeros_like(c) for c in coeffs]

        # set frequency range of interest to non-zero
        new_coeffs[level] = coeffs[level]

        # reconstruct the signal
        return pywt.waverec(new_coeffs, wavelet=wavelet)

    n_trials, n_channels, n_samples = X.shape

    # get max decomposition level using minimum frequency that we're interested in
    min_freq = 8            # 8Hz taken as the lower end of alpha range
    n_levels = int(np.log2(fs / 2 / min_freq))

    # pick number of octaves - e.g. we want (8-16), (16-32), (32,64)
    n_octaves = 3

    # get frequency range of each octave
    octaves = [(fs / (2 ** (j + 1)), fs / (2 ** j)) for j in range(1, n_levels + 1)]

    # take octaves of interest & reverse order
    octaves = octaves[-n_octaves:][::-1]
    print(octaves)

    # create dictionary to store filtered signals for each octave
    filt_dict = {f'{low:.1f}-{high:.1f}Hz': np.zeros_like(X) for low, high in octaves}

    # iterate through each signal (total = n_trials * n_channels)
    for trial in range(n_trials):
        for channel in range(n_channels):
            # get DWT coefficients
            coeffs = pywt.wavedec(X[trial, channel, :], wavelet, level=n_levels)

            # reconstruct the bands we want by iterating through octaves
            for i, (low, high) in enumerate(octaves):
                band_name = f'{low:.1f}-{high:.1f}Hz'

                # convert levels to octaves (ie. first octave of interest == 2nd level)
                level = i + (n_levels - n_octaves)
                # reconstruct the signal
                filt = reconstruct(coeffs, level)
                filt_dict[band_name][trial, channel, :] = filt[:n_samples]

    return filt_dict

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

    ### Methods for extracting EEG frequency bands
    # 1. DWT
    X1_dict = wavelet_transform_dwt(X=X1, fs=fs)
    X2_dict = wavelet_transform_dwt(X=X2, fs=fs)

    # 2. FIR filter
    # X1_dict = fir_method(X=X1, fs=fs, bands=bands)
    # X2_dict = fir_method(X=X2, fs=fs, bands=bands)

    # filter out power line noise
    index = -1      # which octave of DWT to apply filter to
    X1_dict = iir_notch(dict=X1_dict, fs=fs, freq=50, index=index)
    X2_dict = iir_notch(dict=X2_dict, fs=fs, freq=50, index=index)

    # plot_wavelet(X=X1, filt_dict=X1_dict, fs=fs)

    # empirical mode decomposition
    emd_analysis.main(X1_dict)

    #########################

    ### Coherence analysis: note - must use FIR filter method

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

