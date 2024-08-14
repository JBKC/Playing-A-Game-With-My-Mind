
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt



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

def main(X1, X2):
    '''
    X1, X2 shape =
    '''

    # bandpass filter & normalise
    X1_filt = normalise(bpass_filter(X1, 8, 15, 256))
    X2_filt = normalise(bpass_filter(X2, 8, 15, 256))

    print(X1_filt.shape)

    plt.plot(X1_filt)
    plt.show()


    return


if __name__ == '__main__':
    main()
