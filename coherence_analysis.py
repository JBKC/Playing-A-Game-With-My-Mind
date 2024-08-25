'''
8 channels in order: CP3, C3, F5, PO3, PO4, F6, C4, CP4
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def phase_lag_index(tensor):
    '''
    Calculates the PLI over trials
    '''

    # PLIs for each channel combination
    plis = np.zeros((tensor.shape[1], tensor.shape[1]))

    # array of phase angle differences for each trial
    trial_diffs = np.zeros((tensor.shape[0], tensor.shape[-1]))

    for channel in range(tensor.shape[1]):
        for trial in range(tensor.shape[0]):

            # take examples from different electrodes:
            signal1 = tensor[trial, 1, :]
            signal2 = tensor[trial, 6, :]

            # calculate instantaneous phase angle using HT
            analytic1 = scipy.signal.hilbert(signal1)
            analytic2 = scipy.signal.hilbert(signal2)

            ip1 = np.angle(analytic1)
            ip2 = np.angle(analytic2)

            # record running differences
            trial_diffs[trial, :] = abs(ip1-ip2)

    # average over trials
    av_diffs = np.mean(trial_diffs, axis=0)
    print(av_diffs)

    pli = np.abs(np.mean(np.sign(np.imag(np.exp(1j * av_diffs)))))
    print(pli)



def main(dict, band):
    '''
    Takes in dict where dict[band] has shape (no. trials, no. channels, no. samples)
    '''

    phase_lag_index(dict[band])


if __name__ == '__main__':
    main()