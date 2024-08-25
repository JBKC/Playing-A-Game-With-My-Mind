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

    def compute_pli(signal1, signal2):

        # calculate instantaneous phase angle using HT
        analytic1 = scipy.signal.hilbert(signal1)
        analytic2 = scipy.signal.hilbert(signal2)

        ip1 = np.angle(analytic1)
        ip2 = np.angle(analytic2)

        # phase angle difference
        diffs = ip1-ip2
        pli = np.abs(np.mean(np.sign(np.imag(np.exp(1j * diffs)))))

        return pli

    # PLIs for each channel combination
    pli_matrix = np.zeros((tensor.shape[1], tensor.shape[1]))

    # iterate over all possible combinations of channels
    for ch1 in range(tensor.shape[1]):
        for ch2 in range(tensor.shape[1]):

            # array of plis for each trial - shape (no.trials)
            trial_plis = np.zeros((tensor.shape[0]))

            for trial in range(tensor.shape[0]):

                signal1 = tensor[trial, ch1, :]
                signal2 = tensor[trial, ch2, :]

                # calculate pli for a single trial for given channel combination
                pli = compute_pli(signal1, signal2)

                trial_plis[trial] = pli

            # average PLI over trials for given channel combination
            av_pli = np.mean(trial_plis)
            # append to PLI matrix
            pli_matrix[ch1,ch2] = av_pli

    print(pli_matrix)
    print(pli_matrix.shape)

    # visualise PLI between channels
    plt.imshow(pli_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar to indicate the intensity scale
    plt.show()

def main(dict, band):
    '''
    Takes in dict where dict[band] has shape (no. trials, no. channels, no. samples)
    '''

    phase_lag_index(dict[band])


if __name__ == '__main__':
    main()