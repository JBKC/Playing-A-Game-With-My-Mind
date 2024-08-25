'''
Exploring connectivity / coherence between EEG channels during motor imagery
8 channels in order: CP3, C3, F5, PO3, PO4, F6, C4, CP4
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal



def phase_lag_index(tensor):
    '''
    Calculates the PLI over trials
    '''

    # calculate instantaneous phase angle using HT
    def compute_pli_hybrid(tensor):

        n_trials, n_channels, n_samples = tensor.shape
        # matrix for PLIs (1 for every 2-channel combination
        plis = np.zeros((n_channels, n_channels, n_samples))

        # get instantaneous phase
        analytic = scipy.signal.hilbert(tensor, axis=-1)
        phase = np.angle(analytic)

        # to vectorize using broadcasting
        for ch1 in range(n_channels):
            for ch2 in range(n_channels):
                # get differences at each timepoint for each trial - shape (n_trials, n_samples)
                diffs = phase[:, ch1, :] - phase[:, ch2, :]
                # calculate the PLI over trials - shape (n_samples,)
                pli = np.abs(np.mean(np.sign(np.imag(np.exp(1j * diffs))), axis=0))
                plis[ch1, ch2, :] = pli

                # additional - PLI over time:
                


        # plotting
        # comparison channel
        ch2 = 0

        plt.figure(figsize=(12, 8))

        for ch1 in range(n_channels):
            plt.plot(plis[ch1, ch2, :], label=f'{ch1+1}, {ch2+1}')

        plt.xlabel('Time')
        plt.ylabel('PLI')
        plt.title('Phase Lag Index (PLI) Over Time for All Channel Pairs')
        plt.legend(loc='upper right')
        plt.show()


    compute_pli_hybrid(tensor)


def main(dict, band):
    '''
    Takes in dict where dict[band] has shape (n_trials, n_channels, n_samples)
    '''

    phase_lag_index(dict[band])


if __name__ == '__main__':
    main()