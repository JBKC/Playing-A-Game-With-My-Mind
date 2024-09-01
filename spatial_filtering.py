import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def plot_psd(freqs, P1, P2, CSP=True):
    '''
    :params P1, P2: shape (n_trials, n_channels, n_samples/2 + 1)
    '''

    channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    if CSP:
        # plot the first and last eigenvectors (the most extreme discrimination)
        ax1.plot(freqs, P1[:, 0, :].mean(axis=0), color='red', linewidth=1, label='right')
        ax1.plot(freqs, P2[:, 0, :].mean(axis=0), color='blue', linewidth=1, label='left')
        ax2.plot(freqs, P1[:, -1, :].mean(axis=0), color='red', linewidth=1, label='right')
        ax2.plot(freqs, P2[:, -1, :].mean(axis=0), color='blue', linewidth=1, label='left')
        # ax1.set_ylim(0, 1)
        # ax2.set_ylim(0, 1)

    if not CSP:
        # plot channels C3 and C4
        ax1.plot(freqs, P1[:, 1, :].mean(axis=0), color='red', linewidth=1, label='right')
        ax1.plot(freqs, P2[:, 1, :].mean(axis=0), color='blue', linewidth=1, label='left')
        ax2.plot(freqs, P1[:, 6, :].mean(axis=0), color='red', linewidth=1, label='right')
        ax2.plot(freqs, P2[:, 6, :].mean(axis=0), color='blue', linewidth=1, label='left')
        ax1.set_title(f'PSD for {channel_names[1]} (controls right side)')
        ax2.set_title(f'PSD for {channel_names[6]} (controls left side)')
        # ax1.set_ylim(0, 50)
        # ax2.set_ylim(0, 50)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_xlim(0, 30)
    ax1.legend()
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim(0, 30)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def logvar(PSD):
    '''
    Inputs PSD, returns a single power value for the plot as the log variance of the PSD
    :param PSD: shape (n_trials, n_channels, n_psd_points)
    :return: log variance of shape (n_trials, n_channels)
    '''

    return np.log(np.var(PSD, axis=2))

def bar_logvar(L1, L2):
    '''
    Plot logvar bar chart of all channels to compare variance of each channel
    :param L1: shape (n_trials, n_channels)
    '''

    plt.figure(figsize=(8, 5))

    x0 = np.arange(L1.shape[1])                     # x axis = number of channels
    x1 = np.arange(L1.shape[1]) + 0.4

    y0 = np.mean(L1, axis=0)                        # y axis = log variance averaged over trials
    y1 = np.mean(L2, axis=0)

    plt.bar(x0, y0, width=0.4, color='red', label='right')
    plt.bar(x1, y1, width=0.4, color='blue', label='left')

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each CSP component')
    plt.xlabel('CSP components')
    plt.ylabel('log-var')
    plt.legend()

    plt.show()

    return

def scatter_logvar(L1, L2):
    '''
    Plot logvar scatter data of channels of interest OR most disriminative eigenvectors (1 point = 1 trial)
    :params L1,L2: shape (n_trials, n_channels)
    '''


    plt.figure(figsize=(8, 5))

    plt.scatter(L1[:, 0], L1[:, -1], color='red', linewidth=1, label='right')
    plt.scatter(L2[:, 0], L2[:, -1], color='blue', linewidth=1, label='left')
    plt.legend()

    plt.show()


def compute_psd(tensor):
    '''
    :param takes in 3D tensor of shape (n_trials, n_channels, n_samples)
    :return:
    power spectral density of each shape (n_trials, n_channels, n_samples/2 + 1)
    freqs of shape ceil(n_samples+1/2))
    '''

    freqs, PSD = scipy.signal.welch(tensor, fs=256, axis=2, nperseg=tensor.shape[2])

    return np.array(freqs), np.array(PSD)

def spatial_filter(X1, X2):
    """
    Compute CSP for two classes of EEG data.

    X1, X2: Shape = (n_trials, n_channels, n_samples)
    """

    def cov(X):
        '''
        Take in 3D tensor X of shape (n_trials, n_channels, n_samples)
        Calculates the covariance matrix of shape (n_channels, n_channels)
        Normalise the covs by dividing by number of samples
        Returns average of covs over all trials
        '''
        n_samples = X.shape[2]
        covs = [np.dot(trial, trial.T) / n_samples for trial in X]

        return np.mean(covs, axis=0)

    def whitening_matrix(sigma):
        # eigendecomposition of composite covariance matrix
        D, U = np.linalg.eigh(sigma)

        # use PCA whitening (without transformation back into original space)
        return np.dot(np.diag(D ** -0.5), U.T)

    # calculate covariance matrices
    R1 = cov(X1)
    R2 = cov(X2)
    # print(f'Covariance matrix  shape: {R1.shape}')                    # shape (n_channels, n_channels)

    # get whitening matrix P from composite covariance matrix
    P = whitening_matrix(R1 + R2)
    # print(f'Whitening matrix shape: {P.shape}')                       # shape (n_channels, n_channels)

    # whiten covariance matrices
    S1 = np.dot(np.dot(P, R1), P.T)
    S2 = np.dot(np.dot(P, R2), P.T)

    # solve generalised eigenvalue problem
    D, U = scipy.linalg.eigh(S1, S1+S2)

    # sort eigenvalues in descending order
    idx = np.argsort(D)[::-1]
    D = D[idx]
    W = U[:, idx]               # W == spatial filters
    print(f'Discriminative eigenvalues: {D}')

    # transform spatial filters back into original space
    W = np.dot(P.T ,W)
    # keep top eigenvectors
    # W = np.column_stack((W[:, :1], W[:, -1:]))

    # project spatial filters onto data
    X1_csp = np.stack([np.dot(W.T, trial) for trial in X1])
    X2_csp = np.stack([np.dot(W.T, trial) for trial in X2])

    return X1_csp, X2_csp, W

def main(X1,X2):

    X1_csp, X2_csp, W = spatial_filter(X1,X2)

    # get Power Spectral Densities from spatially filtered data
    freqs, P1 = compute_psd(X1_csp)
    _, P2 = compute_psd(X2_csp)

    # get Log Variance
    L1 = logvar(P1)
    L2 = logvar(P2)

    # freqs_raw, P1_raw = compute_psd(X1)
    # _, P2_raw = compute_psd(X2)
    #
    # plot_psd(freqs_raw, P1_raw, P2_raw, CSP=False)
    # plot_psd(freqs, P1, P2, CSP=True)
    # bar_logvar(L1,L2)
    # scatter_logvar(L1,L2)

    return X1_csp, X2_csp


if __name__ == '__main__':
    main()