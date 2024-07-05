'''
Process json data
'''

import json
from scipy.signal import butter, filtfilt
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

with open('data.json', 'r') as f:
    data = json.load(f)

filtered = {}

def bpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalise(signal):
    # zero mean the signal
    return signal - np.mean(signal)

for key, value in data.items():
    filtered[key] = normalise(bpass_filter(value, lowcut=7, highcut=30.0, fs=256, order=5))

# plt.plot(data['C3'])
# plt.show()
# plt.plot(filtered['C3'])
# plt.show()

### this is the part where you stitch all class data together to find overall covariance matrix

def csp(X1, X2, k):
    """
    Compute CSP for two classes of EEG data.

    X1, X2: EEG data for class 1 and class 2
            Shape: (n_trials, n_channels, n_samples)
            trials = number of individual recordings
            channels = 4 (C3,C4,PC3,PC4)
            samples = 768 datapoints
    k: number of top and bottom eigenvectors
    """
    # compute covariance matrices
    cov1 = np.mean([np.cov(trial) for trial in X1], axis=0)
    cov2 = np.mean([np.cov(trial) for trial in X2], axis=0)

    # get composite covariance + do eigendecomposition
    cov_comp = cov1 + cov2
    eig_values, eig_vectors = la.eigh(cov_comp)
    # sort eigs
    idx = np.argsort(eig_values)[::-1]
    eig_values, eig_vectors = eig_values[idx], eig_vectors[:, idx]

    # whitening transformation
    W = np.dot(np.diag(np.sqrt(1/(eig_values + 1e-6))), eig_vectors.T)

    # calculate spatial filters
    S1 = np.dot(np.dot(cov1, W.T), W)
    S2 = np.dot(np.dot(cov2, W.T), W)

    # generalised eigendecomposition
    eig_values, eig_vectors = np.linalg.eigh(S1, S1 + S2)           # key line that maximises variances for one class and minimises for the other
    idx = np.argsort(eig_values)[::-1]
    eig_values, eig_vectors = eig_values[idx], eig_vectors[:, idx]

    # full CSP projection matrix
    W_csp = np.dot(eig_vectors.T, W)

    # Select top CSP components
    W_csp = np.concatenate((W_csp[:, :k], W_csp[:, -k:]), axis=1)

    return W_csp



# X1 for class 1, X2 for class 2
W = csp(X1=X1, X2=X2, k=3)

# To apply CSP to your data:
Z1 = np.dot(W.T, X1)
Z2 = np.dot(W.T, X2)



