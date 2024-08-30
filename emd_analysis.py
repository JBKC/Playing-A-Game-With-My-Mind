import numpy as np
import matplotlib.pyplot as plt
import emd

def emd_sift(dict):
    '''
    Use EMD to break signal into IMFs
    '''

    # parameters for emd.sift.sift()
    sd_thresh = 0.05
    max_imfs = 6
    nensembles = 20
    nprocesses = 6
    ensemble_noise = 0.5

    for k, X in dict.items():

        X = X[12,1,:]

        imf = emd.sift.sift(X, imf_opts={'sd_thresh': sd_thresh})

        plt.figure(figsize=(12, 8))  # Adjust the size as needed

        emd.plotting.plot_imfs(imf)
        plt.show()

    return

def main(dict):
    emd_sift(dict)

    return


if __name__ == '__main__':
    main()
