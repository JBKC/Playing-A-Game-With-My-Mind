'''
Unfinished script
General idea is interpolate regions of the signal where artifacts are located (by setting to the signal mean)
By creating a mask over regions which deviate too far from the stdev
To improve, use a sliding window that dynamically checks for changes in stdev
'''


import numpy

def main():
    mask = np.abs(trial_data - mean) > 0.25 * std

    # segment masked areas
    mask_idx = np.where(mask)[0]
    # print(mask_idx)

    start = [np.where(mask)[0][0]]  # starting point of each masked segment
    end = []  # ending point of each masked segment

    # get start and end indices of each masked segment
    for i, idx in enumerate(mask_idx[:-1]):
        if mask_idx[i + 1] - idx > 1:
            start.append(mask_idx[i + 1])
            end.append(idx)
    end.append(np.where(mask)[0][-1])

    # arrays of mask indices
    masks = [np.arange(start[i], end[i] + 1) for i in range(len(start))]
    print(start, end)
    print(masks)

    # get mean of non-masked area
    non_mean = np.mean(trial_data[~mask])

    # apply non-masked mean to masked areas
    for i, mask in enumerate(masks):
        X[trial, 1, mask] = non_mean


if __name__ == '__main__':
    main()
