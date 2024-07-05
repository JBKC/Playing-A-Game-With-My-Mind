'''
Process json data
'''

import json
from scipy.signal import butter, filtfilt

with open('data.json', 'r') as f:
    data = json.load(f)

# Access the data
C3 = data['C3']
C4 = data['C4']
CP3 = data['CP3']
CP4 = data['CP4']

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)




