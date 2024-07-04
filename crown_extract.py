'''
Similar to crown_realtime_plot but saves all data in a DataFrame
'''

from neurosity import NeurositySDK
import os
import numpy as np
from dotenv import load_dotenv          # used for account details
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

load_dotenv()

def initialise():

    try:
        # Retrieve environment variables
        device_id = os.getenv("DEVICE_ID")
        email = os.getenv("EMAIL")
        password = os.getenv("PASSWORD")

        # Initialize NeurositySDK
        neurosity = NeurositySDK({"device_id": device_id})

        # Perform login
        neurosity.login({
            "email": email,
            "password": password
        })

        print("Successfully logged in")
        return neurosity

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


####################################################################################

# global variables
iter = 0                    # loop counter
startTime = 0               # reference starting point for timer
output = pd.DataFrame()    # all streamed data
data_stream = {}          # dictionary for each data packet


def set_params():

    return {
        'max_points': 1000,
        'xdata': deque(maxlen=1000),        # Use deque to limit data storage
        'ydata': deque(maxlen=1000),
        'buffer': deque(maxlen=16)  # Buffer to store (x, y) pairs
    }

def interpolate(array):
    # custom function for interpolating Crown time output
    start = array[-17]
    end = array[-1]
    correction = end - (end - start) * 1 / 16
    num_steps = 16
    return np.linspace(start, correction, num_steps)

def init(ax, line):
    line.set_data([], [])
    ax.set_xlim(0, 200)  # Set initial x-axis limits
    ax.set_ylim(-50, 50)  # Set initial y-axis limits
    return line,

def update(frame, params, ax, line):

    if params['buffer']:
        # Get the next point from the buffer
        x, y = params['buffer'].popleft()
        params['xdata'].append(x)
        params['ydata'].append(y)

        # Update the line data
        line.set_data(list(params['xdata']), list(params['ydata']))

        # Rescale and autoscale the view
        ax.relim()
        ax.autoscale_view(True, True, True)

        # Update the axis limits to show all data
        xmin, xmax = min(params['xdata']), max(params['xdata'])
        ymin, ymax = min(params['ydata']), max(params['ydata'])

        # Add some padding (e.g., 10% on each side)
        xpad = max((xmax - xmin) * 0.1, 1)
        ypad = max((ymax - ymin) * 0.1, 1)

        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    return line,

def callback(data, params, channel):
    global iter, startTime, data_stream, output

    # extract key data from raw channels
    data_stream['time'] = data['info']['startTime']
    signals = data['data']
    channels = data['info']['channelNames']

    print(data)

    # normalise start time (measured in milliseconds)
    if iter == 0:
        startTime = data_stream['time']
        data_stream['time'] = 0
    else:
        data_stream['time'] -= startTime

    # pull relevant channel data
    i = channels.index(channel)
    data_stream[channel] = signals[i]

    print(data_stream)

    # add data into output dataframe
    data_stream = pd.DataFrame(data_stream)
    output = pd.concat([output, data_stream], ignore_index=True)

    iter += 1
    print(f"Iteration: {iter}")
    print(output)

    if iter > 1:
        # interpolate time values
        x = interpolate(output['time'].values)              # interpolate time values
        y = output[channel].values[-32:-16]                    # results backshifted by 1 iteration

        # Add the new points to the buffer as (x, y) pairs
        params['buffer'].extend(zip(x, y))

def main(channel):

    params = set_params()

    # set up plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    # initialise headset
    neurosity = initialise()

    def threaded_callback(data):
        callback(data, params, channel)

    # start EEG streaming in a separate thread
    neurosity_thread = threading.Thread(target=neurosity.brainwaves_raw, args=(threaded_callback,))
    neurosity_thread.daemon = True
    neurosity_thread.start()

    # set up real-time animation plot
    anim = FuncAnimation(fig, update, fargs=(params, ax, line), interval=0, blit=True)
    plt.show()

    neurosity = initialise()
    # call the callback function every time data is received from crown
    neurosity.brainwaves_raw(callback)


if __name__ == '__main__':
    channelList = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    channel = input(f'Choose channel from list {channelList}: ')
    if channel in channelList:
        main(channel)
    else:
        print(f"NA channel: '{channel}'")


