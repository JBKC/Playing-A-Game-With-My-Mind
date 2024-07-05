'''
Script for plotting real-time raw data from a single channel from Neurosity Crown in a scrolling window
'''

from neurosity import NeurositySDK
import os
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

# load environment variables: device_id, email and password
load_dotenv()

def initialise():
    '''
    Initialise Crown for receiving data
    :return: neurosity: instance of NeurositySDK
    '''

    try:
        # get environment variables
        device_id = os.getenv("DEVICE_ID")
        email = os.getenv("EMAIL")
        password = os.getenv("PASSWORD")

        # initialise NeurositySDK
        neurosity = NeurositySDK({"device_id": device_id})

        # login to Neurosity account
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

# set global variables
iter = 0                        # loop counter
startTime = 0                   # reference starting point for timer
times = []                      # keep track of what time packets are received
data_stream = {}                # dictionary for each data packet

def set_params():
    '''
    Set data storage limits
    returns:
        signal_len: maximum amount of time values to be held. also equates to scrolling speed
        buffer: to store x,y pairs. set to 16 (size of Crown data packets)
    '''

    signal_len = 100

    return {
        'xdata': deque(maxlen=signal_len),
        'ydata': deque(maxlen=signal_len),
        'buffer': deque(maxlen=16)
    }

def interpolate(array):
    '''
    Custom function for interpolating Crown time output
    :param array: array of time values indicating the start time of each data packet
    :return: interpolated array
    '''

    start = array[-2]
    end = array[-1]
    correction = end - (end - start) * 1 / 16
    num_steps = 16
    return np.linspace(start, correction, num_steps)

def update(_, params, ax, line):
    '''
    Function that gets called iteratively to plot new data packets
    :param _: blank for frame
    :param params: xdata, ydata and buffer
    :param ax: axes
    :param line: previous plot line
    :return: udpated plot line
    '''

    if params['buffer']:
        # unpack each point from buffer to create smooth plotting
        x, y = params['buffer'].popleft()
        params['xdata'].append(x)
        params['ydata'].append(y)

        # update the plot with the latest data
        line.set_data(list(params['xdata']), list(params['ydata']))

        # autoscale the plot
        ax.relim()
        ax.autoscale_view(True, True, True)
        ax.set_xlim(min(params['xdata']), max(params['xdata']))

    return line,

def callback(data, params, channel):
    '''
    Function that is called when a data packet is received
    :param data: raw data packet from Crown
    :param params: xdata, ydata and buffer
    :param channel: specified channel to view
    '''

    global iter, startTime, times, data_stream, output

    # extract data from raw channels
    data_stream['time'] = data['info']['startTime']
    signals = data['data']
    channels = data['info']['channelNames']

    # pull specified channel data
    i = channels.index(channel)
    data_stream[channel] = signals[i]

    # normalise start time (measured in milliseconds)
    if iter == 0:
        startTime = data_stream['time']
        data_stream['time'] = 0
    else:
        data_stream['time'] -= startTime

    # append to list of start times of each data packet
    times.append(data_stream['time'])

    iter += 1
    print(f"Iteration: {iter} \nTime (ms): {data_stream['time']}")

    if iter > 1:
        # interpolate time values
        x = interpolate(times)
        y = data_stream[channel]
        # add new data packet to the buffer as (x, y) pairs
        params['buffer'].extend(zip(x, y))


def main(channel):
    '''
    Main function to set up and run SDK instance
    '''

    # initialise headset
    neurosity = initialise()
    params = set_params()

    # set up animated plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    def threaded_callback(data):
        # specify parameters that callback function takes
        callback(data, params, channel)

    # start EEG streaming in a separate thread
    neurosity_thread = threading.Thread(target=neurosity.brainwaves_raw, args=(threaded_callback,))
    neurosity_thread.daemon = True
    neurosity_thread.start()

    # call update function as often as possible to plot in real-time
    anim = FuncAnimation(fig, update, fargs=(params, ax, line), interval=0, blit=True)
    plt.title(f'Channel: {channel}')
    plt.show()


if __name__ == '__main__':

    channelList = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    channel = input(f'Choose channel from list {channelList}: ')
    if channel in channelList:
        main(channel)
    else:
        print(f"NA channel: '{channel}'")


