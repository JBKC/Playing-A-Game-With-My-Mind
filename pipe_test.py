'''
Send realtime gyro inputs to pipeline to be used by gaming emulator
'''

import os
import time
import serial
import fcntl


fifo_path = "/Users/jamborghini/Library/Application Support/Dolphin/Pipes/fifo_pipe"
arduino_port = '/dev/tty.usbmodem14201'
baud_rate = 115200

# Open the serial port
ser = serial.Serial(arduino_port, baud_rate)

try:
    with open(fifo_path, 'w') as fifo:
        while True:
            data = ser.readline().decode('utf-8').strip().split(',')

            # initialize command
            command = ''
            print(data)


            # get individual gyro data
            x_data = float(data[0])
            y_data = float(data[1])
            z_data = float(data[2])


            # if z_data > 0:
            #     command = 'PRESS A'
            #
            # if command:
            #     fifo.write(command + '\n')
            #     fifo.flush()
            #     print("COMMAND SENT")


finally:
    ser.close()



'''
- calibrate gyro by holding steady for 5 seconds in "resting" state
- movements to the emulator are a function of how much the gyro has moved from the calibrated state
'''