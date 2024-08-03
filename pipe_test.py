'''
Send realtime accelerometer inputs to pipeline to be used by gaming emulator
Designed for Mario Kart Wii on Dolphin Emulator
To be eventually replaced by EEG inputs
'''

import serial

# // {PRESS, RELEASE} {A, B, X, Y, Z, START, L, R, D_UP, D_DOWN, D_LEFT, D_RIGHT}
# // SET {L, R} [0, 1]
# // SET {MAIN, C} [0, 1] [0, 1]

fifo_path = "/Users/jamborghini/Library/Application Support/Dolphin/Pipes/fifo_pipe"
arduino_port = '/dev/tty.usbmodem14201'
baud_rate = 115200

# Open the serial port
ser = serial.Serial(arduino_port, baud_rate)

try:
    with (open(fifo_path, 'w') as fifo):
        while True:
            data = ser.readline().decode('utf-8').strip().split(',')

            # initialize commands
            commands = ''
            print(data)

            # get individual accelerometer data
            x_data = float(data[0])
            y_data = float(data[1])
            # z_data = float(data[2])

            if y_data < -5:
                commands = {
                    'PRESS A'}                  # keep A stuck on (for driving)
            if y_data > 5:
                commands = {
                    'RELEASE A'}                # option to release A (mostly for menu)

            if x_data > 5:
                commands = {
                    'PRESS D_DOWN',                                 # menu control down
                    f'SET MAIN {0.2 * x_data} 0.5',                 # analog stick right
                }

            if x_data < -5:
                commands = {
                    'PRESS D_UP',
                    f'SET MAIN {0.2 * x_data + 1} 0.5',             # analog stick left
                }

            if -2 < x_data < 2:
                # when accelerometer held level, release commands
                commands = {
                    'RELEASE D_DOWN',
                    'RELEASE D_UP',
                    'SET MAIN 0.5 0.5',
                }

            if commands:
                for command in commands:
                    # fifo.write('RELEASE B' + '\n')
                    fifo.write(command + '\n')
                    fifo.flush()
                    print(f'{command} SENT')


finally:
    ser.close()

'''
improvements:
- calibrate accelerometer by holding steady for 5 seconds in "resting" state
- movements to the emulator are a function of how much the accelerometer has moved from the calibrated state
'''