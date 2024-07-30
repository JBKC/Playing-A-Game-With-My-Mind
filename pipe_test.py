import os
import time
import serial

fifo_path = "/Users/jamborghini/Library/Application Support/Dolphin/Pipes/fifo_pipe"
arduino_port = '/dev/tty.usbmodem14201'
baud_rate = 115200



# Open the named pipe
# fifo = os.open(fifo_path, os.O_WRONLY)
print("test")

# Open the serial port
ser = serial.Serial(arduino_port, baud_rate)

try:
    while True:
        data = ser.readline().decode('utf-8').strip().split(',')
        x_data = data[0]
        y_data = data[1]
        z_data = data[2]
        print(x_data)
        # os.write(fifo, gyro_data.encode('utf-8'))
        # time.sleep(0.01)  # Add a small delay to avoid overwhelming the pipe
finally:
    ser.close()
    # os.close(fifo)