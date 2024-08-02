'''
Translate gyro inputs into pygame onscreen input
'''

from neurosity import NeurositySDK
import asyncio
from dotenv import load_dotenv
import numpy as np
import os
import crown_realtime_processing
import joblib
import collections
import pygame
import serial
import tracemalloc
import time
import cProfile
import pstats
import io


################################

# coroutine 1: get gyro data (producer)
async def gyro_stream(buffer, ser):

    loop = asyncio.get_running_loop()

    try:
        while True:
            data = ser.readline().decode('utf-8').strip().split(',')
            data = data[2]
            buffer.append(data)

            # yield control back to event loop after callback has been executed
            await asyncio.sleep(0)

    # stop stream when program exits
    finally:
        await buffer.put(None)          # signal end of stream

# coroutine 2: display orientation data (consumer)
async def pygame_display(buffer):
    '''
    :param: buffer: of incoming gyro data
    '''


    pygame.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))

    dot_radius = 10
    dot_color = (255, 0, 0)  # Red color
    BLACK = (0, 0, 0)

    data = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # check for updates in the queue
        if not buffer.empty():
            data = await buffer.get()

        screen.fill(BLACK)

        if data is not None:

            x_distance = screen.get_width() / 2 + (30 * float(data))
            print(x_distance)

            # Draw the dot
            pygame.draw.circle(screen, dot_color, (x_distance, screen.get_height() / 2), dot_radius)

            # Update the display
            pygame.display.flip()

        await asyncio.sleep(0)  # yield control to the event loop

    pygame.quit()


async def main():

    arduino_port = '/dev/tty.usbmodem14201'
    baud_rate = 115200

    buffer = collections.deque()      # create buffer

    # Open the serial port
    ser = serial.Serial(arduino_port, baud_rate)

    # create tasks
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(gyro_stream(buffer, ser))
        task2 = tg.create_task(pygame_display(buffer))

    # run tasks
    await asyncio.gather(task1, task2)

if __name__ == '__main__':
    asyncio.run(main())




