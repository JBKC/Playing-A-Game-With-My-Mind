'''
Translate gyro inputs into pygame onscreen input
'''

import asyncio
import collections
import pygame
import serial

################################

# coroutine 1: get gyro data (producer)
async def gyro_stream(buffer, ser):

    try:
        while True:
            data = ser.readline().decode('utf-8').strip().split(',')
            buffer.append((float(data[0]),float(data[1]))) # save (x,y) gyro data

            # yield control back to event loop after callback has been executed
            await asyncio.sleep(0)

    # stop stream when program exits
    finally:
        buffer.append(None)          # signal end of stream

# coroutine 2: display orientation data (consumer)

async def pygame_display(buffer):
    '''
    :param: buffer: of incoming gyro data
    '''

    pygame.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))

    dot_radius = 10
    dot_color = (255, 0, 0)  # red
    BLACK = (0, 0, 0)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        if buffer:
            data = buffer[-1]  # Get the most recent data
            buffer.clear()  # Clear the buffer
            print(data)

            if data is not None:
                # convert gyro data to onscreen image
                x_dist = int(width / 2 + (30 * data[0]))
                y_dist = int(height / 2 + (30 * data[-1]))
                pygame.draw.circle(screen, dot_color, (x_dist, y_dist), dot_radius)

        pygame.display.flip()
        clock.tick(5)  # fps limit
        await asyncio.sleep(0)

    pygame.quit()



async def main():

    arduino_port = '/dev/tty.usbmodem14201'
    baud_rate = 115200

    buffer = collections.deque(maxlen=1)

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




