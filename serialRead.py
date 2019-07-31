import time
import serial

ser = serial.Serial(
    port = "/dev/ttyS0",
    baudrate = 9600,
    timeout = 0.5
)

while True:
    if ser.in_waiting > 0:
        line = ser.readline()
        print(time.asctime())
        print(line)