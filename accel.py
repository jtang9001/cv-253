import traceback
from mpu6050 import mpu6050
try:
    import RPi.GPIO as GPIO
except RuntimeError:
    print("Error importing RPi.GPIO! Use 'sudo' to run script")

sensor = mpu6050(0x68)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT, initial = GPIO.LOW)
THRESH = 15

try:
    while True:
        accelData = sensor.get_accel_data()
        mag = (accelData["x"] ** 2 + accelData["y"] ** 2 + accelData["z"] ** 2) ** (0.5)
        print("x={x}, y={y}, z={z}".format(**accelData))
        print("mag:", mag)
        if mag > THRESH:
            GPIO.output(18, GPIO.HIGH)
        else:
            GPIO.output(18, GPIO.LOW)
except:
    traceback.print_exc()
finally:
    GPIO.cleanup()