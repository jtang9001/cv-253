# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from .gauntlet_alg import getGauntlet, Gauntlet
import serial
 
# initialize the camera and grab a reference to the raw camera capture
res = (640, 480)

ser = serial.Serial(
    port = "/dev/ttyS0",
    timeout = 1.0
)

camera = PiCamera(resolution=res, framerate=20)
rawCapture = PiRGBArray(camera, size=res)

def writeGauntletPos(gauntObj: Gauntlet):
    dataStr = "G"
    for rect in gauntObj.rects:
        dataStr += "{},{};".format(rect.intrinsicVector.end[0], rect.intrinsicVector.end[1])
    dataStr += "\n"
    ser.write(dataStr)

# allow the camera to warmup
time.sleep(0.25)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    originalImage = frame.array

    processedImage, gauntletObj = getGauntlet(frame)
    writeGauntletPos(gauntletObj)
 
    # show the frame
    cv2.imshow("Frame", processedImage)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break