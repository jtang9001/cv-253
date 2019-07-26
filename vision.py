# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import algorithms as alg
import serial
import time
import traceback
 
# initialize the camera and grab a reference to the raw camera capture
res = (640,480)

ser = serial.Serial(
    port = "/dev/ttyS0",
    baudrate = 9600,
    timeout = 1.0
)

camera = PiCamera(resolution=res, framerate=20)
rawCapture = PiRGBArray(camera, size=res)

def writeGauntletPos(img, gauntObj: alg.Gauntlet):
    dataStr = "G"
    for rect in gauntObj.rects:
        coords = alg.shiftImageCoords(img, rect.intrinsicVector.end)
        dataStr += "{},{};".format(*coords)
    dataStr += "\n"
    print(dataStr)
    ser.write(dataStr.encode("ascii", "ignore"))
    
# allow the camera to warmup
time.sleep(0.25)

lastGoodGauntlet = None
TEMPLATE = cv2.imread("template.png")

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    startTime = time.time()
    
    try:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        originalImage = frame.array
        
        processedImage, dispImage = alg.preprocessFrame(originalImage)
        undistortTime = time.time()

        resRect = alg.findTemplate(processedImage, TEMPLATE)
        resRect.draw(dispImage)

        # gauntletObj, contours = alg.getGauntlet(processedImage)
        
        # if len(gauntletObj.rects) == 6:
        #     lastGoodGauntlet = gauntletObj
        #     writeGauntletPos(processedImage, gauntletObj)
            
    except Exception:
        traceback.print_exc()
    finally:
        # dispImage = cv2.cvtColor(dispImage, cv2.COLOR_GRAY2BGR)
        # if lastGoodGauntlet is not None:
        #     lastGoodGauntlet.draw(dispImage)
        #     cv2.drawContours(dispImage, contours, -1, alg.YELLOW, 2)
            
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # show the frame
        cv2.imshow("Frame", dispImage)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
    endTime = time.time()
    totalTime = endTime - startTime
    pctUndistorting = (undistortTime - startTime) / totalTime * 100
    print("FPS: {:.2f}, {.2f}% spent undistorting".format(
        1/(endTime - startTime), pctUndistorting
    ))
