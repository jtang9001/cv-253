# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from gauntlet_alg import getGauntlet, Gauntlet, preprocessFrame, YELLOW
import serial
import time
import traceback
 
# initialize the camera and grab a reference to the raw camera capture
res = (640,480)

ser = serial.Serial(
    port = "/dev/ttyAMA0",
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

lastGoodGauntlet = None

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    startTime = time.time()
    
    try:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        originalImage = frame.array
        
        processedImage = preprocessFrame(originalImage)

        gauntletObj, contours = getGauntlet(processedImage)
        
        if len(gauntletObj.rects) == 6:
            lastGoodGauntlet = gauntletObj
        
        
        #writeGauntletPos(gauntletObj)
    except Exception:
        traceback.print_exc()
    finally:
        processedImage = cv2.cvtColor(processedImage, cv2.COLOR_GRAY2BGR)
        if lastGoodGauntlet is not None:
            lastGoodGauntlet.draw(processedImage)
            cv2.drawContours(frame, contours, -1, YELLOW, 2)
            
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # show the frame
        cv2.imshow("Frame", processedImage)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
    endTime = time.time()
    print("FPS: {:.2f}".format(1/(endTime - startTime)))
