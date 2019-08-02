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

camera = PiCamera(resolution=res, framerate=30)
rawCapture = PiRGBArray(camera, size=res)

#def writeGauntletPos(img, gauntObj: alg.Gauntlet):
#    dataStr = "G"
#    for rect in gauntObj.rects:
#        coords = alg.shiftImgCoords(img, rect.intrinsicVector.end)
#        dataStr += "{},{};".format(*coords)
#    dataStr += "\n"
#    print(dataStr)
#    ser.write(dataStr.encode("ascii", "ignore"))
    

    
# allow the camera to warmup
time.sleep(0.25)

lastGoodGauntlet = None
circles = None
tapeCnt = None
reprCircle = None
lastGoodIsect = None
TEMPLATE = cv2.imread("template.png")
TEMPLATE = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    startTime = time.time()
    if ser.in_waiting > 0:
        line = ser.readline().decode('ascii')
        if "~" in line.decode() and lastGoodIsect is not None:
            print("Received serial request to print intersection type")
            ser.write("{}".format(lastGoodIsect).encode("ascii", "ignore"))
        print(time.strftime("%H:%M:%S"), line)
    
    try:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        originalImg = frame.array
        
        processedImg, houghImg, dispImg = alg.preprocessFrame(originalImg)
        preprocessTime = time.time()

        # resRect = alg.findTemplate(processedImg, TEMPLATE)
        # resRect.draw(dispImg)

        imageCnts = alg.getContours(processedImg)
        gauntletObj, rectContours = alg.getGauntlet(imageCnts)
        
        if gauntletObj is not None:
            if len(gauntletObj.rects) == 6:
                lastGoodGauntlet = gauntletObj
                gauntletObj.serialWrite(processedImg, ser)
            circles = None
            tapeCnt = None
            reprCircle = None
            lastGoodIsect = None
        else:
            houghImg = alg.undistortPerspective(houghImg)
            circles = alg.findCircles(houghImg)
            tapeCnt, reprCircle = alg.identifyTapeStrip(imageCnts)
            if tapeCnt is not None:
                lastGoodIsect = tapeCnt.descriptor
            #dispImg = cv2.cvtColor(dispImg, cv2.COLOR_GRAY2BGR)

            
            
    except Exception:
        #traceback.print_exc()
        pass
    finally:
        # dispImg = cv2.cvtColor(dispImg, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(dispImg, imageCnts, -1, alg.VIOLET, 1)

        if lastGoodGauntlet is not None and circles is None:
            lastGoodGauntlet.draw(dispImg)
            
        if tapeCnt is not None:
            
            tapeCnt.draw(dispImg)
            reprCircle.draw(dispImg)
        
        
            

        if circles is not None:
            dispImg = cv2.cvtColor(alg.autoCanny(houghImg), cv2.COLOR_GRAY2BGR)
            circles[0].draw(dispImg)
            circles[0].serialWrite(houghImg, ser)
            
        
        #dispImg = cv2.cvtColor(houghImg, cv2.COLOR_GRAY2BGR)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # show the frame
        cv2.imshow("Frame", dispImg)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
    endTime = time.time()
    totalTime = endTime - startTime
    pctUndistorting = (preprocessTime - startTime) / totalTime * 100
#    print("FPS: {:.2f}, {:.2f}% spent undistorting".format(
#        1/(endTime - startTime), pctUndistorting
#    ))
