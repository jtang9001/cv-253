# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
 
# initialize the camera and grab a reference to the raw camera capture
res = (640, 480)

camera = PiCamera(
    #sensor_mode=4,
    resolution=res,
    framerate=20)
#camera.resolution = (640, 480)
#camera.framerate = 20
rawCapture = PiRGBArray(camera, size=res)

# allow the camera to warmup
time.sleep(0.25)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def auto_canny_upper(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    upper = int(min(255, (1.0 + sigma) * v))
 
    # return the upper bound
    return upper
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#    threshed = cv2.adaptiveThreshold(
#        blurred,
#        255,
#        cv2.ADAPTIVE_THRESH_MEAN_C,
#        cv2.THRESH_BINARY,
#        blockSize = 5,
#        C = 1)
    
    #print("Doing Hough circle transform...")
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp = 1.5,
        minDist = 20,
        param1 = auto_canny_upper(blurred)
        )
    
    if circles is not None:
        print(circles)
        for (x,y,r) in circles[0,:]:
            cv2.circle(image, (x,y), r, (0,255,0), 2)
            cv2.circle(image, (x,y), 2, (0,0,255), 3)
        
 
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break