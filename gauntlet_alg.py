import cv2
import numpy as np
import imutils
import itertools
from math import pi

IMGWIDTH = 972
TAPE_TO_HOLE_RATIO = 1.4
BINARIZATION_THRESHOLD = 80
POLY_APPROX_COEFF = 0.05
IMGRES = (1296,972)
IMGAREA = 1296*972

#openCV uses BGR
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
CYAN = (255,255,0)
YELLOW = (0,204,204)
VIOLET = (255,0,255)
BLACK = (0,0,0)
WHITE = (255,255,255)
GRAY = (127,127,127)

class Circle:
    def __init__(self,x,y,r):
        self.x = x
        self.y = y
        self.center = (self.x, self.y)
        self.r = abs(r)
    
    def draw(self, img, color = RED, thickness = 2):
        cv2.circle(
            img, 
            ( int(round(self.x)), int(round(self.y)) ), 
            int(round(self.r)), 
            color, thickness)

class ThreePointCircle(Circle):
    def __init__(self,a,b,c):
        x,y,z = a[0] + (a[1])*1j, b[0] + (b[1])*1j, c[0] + (c[1])*1j
        w = z-x
        w /= y-x
        c = (x-y)*(w-abs(w)**2)/2j/w.imag-x

        xCoord = float(-1*c.real)
        yCoord = float(-1*c.imag)
        r = abs(c+x)
        super().__init__(xCoord,yCoord,r)

class Vector:
    def __init__(self, start, end):
        assert len(start) == 2
        assert len(end) == 2
        self.start = start
        self.end = end
        self.vector = [end[0] - start[0], start[1] - end[1]]
        self.magnitude = (self.vector[0]**2 + self.vector[1]**2) ** 0.5
        self.angle = np.arctan2(self.vector[1], self.vector[0])

    def draw(self, img, color = CYAN, thickness = 2):
        cv2.line(
            img, 
            tuple(int(x) for x in self.start), 
            tuple(int(x) for x in self.end), 
            color, thickness)

    def drawEnd(self, img, color = BLUE, radius = 2, thickness = 2):
        cv2.circle(img, tuple(int(x) for x in self.end), radius, color, thickness)

    def drawStart(self, img, color = RED, radius = 2, thickness = 2):
        cv2.circle(img, tuple(int(x) for x in self.start), radius, color, thickness)

    def drawEndpoints(self, img, color = BLUE, radius = 2, thickness = 2):
        cv2.circle(img, tuple(int(x) for x in self.end), radius, color, thickness)
        cv2.circle(img, tuple(int(x) for x in self.start), radius, color, thickness)

class PolarVector(Vector):
    def __init__(self, start, mag, angle):
        super().__init__(start, (start[0] + mag*np.cos(angle), start[1] - mag*np.sin(angle)))

class TapeRect:
    def __init__(self, contour):
        self.contour = contour
        self.contourArea = cv2.contourArea(self.contour)
        self.boundingBox = cv2.minAreaRect(contour)
        self.boundingBoxContour = np.int0(cv2.boxPoints(self.boundingBox))
        self.center = self.boundingBox[0]
        self.dims = self.boundingBox[1]
        self.aspectRatio = self.dims[0] / self.dims[1]
        self.boundingBoxArea = self.dims[0] * self.dims[1]
        self.angle = -1 * self.boundingBox[2]

        if self.aspectRatio < 1:
            self.aspectRatio = 1 / self.aspectRatio
            self.angle += 90

        self.angle = degToRad(self.angle)

    def assignNumber(self, number: int):
        self.number = number

    def assignVector(self, vector: Vector):
        self.vector = vector

    def getIntrinsicVector(self):
        assert hasattr(self, "number")
        assert hasattr(self, "vector")
        if abs(angleDiff(self.vector.angle, self.angle)) > pi/2:
            self.angle = (self.angle + pi) % (2*pi)
        self.intrinsicVector = PolarVector(self.center, TAPE_TO_HOLE_RATIO*max(self.dims), self.angle)

class Gauntlet:
    def __init__(self, rectObjs):
        # centerX = np.mean([rect.center[0] for rect in rectObjs])
        # centerY = np.mean([rect.center[1] for rect in rectObjs])
        # self.rectMean = (centerX, centerY)
        self.rects = []
        
        for rect in rectObjs:
            self.addTapeRect(rect)

    def addTapeRect(self, tapeRect):
        #tapeRect.assignVector(Vector(self.rectMean, tapeRect.center))
        self.rects.append(tapeRect)

    def enumerateTapeRects(self):
        numRects = len(self.rects)
        if numRects == 0:
            print("Warning: no rects found in enumerateTapeRects")
            return
        elif numRects == 1:
            self.rects[0].assignNumber(0)
            return
        
        minCcwDiffs = np.zeros(numRects)

        for i in range(numRects):
            ccwDiffs = [angleDiffCCW(self.rects[i].vector.angle, targetRect.vector.angle) for targetRect in self.rects]
            ccwDiffs.remove(0)
            minCcwDiffs[i] = min(ccwDiffs)
        
        maxDiffIndex = np.argmax(minCcwDiffs)
        refRect = self.rects[maxDiffIndex]
        refRect.assignNumber(0)
        refAngle = refRect.vector.angle

        self.rects.sort(key = lambda rect: angleDiffCW(refAngle, rect.vector.angle), reverse = True)

        for i, rect in enumerate(self.rects):
            rect.assignNumber(i)

    def encircleRects(self):
        assert len(self.rects) >= 3
        self.circles = []

        for comb in itertools.combinations(self.rects, 3):
            centers = tuple(rect.center for rect in comb)
            circle = ThreePointCircle(*centers)
            if 80 < circle.r < 120:
                self.circles.append(circle)

        smallestCircle = min(self.circles, key = lambda circle: circle.r)
        self.circles = [
            circle for circle in self.circles \
                if dist(circle.center, smallestCircle.center) < 0.05*IMGWIDTH
            ]

        centerX = np.mean([circle.x for circle in self.circles])
        centerY = np.mean([circle.y for circle in self.circles])
        avgR = np.mean([circle.r for circle in self.circles])
        self.center = (centerX, centerY)
        self.centerCircle = Circle(centerX, centerY, avgR)

    def getRectByNum(self, number):
        for rect in self.rects:
            if hasattr(rect, "number"):
                if rect.number == number:
                    return rect



def getGauntlet(frame):
    #assumes frame is an opencv image object

    greyImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredImg = cv2.GaussianBlur(greyImg, (5,5), 0)
    #threshedImg = autoCanny(blurredImg)
    #threshedImg = cv2.threshold(blurredImg, BINARIZATION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    threshedImg = cv2.adaptiveThreshold(blurredImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 10)
    #note: last two arguments change the adaptive behavior of this threshold.
    #the second last argument is the size of the sample to take to determine a mean to use
    #as the threshold value. lower tends to make the image look more "grainy".
    #last argument can be thought of as exposure. it is a constant that is subtracted from
    #each pixel before deciding if it goes to black or white. higher values make the image
    #appear more bright/whiter/washed out.
    
    frame = cv2.cvtColor(threshedImg, cv2.COLOR_GRAY2BGR)

    contours = cv2.findContours(threshedImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(frame, contours, -1, YELLOW, 2)

    rectangles = identifyContours(contours)
    gauntlet = Gauntlet(rectangles)
    gauntlet.encircleRects()

    for rect in rectangles:
        cv2.drawContours(frame, [rect.contour], -1, GREEN, 2)
        # rect.getIntrinsicVector()
        # rect.vector.draw(frame)
        # rect.intrinsicVector.draw(frame, color = BLUE)
        # rect.intrinsicVector.drawEnd(frame)
        # relCircleCenter = shiftImageCoords(frame, rect.intrinsicVector.end)

        # cv2.putText(
        #     frame,
        #     "{:.4}".format(rect.aspectRatio),
        #     (int(rect.center[0]), int(rect.center[1])),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.75, RED, 2
        # )

    # for circle in gauntlet.circles:
    #     circle.draw(frame)

    gauntlet.centerCircle.draw(frame)
    cv2.circle(frame, (int(gauntlet.center[0]), int(gauntlet.center[1])), 3, RED, 2)

    return frame, gauntlet

def identifyContours(contours):
    rectangles = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        approxCnt = cv2.approxPolyDP(contour, POLY_APPROX_COEFF*perimeter, True)
        if len(approxCnt) == 4:
            tapeObj = TapeRect(approxCnt)
            if all((
                1.5 < tapeObj.aspectRatio < 3,
                0.0006 < tapeObj.boundingBoxArea/IMGAREA < 0.0015,
                tapeObj.contourArea / tapeObj.boundingBoxArea > 0.8
            )):
                rectangles.append(tapeObj)

    #medianArea = np.median([rect.boundingBoxArea for rect in rectangles])
    while len(rectangles) > 6:
        medianArea = np.median([rect.boundingBoxArea for rect in rectangles])
        rectangles.remove(
            max(rectangles, key = lambda rect: abs(rect.boundingBoxArea - medianArea))
        )

    return rectangles

def autoCanny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def angleDiff(fromAngle, toAngle):
    return min(
        (
            angleDiffCCW(fromAngle, toAngle),
            angleDiffCW(fromAngle, toAngle)
        ), 
        key = lambda x: abs(x)
    )

def angleDiffCCW(fromAngle, toAngle):
    naiveDiff = toAngle - fromAngle
    if naiveDiff < 0:
        return naiveDiff + 2*pi
    else:
        return naiveDiff

def angleDiffCW(fromAngle, toAngle):
    naiveDiff = toAngle - fromAngle
    if naiveDiff > 0:
        return naiveDiff - 2*pi
    else:
        return naiveDiff

def degToRad(degrees):
    return degrees * pi / 180

def shiftImageCoords(img, coord):
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    return (round(coord[0] - imgWidth/2), round(imgHeight/2 - coord[1]))

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

if __name__ == "__main__":
    import glob
    import traceback

    for img in glob.glob("batch 3 photos/*.jpg"):
        try:
            originalImg = cv2.imread(img)
            resizedImg = imutils.resize(originalImg, width=IMGWIDTH)

            cv2.imshow("Frame", getGauntlet(resizedImg)[0])
            cv2.waitKey(0)
        except Exception:
            traceback.print_exc()