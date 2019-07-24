import cv2
import numpy as np
import imutils
from math import pi

IMGWIDTH = 1200

#openCV uses BGR
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
CYAN = (255,255,0)
YELLOW = (0,255,255)
VIOLET = (255,0,255)
BLACK = (0,0,0)
WHITE = (255,255,255)
GRAY = (127,127,127)

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

class PolarVector(Vector):
    def __init__(self, start, mag, angle):
        super().__init__(start, (start[0] + mag*np.cos(angle), start[1] - mag*np.sin(angle)))

class TapeRect:
    def __init__(self, contour):
        self.contour = contour
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
        self.intrinsicVector = PolarVector(self.center, 1.4*max(self.dims), self.angle)

class Gauntlet:
    def __init__(self, rectObjs):
        centerX = np.mean([rect.center[0] for rect in rectObjs])
        centerY = np.mean([rect.center[1] for rect in rectObjs])
        self.center = (centerX, centerY)
        self.rects = []
        
        for rect in rectObjs:
            self.addTapeRect(rect)

    def addTapeRect(self, tapeRect):
        tapeRect.assignVector(Vector(self.center, tapeRect.center))
        self.rects.append(tapeRect)

    def enumerateTapeRects(self):
        numRects = len(self.rects)
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

    def getRectByNum(self, number):
        for rect in self.rects:
            if hasattr(rect, "number"):
                if rect.number == number:
                    return rect



def getGauntlet(frame):
    #assumes frame is an opencv image object

    greyImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredImg = cv2.GaussianBlur(greyImg, (5,5), 0)
    threshedImg = cv2.threshold(blurredImg, 127, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(threshedImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    rectangles = identifyContours(contours)
    gauntlet = Gauntlet(rectangles)
    gauntlet.enumerateTapeRects()

    for rect in rectangles:
        cv2.drawContours(frame, [rect.boundingBoxContour], -1, GREEN, 2)
        rect.getIntrinsicVector()
        rect.vector.draw(frame)
        rect.intrinsicVector.draw(frame, color = BLUE)

        cv2.putText(
            frame,
            "{:.3f}, {:.3f}".format(rect.vector.angle, rect.intrinsicVector.angle),
            (int(rect.center[0]), int(rect.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, RED, 2
        )

    cv2.circle(frame, (int(gauntlet.center[0]), int(gauntlet.center[1])), 3, RED, 2)

    return frame

def identifyContours(contours):
    rectangles = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        approxCnt = cv2.approxPolyDP(contour, 0.03*perimeter, True)
        if len(approxCnt) == 4:
            tapeObj = TapeRect(approxCnt)
            if 1.5 < tapeObj.aspectRatio < 2.5:
                rectangles.append(tapeObj)

    medianArea = np.median([rect.boundingBoxArea for rect in rectangles])
    while len(rectangles) > 6:
        rectangles.remove(
            max(rectangles, key = lambda rect: abs(rect.boundingBoxArea - medianArea))
        )

    return rectangles

def angleDiff(fromAngle, toAngle):
    return min(
        (
            angleDiffCCW(fromAngle, toAngle),
            angleDiffCW(fromAngle, toAngle)
        ), 
        key = lambda x: abs(x)
    )
    # naiveDiff = fromAngle - toAngle
    # return (naiveDiff + pi) % (2*pi) - pi

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

import glob

for img in glob.glob("*.jpg"):
    originalImg = cv2.imread(img)
    resizedImg = imutils.resize(originalImg, width=IMGWIDTH)

    cv2.imshow("Frame", getGauntlet(resizedImg))
    cv2.waitKey(0)