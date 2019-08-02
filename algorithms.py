import cv2
import numpy as np
import imutils
import itertools
import traceback
from math import pi

#configuration constants
TAPE_TO_HOLE_RATIO = 1.35

PREP_GAUSS_SIZE = 11
PREP_ADAPTIVE_BIN_SIZE = 101
PREP_ADAPTIVE_EXPOSURE = 30
BINARIZATION_THRESHOLD = 100

ENCIRCLE_MIN_R = 50
ENCIRCLE_MAX_R = 120
ENCIRCLE_RECT_DIST_DEV = 0.3
ENCIRCLE_MAX_RECTS = 8

REF_VECT_MAX_CW_DIFF = -5*pi/6
REF_VECT_MIN_CW_DIFF = -1*pi/6

HOUGH_CIRCLE_THRESH = 140 # larger means less circles detected
HOUGH_ACCUM_RES = 1.3 #larger means less resolution in accumulator
HOUGH_MIN_SEPARATION = 20
HOUGH_MIN_R = 50
HOUGH_MAX_R = 115

TAPE_STRIP_MIN_LINE_RATIO = 0.05
TAPE_STRIP_POINT_MARGIN = 0.02
TAPE_STRIP_POLY_COEFF = 0.005
TAPE_STRIP_MIN_Y = 240
TAPE_STRIP_MIN_ANGLE = pi/5
TAPE_STRIP_MAX_ANGLE = 5*pi/8
TAPE_STRIP_ANGLE_THRESH = 3*pi/8

PERS_X_OFFSET = 85
PERS_Y_OFFSET = 95
ADDL_X_OFFSET = 50

RECT_MIN_AR = 1.4 #min aspect ratio
RECT_MAX_AR = 2.5 #max aspect ratio
RECT_MIN_AREA_PCT = 0.00075
RECT_MAX_AREA_PCT = 0.003
RECT_MIN_BBOX_FILL = 0.8 #min pct that rect contour fills its minimal bounding box

POLY_APPROX_COEFF = 0.04
DEFAULT_IMG_X_OFFSET = 40
IMGRES = (640,480)
IMGWIDTH = IMGRES[0]
IMGHEIGHT = IMGRES[1]
IMGAREA = IMGRES[0] * IMGRES[1]

#standardized color tuples
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
ORANGE = (0,165,255)

class Rectangle:
    def __init__(self, center, dims, angle):
        self.center = center
        self.dims = dims
        self.angle = angle
        self.contour = np.int0(cv2.boxPoints((center, dims, angle)))

    def draw(self, img):
        cv2.drawContours(img, [self.contour], -1, GREEN, 2)

class ResultRect(Rectangle):
    def __init__(self, center, dims, angle, metric):
        self.metric = metric
        super().__init__(center, dims, angle)
    
    def draw(self, img):
        cv2.drawContours(img, [self.contour], -1, GREEN, 2)
        cv2.putText(
            img, "{:.2f}".format(self.metric), 
            self.center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, ORANGE, 1
        )

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
        cv2.putText(
            img, "{:.2f}".format(self.r),
            ( int(round(self.x)), int(round(self.y)) ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, ORANGE, 1
        )
        
    def serialWrite(self, img, serialObject):
        coords = shiftImageCoords(img, self.center, ADDL_X_OFFSET)
        dataStr = "P{},{};\n".format(*coords)
        print(dataStr, self.r)
        serialObject.write(dataStr.encode("ascii", "ignore"))
        
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
        self.magnitude = dist(self.start, self.end)
        self.angle = np.arctan2(self.vector[1], self.vector[0])

    def scale(self, scaleFactor):
        self.magnitude *= scaleFactor
        self.vector = [component * scaleFactor for component in self.vector]
        self.end = (self.start[0] + self.vector[0], self.start[1] - self.vector[1])

    def isOffEdge(self):
        return isPointOffEdge(self.start) or isPointOffEdge(self.end)

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
        self.perimeter = cv2.arcLength(contour, True)
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
        self.intrinsicVector = PolarVector(self.center, max(self.dims), self.angle)
        self.intrinsicVector.scale(TAPE_TO_HOLE_RATIO)

    def draw(self, frame):
        cv2.drawContours(frame, [self.contour], -1, GREEN, 2)

    def drawBB(self, frame):
        cv2.drawContours(frame, [self.boundingBoxContour], -1, GREEN, 2)

class TapePoly(TapeRect):
    def __init__(self, contour):
        super().__init__(contour)
    
    def simplifyContour(self, simplifyCoeff = POLY_APPROX_COEFF):
        self.contour = cv2.approxPolyDP(self.contour, simplifyCoeff*self.perimeter, True)
        self.numVertices = self.contour.shape[0]
        self.perimeter = cv2.arcLength(self.contour, True)
        self.contourArea = cv2.contourArea(self.contour)
        self.boundingBox = cv2.minAreaRect(self.contour)
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

    def assignDescriptor(self, descriptor):
        self.descriptor = descriptor

    def draw(self, frame):
        cv2.drawContours(frame, [self.contour], -1, GREEN, 2)
        if hasattr(self, "descriptor"):
            cv2.putText(
                frame,
                "{}".format(self.descriptor),
                (int(self.center[0]), int(self.center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, ORANGE, 2
            )

class Gauntlet:
    def __init__(self, rectObjs):
        centerX = np.mean([rect.center[0] for rect in rectObjs])
        centerY = np.mean([rect.center[1] for rect in rectObjs])
        self.rectMean = (centerX, centerY)
        self.rects = []
        for rect in rectObjs:
            self.addRect(rect)

    def addRect(self, tapeRect):
        #tapeRect.assignVector(Vector(self.rectMean, tapeRect.center))
        self.rects.append(tapeRect)

    def enumerateRects(self):
        numRects = len(self.rects)
        if numRects == 0:
            print("Warning: no rects found in enumerateRects")
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
        
        if len(self.rects) > ENCIRCLE_MAX_RECTS:
            #print("More than {} rectangles detected. Truncating list!".format(ENCIRCLE_MAX_RECTS))
            processRects = self.rects[:ENCIRCLE_MAX_RECTS]
        else:
            processRects = self.rects

        for comb in itertools.combinations(processRects, 3):
            centers = tuple(rect.center for rect in comb)
            circle = ThreePointCircle(*centers)
            if ENCIRCLE_MIN_R < circle.r < ENCIRCLE_MAX_R:
                self.circles.append(circle)

        smallestCircle = min(self.circles, key = lambda circle: circle.r)
        self.circles = [
            circle for circle in self.circles \
                if dist(circle.center, smallestCircle.center) < 0.15*IMGWIDTH
            ]

        centerX = np.mean([circle.x for circle in self.circles])
        centerY = np.mean([circle.y for circle in self.circles])
        self.avgR = np.mean([circle.r for circle in self.circles])
        self.center = (centerX, centerY)
        self.centerCircle = Circle(centerX, centerY, self.avgR)

        #print("Before encircling rects, there were {} rects".format(len(self.rects)))
        
        self.rects = [
            rect for rect in self.rects \
                if (1-ENCIRCLE_RECT_DIST_DEV)*self.avgR < dist(rect.center, self.center) \
                < (1+ENCIRCLE_RECT_DIST_DEV)*self.avgR
        ]

        #print("After encircling rects, {} rects remain".format(len(self.rects)))

    def assignRadialVectors(self):
        assert hasattr(self, "center")
        for rect in self.rects:
            rect.assignVector(Vector(self.center, rect.center))

    def getRefVector(self):
        try:
            leftmostRect = min(
                self.rects, 
                key = lambda rect: angleDiffCW(rect.vector.angle, -pi/2)
            )
            rightmostRect = min(
                self.rects, 
                key = lambda rect: -1 * angleDiffCCW(rect.vector.angle, -pi/2)
            )

            if abs(angleDiff(leftmostRect.vector.angle + pi, rightmostRect.vector.angle)) < pi/4:
                leftmostRect.assignNumber(0)
                if not hasattr(rightmostRect, "number"):
                    rightmostRect.assignNumber(5)
                self.refAngle = angleDiffCCW(rightmostRect.vector.angle, leftmostRect.vector.angle) / 2 + rightmostRect.vector.angle - pi/2
            else:
                if abs(angleDiff(leftmostRect.vector.angle, pi)) < abs(angleDiff(rightmostRect.vector.angle, 0)):
                    self.refAngle = (leftmostRect.vector.angle - pi) % (2*pi)
                else:
                    self.refAngle = rightmostRect.vector.angle

            self.refVector = PolarVector(self.center, self.avgR, self.refAngle)
            
            
            self.rects = [rect for rect in self.rects \
                if not REF_VECT_MAX_CW_DIFF < angleDiffCW(self.refVector.angle, rect.vector.angle) \
                 < REF_VECT_MIN_CW_DIFF
            ]
                
        
        except AttributeError:
            print("Attempted to generate ref vector without first assigning vectors to rects")

    def getRectByNum(self, number):
        for rect in self.rects:
            if hasattr(rect, "number"):
                if rect.number == number:
                    return rect

    def draw(self, frame):
        for rect in self.rects:
            rect.draw(frame)
            #rect.vector.draw(frame)
            rect.intrinsicVector.draw(frame, color = BLUE)
            rect.intrinsicVector.drawEnd(frame)
            # relCircleCenter = shiftImageCoords(frame, rect.intrinsicVector.end)

            if hasattr(rect, "number"):
                cv2.putText(
                    frame,
                    #"{:.3f}".format(rect.aspectRatio),
                    "{:.3}".format(rect.contourArea / IMGAREA),
                    #"{},{}".format(*shiftImageCoords(frame, rect.intrinsicVector.end)),
                    (int(rect.center[0]), int(rect.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, ORANGE, 1
                )

        # for circle in gauntlet.circles:
        #     circle.draw(frame)

        if hasattr(self, "center"):
            self.centerCircle.draw(frame)
            #cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 3, RED, 2)
            cv2.putText(
                frame,
                "{:.2f}".format(self.centerCircle.r),
                (0,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ORANGE, 1)
        
        if hasattr(self, "refVector"):
            self.refVector.draw(frame, color = VIOLET)
            
    def serialWrite(self, img, serialObject):
        dataStr = "G"
        for rect in self.rects:
            coords = shiftImageCoords(img, rect.intrinsicVector.end)
            dataStr += "{},{};".format(*coords)
        dataStr += "\n"
        print(dataStr)
        serialObject.write(dataStr.encode("ascii", "ignore"))

hsvLower = (0,0,0)
hsvUpper = (255,200,255)
def preprocessFrame(frame):
    #assumes frame is an opencv image object
    
    #undistortedImg = undistort(frame)

    # hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsvMask = cv2.inRange(hsvImg, hsvLower, hsvUpper)
    # hsvFiltered = cv2.bitwise_and(hsvImg, hsvImg, mask = hsvMask)

    # _, _, greyImg = cv2.split(hsvFiltered)
    greyImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredImg = cv2.GaussianBlur(greyImg, (PREP_GAUSS_SIZE, PREP_GAUSS_SIZE), 0)
    #invThreshImg = cv2.threshold(blurredImg, BINARIZATION_THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]
    threshedImg = cv2.adaptiveThreshold(
        blurredImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, PREP_ADAPTIVE_BIN_SIZE, PREP_ADAPTIVE_EXPOSURE
    )
    #note: last two arguments change the adaptive behavior of this threshold.
    #the second last argument is the size of the sample to take to determine a mean to use
    #as the threshold value. lower tends to make the image look more "grainy".
    #last argument can be thought of as exposure. it is a constant that is subtracted from
    #each pixel before deciding if it goes to black or white. higher values make the image
    #appear more bright/whiter/washed out.

    #edgedImg = autoCanny(blurredImg)
    
    frame = cv2.cvtColor(greyImg, cv2.COLOR_GRAY2BGR)
    return threshedImg, blurredImg, frame

def findCircles(img):
    circles = []
    houghResults = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 
        HOUGH_ACCUM_RES, HOUGH_MIN_SEPARATION,
        param1 = 155,
        param2 = HOUGH_CIRCLE_THRESH,
        minRadius = HOUGH_MIN_R,
        maxRadius = HOUGH_MAX_R
    )
    if houghResults is not None:
        #print(houghResults)
        for (x,y,r) in houghResults[0,:]:
            if r == 0:
                continue
            circles.append(Circle(x,y,r))
    else:
        #print("No circles found")
        return None

    if len(circles) == 0:
        return None

    return circles

persMtx = cv2.getPerspectiveTransform(
    np.float32([ 
        [PERS_X_OFFSET, PERS_Y_OFFSET], 
        [IMGRES[0] - PERS_X_OFFSET, PERS_Y_OFFSET],
        [0, IMGRES[1]], 
        [IMGRES[0], IMGRES[1]] 
    ]),
    np.float32([ 
        [0, 0], 
        [IMGRES[0], 0], 
        [0, IMGRES[1]], 
        [IMGRES[0], IMGRES[1]] 
    ])
)
def undistortPerspective(img):
    return cv2.warpPerspective(img, persMtx, IMGRES)

def getContours(frame):
    return cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

def simplifyContour(contour, simplifyCoeff = POLY_APPROX_COEFF):
    perimeter = cv2.arcLength(contour, True)
    approxCnt = cv2.approxPolyDP(contour, simplifyCoeff*perimeter, True)
    return approxCnt

def getGauntlet(contours):
    rectangles = identifyContours(contours)
    try:
        
        # cv2.drawContours(frame, contours, -1, YELLOW, 2)
        # lines = cv2.HoughLinesP(edgedImg, 1, degToRad(1), 50, None, 50, 10)
        # for line in lines:
        #     line = line[0]
        #     vector = Vector((line[0], line[1]), (line[2], line[3]))
        #     vector.draw(frame, color=BLUE)
        gauntlet = Gauntlet(rectangles)
        gauntlet.encircleRects()
        gauntlet.assignRadialVectors()
        gauntlet.getRefVector()
        gauntlet.enumerateRects()
        for rect in gauntlet.rects:
            rect.getIntrinsicVector()

        return gauntlet, [rect.contour for rect in rectangles]
    except Exception:
        #print("Could not find gauntlet. Exception: ")
        #traceback.print_exc()
        return None, [rect.contour for rect in rectangles]

def identifyContours(contours):
    rectangles = []

    for contour in contours:
        approxCnt = simplifyContour(contour)
        if len(approxCnt) == 4:
            tapeObj = TapeRect(approxCnt)
            if all((
                RECT_MIN_AR < tapeObj.aspectRatio < RECT_MAX_AR,
                RECT_MIN_AREA_PCT < tapeObj.boundingBoxArea/IMGAREA < RECT_MAX_AREA_PCT,
                tapeObj.contourArea / tapeObj.boundingBoxArea > RECT_MIN_BBOX_FILL
            )):
                rectangles.append(tapeObj)

    #medianArea = np.median([rect.boundingBoxArea for rect in rectangles])
    # while len(rectangles) > 6:
    #     medianArea = np.median([rect.boundingBoxArea for rect in rectangles])
    #     rectangles.remove(
    #         max(rectangles, key = lambda rect: abs(rect.boundingBoxArea - medianArea))
    #     )

    return rectangles

def identifyTapeStrip(contours):
    largestCont = max(contours, key = lambda contour: cv2.arcLength(contour, True))
    contObj = TapePoly(largestCont)
    contObj.simplifyContour(simplifyCoeff = TAPE_STRIP_POLY_COEFF)
    reprAngle = pi
    reprCircle = Circle(0, TAPE_STRIP_MIN_Y, 3) # only start counting vertices in lower half of image

    for i in range(contObj.numVertices):
        prevPoint = contObj.contour[ (i-1) % contObj.numVertices ][0]
        thisPoint = contObj.contour[i][0]
        nextPoint = contObj.contour[ (i+1) % contObj.numVertices ][0]

        vec1 = Vector(thisPoint, prevPoint)
        vec2 = Vector(thisPoint, nextPoint)

        if any((
            vec1.magnitude < TAPE_STRIP_MIN_LINE_RATIO*IMGWIDTH,
            vec2.magnitude < TAPE_STRIP_MIN_LINE_RATIO*IMGWIDTH,
            isPointOffEdge(thisPoint, margin = TAPE_STRIP_POINT_MARGIN)
        )):
            continue
        
        if (TAPE_STRIP_MIN_ANGLE < abs(angleDiff(vec1.angle, vec2.angle)) < TAPE_STRIP_MAX_ANGLE
            and thisPoint[1] > reprCircle.y):
            reprAngle = abs(angleDiff(vec1.angle, vec2.angle))
            reprCircle = Circle(thisPoint[0], thisPoint[1], 10)

            if reprAngle > TAPE_STRIP_ANGLE_THRESH:
                contObj.assignDescriptor("T")
            else:
                contObj.assignDescriptor("Y")

    if reprAngle == pi:
        contObj = None
        reprCircle = None

    return contObj, reprCircle

def shiftImageCoords(img, coord, offset = DEFAULT_IMG_X_OFFSET):
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    return (int(round(coord[0] - imgWidth/2))+offset, int(round(imgHeight/2 - coord[1])))

def autoCanny(image, sigma=0.333):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged


def autoCannyUpper(image, sigma = 0.333):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    return int(min(255, (1.0 + sigma) * v))

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

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def isPointOffEdge(point, margin = 0):
    assert len(point) == 2
    return any((
            point[0] <= margin * IMGWIDTH,
            point[1] <= margin * IMGHEIGHT,
            point[0] >= (1-margin) * IMGWIDTH,
            point[1] >= (1-margin) * IMGHEIGHT,
        ))

if __name__ == "__main__":

    # originalImg = cv2.imread("gauntlet.jpg")
    # template = cv2.imread("template.png")
    # resizedImg = imutils.resize(originalImg, width=IMGWIDTH)
    # searchImg, dispImg = preprocessFrame(resizedImg)
    # resRect = findTemplate(searchImg, template)
    # resRect.draw(dispImg)
    
    # cv2.imshow("Frame", dispImg)
    # cv2.waitKey(0)

    import glob
    import traceback

    for img in glob.glob("Tape photos/*.jpg"):
        try:
            originalImg = cv2.imread(img)
            #originalImg = cv2.imread("gauntlet.jpg")
            resizedImg = imutils.resize(originalImg, width=IMGWIDTH)

            threshImg, houghImg, dispImg = preprocessFrame(resizedImg)
            contours = getContours(threshImg)
            tapeOutline, centerCircle = identifyTapeStrip(contours)
            tapeOutline.draw(dispImg)
            centerCircle.draw(dispImg)

            cv2.imshow("Frame", dispImg)
            cv2.waitKey(0)
        except OSError:
            traceback.print_exc()