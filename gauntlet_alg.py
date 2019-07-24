import cv2
import numpy as np
import imutils
import statistics

IMGWIDTH = 1200

class PieceOfTape:
    def __init__(self, contour):
        self.contour = contour
        self.boundingBox = cv2.minAreaRect(contour)
        self.boundingBoxContour = np.int0(cv2.boxPoints(self.boundingBox))
        self.center = self.boundingBox[0]
        self.dims = self.boundingBox[1]
        self.angle = self.boundingBox[2]
        self.boundingBoxArea = self.dims[0] * self.dims[1]
        self.aspectRatio = round(max(self.dims) / min(self.dims), 3)


def getGauntlet(frame):
    #assumes frame is an opencv image object

    greyImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredImg = cv2.GaussianBlur(greyImg, (5,5), 0)
    threshedImg = cv2.threshold(blurredImg, 127, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(threshedImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    rectangles = identifyContours(contours)

    for rect in rectangles:
        cv2.drawContours(frame, [rect.boundingBoxContour], -1, (0,255,0), 2)
        cv2.putText(
            frame,
            str(rect.aspectRatio),
            (int(rect.center[0]), int(rect.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255,255,0), 2
        )

    return frame

def identifyContours(contours):
    rectangles = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        approxCnt = cv2.approxPolyDP(contour, 0.05*perimeter, True)
        if len(approxCnt) == 4:
            tapeObj = PieceOfTape(approxCnt)
            if 1.5 < tapeObj.aspectRatio < 2.5:
                rectangles.append(tapeObj)

    medianArea = statistics.median([rect.boundingBoxArea for rect in rectangles])
    while len(rectangles) > 6:
        rectangles.remove(
            max(rectangles, key = lambda rect: abs(rect.boundingBoxArea - medianArea))
        )

    return rectangles

originalImg = cv2.imread("7.jpg")
resizedImg = imutils.resize(originalImg, width=IMGWIDTH)

cv2.imshow("Frame", getGauntlet(resizedImg))
cv2.waitKey(0)