import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imutils import perspective
from lp_detection import LPDetection
from data_utils import get_arguments

class CharacterSegmentation(object):
    def __init__(self):
        self.licensePlate = LPDetection()
        self.candidates = []

    def get_char(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        output = cv.connectedComponentsWithStats(thresh, 4, cv.CV_32S)
        (numLabels, labels, stats, centroids) = output
        mask = np.zeros(gray.shape, dtype="uint8")
        
        # loop over the number of unique connected component labels
        characters = []
        for i in range(1, numLabels):
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            area = stats[i, cv.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            # ensure the width, height, and area are all neither too small
            # nor too big
            keepWidth = w > 1 and w < 30
            keepHeight = h > 18 and h < 30
            keepArea = area > 18 and area < 900
            
            # ensure the connected component we are examining passes all
            # three tests
            if all((keepWidth, keepHeight, keepArea)):
                im = image[y:y+h, x:x+w]
                componentMask = (labels == i).astype("uint8") * 255
                img = componentMask[y:y+h, x:x+w]
                img = cv.resize(img, (28, 28), cv.INTER_AREA)
                img = img.reshape((28, 28, 1))
                characters.append(img)
        return characters            
