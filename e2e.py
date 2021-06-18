import cv2 as cv
import os
import numpy as np
from lp_detection import LPDetection
from character_segmentation import CharacterSegmentation
from data_utils import get_arguments
from imutils import perspective
from cnn import CNN


class E2E(object):
    def __init__(self):
        self.lp_detect = LPDetection()
        self.char_seg = CharacterSegmentation()
        self.cnn = CNN()

    def process(self, image):
        x,y,w,h = self.lp_detect.detect(image)[0]
        pts = np.float32([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
        lp_area = perspective.four_point_transform(image, pts)

        char_array = self.char_seg.get_char(lp_area)

        text = ""
        for char in char_array:
            text += self.cnn.get_char_name(char)

        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
			0.5, 230, 2)
        
        cv.imshow("anh du doan", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
