import cv2 as cv
import argparse
from e2e import E2E

def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image', help='path to image', default='test.jpg')

    return arg.parse_args()

args = get_arguments()

image = cv.imread(args["image"])
e2e = E2E()
e2e.process(image)