import numpy as np
import argparse
import cv2 as cv

def get_arguments():
    parser = argparse.ArgumentParser(description='object detection')
    parser.add_argument('-i', '--image', help='path to image', default='test.jpg')
    parser.add_argument('-y', '--yolo', help='path to yolo directory', default='./yolo')
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")

    return vars(parser.parse_args())
    
    