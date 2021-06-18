import cv2 as cv
import argparse
import os
import numpy as np
import time
from data_utils import get_arguments

class LPDetection(object):
	def __init__(self):
		self.args = get_arguments()
		self.weightsPath = os.path.sep.join([self.args["yolo"], "yolov4-tiny-obj_final.weights"])
		self.configPath = os.path.sep.join([self.args["yolo"], "yolov4-tiny-obj.cfg"])
		self.net = cv.dnn.readNetFromDarknet(self.configPath, self.weightsPath)

	def detect(self, image):
		np.random.seed(42)
		color = 230

		(H, W) = image.shape[:2]
		# determine only the *output* layer names that we need from YOLO
		ln = self.net.getLayerNames()
		ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)

		self.net.setInput(blob)
		layerOutputs = self.net.forward(ln)

		# initialize lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				# filter out weak predictions 
				if confidence > self.args["confidence"]:
					# scale the bounding box coordinates back relative to the
					# size of the image
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					# derive the top and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					# update list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv.dnn.NMSBoxes(boxes, confidences, self.args["confidence"],
			self.args["threshold"])

		coordinates = []
		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				coordinates.append((x, y, w, h))
				
		return coordinates