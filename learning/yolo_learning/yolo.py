import numpy as np
import argparse
import time
import cv2
import os

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", required=True, help="base path to Yolo dir")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detection")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

#load the paths for the labels, weights and config
labelspath = os.path.sep.join([args["yolo"], "coco.names"])
weightspath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configpath = os.path.sep.join([args["yolo"], "yolo.cfg"])

#generate the list of labels
LABELS = open(labelspath).read().strip().split("\n")

#generate the list of colors to represent each label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#load darknet config and weights
print("[INFO] Loading YOLO from disk ...")
net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

#load image from path
image = cv2.imread(args["image"])
(H,W) = image.shape[:2]

#determine the output layer names we nede from YOLO
ln = net.getLayerNames()
ln = [ ln[ i[0] - 1 ] for i in net.getUnconnectedOutLayers() ]

#construct blob from input image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=false)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took [:.6f] seconds".format(end-start))
