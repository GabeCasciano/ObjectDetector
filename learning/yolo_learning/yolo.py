# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/ article was used as reference

import numpy as np
import argparse
import time
import cv2
import os


def init_arg_parser():
    #argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-y", "--yolo", required=True, help="base path to Yolo dir")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detection")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())
    return args

args = init_arg_parser()
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
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


#load image from path
image = cv2.imread(args["image"])
print("[INFO] image {}".format(args["image"]))
(H,W,_) = image.shape

#determine the output layer names we nede from YOLO
ln = net.getLayerNames()
ln = [ ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#construct blob from input image
blob = cv2.dnn.blobFromImage(image, 1.0/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.6f} seconds".format(end-start))

boxes = []
confidences = []
classIds = []

for output in layerOutputs:
    for detection in output:
        # extract the class id a probability from the current object detection
        print(detection)
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classIds]

        # remove weak predictions
        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            (centX, centY, width, height) = box.astype("int")

            # use the center to locate the bounding box
            x = int(centX - (width/2))
            y = int(centY - (height/2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIds.append(classId)

# Non maxima suppresion, remove all boxes that do not meet a threshold
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

if len(idxs) > 0:
    for i in idxs.flatten():
        # extract bounding box locations
        (x,y) = (boxes[i][0], boxes[i][1])
        (w,h) = (boxes[i][2], boxes[i][3])

        # draw bounding boxes
        color = [int(c) for c in COLORS[classIds[i]]]
        cv2.rectangle(image, (x,y), (w+x,y+h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIds[i]], confidences[i])
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# show image
cv2.imshow("Image", image)
cv2.waitKey(0)
