import numpy as np
import argparse
import time
import cv2
import os

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("")