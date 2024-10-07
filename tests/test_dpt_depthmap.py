import cv2
from ultralytics import YOLO
import cv2
import sys


sys.path.append('./dpt_module/')


from utils.monodepth import load_model, depth_map_from_frame
# Load the model
dpt_model, transform, device = load_model()
import numpy as np


# Open an image frame using OpenCV (for example)
frame = cv2.imread('2024-10-06-145518.jpg')

# Get the depth map
depth_map = depth_map_from_frame(frame, dpt_model, transform, device)

# Display the depth map
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
