import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from binary_warp import binary_warp
from new_lane_line_search import new_line_search
from cal_curvature import cal_curvature
from draw_back import proj_back
from pipeline1 import pipeline1

        
vidcap = cv2.VideoCapture('../project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
with open('../params/camera_calibration.pkl','rb') as file:
	(cameraMatrix, distCoeffs) = pickle.load(file)
with open('../params/perspective.pkl','rb') as file:
	(M, Minv) = pickle.load(file)


while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  new_img = pipeline1(image, cameraMatrix, distCoeffs, M, Minv)
  cv2.imwrite("../frames1/frame" + "0" * (4-len(str(count))) + "{:d}.jpg".format(count), new_img)     # save frame as JPEG file
  count += 1
  # if count == 9:
  # 	break

