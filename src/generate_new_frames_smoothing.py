import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from binary_warp import binary_warp
from new_lane_line_search import new_line_search
from lane_line_search import line_search
from cal_curvature import cal_curvature
from draw_back import proj_back
from pipeline1 import pipeline1

class Line():
    def __init__(self):
        #
        # num of non-detected lines in the last iterations
        self.missed = 0  
        # the last n good fits of the line
        self.recent_fits = [] 
        self.best_fit = None  
        self.radius_of_curvatures = []
        self.radius_of_curvature = None 
        self.offsets = []
        self.offset = None

def sanity_check(left_fit, right_fit, cur, cur_best, offset, offset_best, ploty):
  if cur_best == None or offset_best == None:
    return 0
  # if parallel?
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
  if np.std(left_fitx - right_fitx) * 3.7 / 640 > parallel_thresh:
    return 1  #meaning not parallel
  print(cur)
  print(cur_best)
  if np.abs(cur - cur_best) > cur_thresh:
    return 2  #meaning either left_cur or right_cur deviates too much
  if np.abs(offset - offset_best) > offset_thresh:
    return 4 #meaning the offset diff is too much
  return 0  #meaning sanity-check is passed

      
vidcap = cv2.VideoCapture('../project_video.mp4')
count = 0
success = True
with open('../params/camera_calibration.pkl','rb') as file:
	(cameraMatrix, distCoeffs) = pickle.load(file)
with open('../params/perspective.pkl','rb') as file:
	(M, Minv) = pickle.load(file)

success,img = vidcap.read() # assuming there is at least one frame in the video
leftLine = Line()
rightLine = Line()

N = 5
n = 0
parallel_thresh = 1.5       # parallel threshold in meters STD
cur_thresh = 1000            # meters
hori_dist_thresh = 1      # horizontal distance in meters
offset_thresh = 0.5         # offset in meters

success,img = vidcap.read()
while success:
  img_size = (img.shape[1], img.shape[0])
  gray = binary_warp(img, cameraMatrix, distCoeffs)
  bird_view = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)
  if count == 0 or leftLine.missed >= N or rightLine.missed >= N:
    (_, left_fit, right_fit) = new_line_search(bird_view)
  else:
    (left_fit, right_fit) = line_search(bird_view, leftLine.best_fit, rightLine.best_fit)

  ploty = np.linspace(0, bird_view.shape[0]-1, bird_view.shape[0] )
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
  offset = 3.7/(right_fitx[-1] - left_fitx[-1]) * ((left_fitx[-1] + right_fitx[-1])/2 - 600)
  (left_cur, right_cur) = cal_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty)
  cur = (left_cur + right_cur) / 2

  leftLine.missed = leftLine.missed + 1
  rightLine.missed = rightLine.missed + 1
  if len(leftLine.radius_of_curvatures) >= N:
    leftLine.radius_of_curvatures = leftLine.radius_of_curvatures[-(N-1):]
  leftLine.radius_of_curvatures.append(cur)
  leftLine.radius_of_curvature = np.mean(leftLine.radius_of_curvatures)

  if len(leftLine.offsets) >= N:
    leftLine.offsets = leftLine.offsets[-(N-1):]
  leftLine.offsets.append(offset)
  leftLine.offset = np.mean(leftLine.offsets)


  result = sanity_check(left_fit, right_fit, cur, leftLine.radius_of_curvature, offset, leftLine.offset, ploty)
  if result != 0:
    left_fit = leftLine.best_fit
    right_fit = rightLine.best_fit
    cur = leftLine.radius_of_curvature
    offset = leftLine.offset
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
  else:
    leftLine.missed = 0
    rightLine.missed = 0
    if len(leftLine.recent_fits) >= N:
      leftLine.recent_fits = leftLine.recent_fits[-(N-1):]
    if len(rightLine.recent_fits) >= N:
      rightLine.recent_fits = rightLine.recent_fits[-(N-1):]
    leftLine.recent_fits.append(left_fit)
    rightLine.recent_fits.append(right_fit)
    leftLine.best_fit = np.mean(leftLine.recent_fits, 0)
    rightLine.best_fit = np.mean(rightLine.recent_fits, 0)


  paint = proj_back(bird_view,left_fitx, right_fitx, ploty, Minv)
  # Combine the result with the original image
  new_img = cv2.addWeighted(img, 1, paint, 0.3, 0)
  note1 = "Radius is: {:.1f} meters.".format(cur)
  note2 = "Offset is {:.2f} meters".format(offset)
  cv2.putText(new_img, note1, (300, 100), cv2.FONT_HERSHEY_PLAIN, 3, 2) #3, (0,0,0))
  cv2.putText(new_img, note2, (300, 150), cv2.FONT_HERSHEY_PLAIN, 3, 2) #3, (0,0,0))
  print(note1)
  print(note2)
  cv2.imwrite("../frames1.2/frame" + "0" * (4-len(str(count))) + "{:d}.jpg".format(count), new_img)     # save frame as JPEG file
  count += 1
  success,img = vidcap.read()
  print('Read a new frame: ', success)




