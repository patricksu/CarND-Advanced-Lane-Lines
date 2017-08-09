import numpy as np
import cv2
import pickle

src = np.float32([[200, 720],[1130,720], [702, 460], [578, 460]])
dst = np.float32([[320, 720],[960,720], [960, 0], [320, 0]])
# src = np.float32([[203, 720],[1127,720], [695, 460], [585, 460]])
# dst = np.float32([[320, 720],[960,720], [960, 0], [320, 0]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

with open('../params/perspective.pkl','wb') as file:
	pickle.dump((M, Minv), file)