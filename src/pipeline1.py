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


def pipeline1(img, cameraMatrix, distCoeffs, M, Minv):
	img_size = (img.shape[1], img.shape[0])
	gray = binary_warp(img, cameraMatrix, distCoeffs)
	bird_view = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)


	(out_img, left_fit, right_fit) = new_line_search(bird_view)

	ploty = np.linspace(0, bird_view.shape[0]-1, bird_view.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	(left_cur, right_cur) = cal_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty)
	paint = proj_back(bird_view,left_fitx, right_fitx, ploty, Minv)
	# Combine the result with the original image
	new_img = cv2.addWeighted(img, 1, paint, 0.3, 0)
	curva = (left_cur + right_cur) / 2
	offset = 3.7/(right_fitx[-1] - left_fitx[-1]) * ((left_fitx[-1] + right_fitx[-1])/2 - 600)
	note1 = "Radius is: {:.1f} meters.".format(curva)
	note2 = "Offset is {:.2f} meters".format(offset)
	cv2.putText(new_img, note1, (300, 100), cv2.FONT_HERSHEY_PLAIN, 3, 2) #3, (0,0,0))
	cv2.putText(new_img, note2, (300, 150), cv2.FONT_HERSHEY_PLAIN, 3, 2) #3, (0,0,0))
	return new_img

if __name__ == '__main__':
	with open('../params/camera_calibration.pkl','rb') as file:
		(cameraMatrix, distCoeffs) = pickle.load(file)
	with open('../params/perspective.pkl','rb') as file:
		(M, Minv) = pickle.load(file)
	img = mpimg.imread('../test_images/test1.jpg')
	new_img = pipeline1(img, cameraMatrix, distCoeffs, M, Minv)

	f, (ax1, ax2) = plt.subplots(2,1,figsize=(8,8))
	ax1.imshow(img)
	ax2.imshow(new_img)
	plt.show()


# f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=(8,16))
# ax1.imshow(img)
# ax2.imshow(gray,cmap='gray')
# ax3.imshow(bird_view, cmap = 'gray')
# ax4.imshow(out_img)
# ax4.plot(left_fitx, ploty, color='yellow')
# ax4.plot(right_fitx, ploty, color='yellow')
# ax5.imshow(new_img)
# note = 'Radius is: {:.2f} meters. \n Offset is {:.2f} meters'.format((left_cur + right_cur) / 2, 
# 	3.7/(right_fitx[-1] - left_fitx[-1]) * ((left_fitx[-1] + right_fitx[-1])/2 - 600))
# ax1.text(500, 100, note)
# plt.show()



