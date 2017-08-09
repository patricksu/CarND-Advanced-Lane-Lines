import numpy as np
import cv2

# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!

# image = mpimg.imread('../test_images/test1.jpg')
# undist = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)
# img_size = image.shape[:2][::-1]
# color, gray = pipeline(undist, x_thresh = (20, 100), y_thresh = (20, 100), mag_thresh = (50, 100), dir_thresh=(.7, 1.3), color_thresh=(160, 255))
# binary_warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)

def line_search(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    nleft_fit = np.polyfit(lefty, leftx, 2)
    nright_fit = np.polyfit(righty, rightx, 2)
    return (nleft_fit, nright_fit)