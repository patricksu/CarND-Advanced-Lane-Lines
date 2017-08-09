## Report Writeup


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistortion_demo_peng.png "Undistorted"
[image2]: ./examples/undistortion_test1_before_after.png "Road Transformed"
[image3]: ./examples/binary_thresholded_test1_before_after.png "Binary thresholded Example"
[image4]: ./examples/perspective_transformed_before_after.png "Warp calibration"
[image5]: ./examples/perspective_transformed_before_after_test3.png "Warp example"
[image6]: ./examples/color_fit_lines_box.png "Fit Visual"
[image7]: ./examples/drawback_demo.png "Output"
[video1]: ./my_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, the original image and the undistorted image are shown below.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #110 through #113 in `src/binary_warp.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform uses cv2.warpPerspective(), which appears in lines 16 in the file `src/pipeline1.py`.I chose the hardcode the source and destination points in the following manner (in the file `src/perspective.py`):
```python
src = np.float32([[200, 720],[1130,720], [702, 460], [578, 460]])
dst = np.float32([[320, 720],[960,720], [960, 0], [320, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720      | 
| 1130, 720     | 960, 720      |
| 702, 460      | 960, 0        |
| 578, 460      | 320, 0        |

The perspective transformation is shown in the image below.
![alt text][image4]

I applied the transformation to another test image to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
In line 1 5and 16 of file `src/cal_curvature.py`, I used the detected left lane points and right lane points to fit a 2nd-order polynomial to find the lane lines. The left and right lane points are detected using the sliding window method as in the lectures,code lines 39 to 60 in the file `src/new_lane_line_search.py`.

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 15 through 19 in my code in `src/cal_curvature.py`. Using the formula given in the lecture, I calculated the radius in pixels and thenconver them to meters. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 13 through 34 in my code in `src/pipeline1.py` in the function `pipeline1()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./my_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1) Sometimes very few lane points are detected and the fitted lines are way off. I tuned the Sobel and color thresholds to get as much points as possible, but in shadow or white road conditions, the problem still exists. I used a smoothing technique to make it work. I.e. I keep track of the previous 5 frames' fitted lines, and if the new lines are not parallel, I simply use the historical lines instead of the new one. 
This pipeline will fail in urban traffic conditions, or in road segments when lane lines fade.
If I will pursue this project further, I am gonna combine neural network with this technique for better results. 
