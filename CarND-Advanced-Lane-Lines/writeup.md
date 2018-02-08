## **Advanced Lane Finding Project**

---

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

[image1]: ./output_images/camera_calibration_1.png "Undistorted Image"
[image2]: ./output_images/camera_calibration_2.png "Undistorted Image"
[image3]: ./output_images/undistortion.png "Road Transformed"
[image4]: ./output_images/binary.png "Binary Example"
[image5]: ./output_images/warped_1.png "Warped Example"
[image6]: ./output_images/warped_2.png "Warped Example"
[image7]: ./examples/color_fit_lines.jpg "Fit Visual"
[image8]: ./output_images/result.png "Result"
[image9]: ./output_images/test_pipeline_1.png "Test Pipeline"
[image10]: ./output_images/test_pipeline_2.png "Test Pipeline"
[image11]: ./output_images/test_pipeline_3.png "Test Pipeline"
[image12]: ./output_images/test_pipeline_4.png "Test Pipeline"
[image13]: ./output_images/original.png "Original Example"
[video1]: ./output_videos/project_detection_debug_video.mp4 "Debug Video"
[video2]: ./output_videos/project_detection_video.mp4 "Detection Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in IPython notebook located in `camera_calibration.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this results:

![alt text][image1]

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I apply the distortion correction to one of the test images:

![alt text][image13] ![alt text][image3]

The first image is the original image and the second one is the undistortion version of the same image. We can see that the corner of the picture had been corrected.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps contained in 3rd cell of IPython notebook located in `finding_lanelines.ipynb`. Here's an example of my output for this step.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 10 through 27 in the 2nd code cell of the IPython notebook named `finding_lanelines.ipynb`.  

The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
## Perspective transform
def perspective_transform(img, vertices=[[578, 460], [705, 460], [1120, 720], [190, 720]], offset=300):        
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(vertices)
    dst = np.float32([[offset, 0],
                      [img_size[0]-offset, 0],
                      [img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]]])

    # Given src and dst points, calcula  te the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, M_inv
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 578, 460      | 300, 0        |
| 190, 720      | 300, 720      |
| 1120, 720     | 980, 720      |
| 705, 460      | 980, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this(the code contained in lines 69 through 71 in the 4th code cell of the IPython notebook named `finding_lanelines.ipynb`):

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 139 through 162 in the 4th code cell of the IPython notebook named `finding_lanelines.ipynb`


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 164 through 194 in my code in the function `apply_inverse()` which appears in the 4th code cell my IPython notebook named `finding_lanelines.ipynb`.  Here is an example of my result on a test image:

![alt text][image8]

Also, I have implemented a function called `test pipeline()` in the 10th code cell in the IPython notebook named `finding_lanelines.ipynb`. This test function can demonstrate the whole processing of this project. Here are some examples of the test images:

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my debug video result, which includes main processing steps in the pipeline](./output_videos/project_detection_debug_video.mp4)

![alt text][video1]

Here's a [link to my final video result](./output_videos/project_detection_video.mp4)

![alt text][video2]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

###  

My first version of the pipeline almost failed when the light condition is strong or the car passed through the shade, especially when there were a lot of stains on the pavement.

I extracted the pictures one by one from the hard part of the project video. Then I used the pipeline code to process these image respectively. After a careful analysis, I found that I could employ a more smart **color and gradient threshold** to generate the binary image. Finally, I used a combination of
* Sobel-x graident
* HLS  color threshold
* Red channel threshold
* Gray channel threshold
* Yellow color range in HSV color spaces
* White color range in RGB color spaces

and got a more robust version of the processing pipeline.

Further more, I used a simple sanity checking strategy to give more robust result, which incluing checking whether the lane line were separated by approximately the right distance horizontally, whether they had similar curvature, whether were roughly parallel and whether the vehicle were too far from the center of the pavement.


If I were going to pursue this project further, the following aspects might be very important:

* Machine learning based pipeline to segment the lane lines from the original images may generalize more better than the hand crafted rules.

* More smart sanity checking and exception handling strategy should be developed to make the detection pipeline more robust.

* More reliable version of perspective transformation should be employed.
