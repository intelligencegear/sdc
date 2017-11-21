# **Finding Lane Lines on the Road**

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_intermediate_result/solidYellowCurve2_color_select.jpg "Color Selection"

[image2]: ./test_images_intermediate_result/solidYellowCurve2_gray.jpg "Gray"

[image3]: ./test_images_intermediate_result/solidYellowCurve2_blur_gray.jpg "Gassian Blur"

[image4]: ./test_images_intermediate_result/solidYellowCurve2_edges.jpg "Canny Edge"

[image5]: ./test_images_intermediate_result/solidYellowCurve2_masked_edges.jpg "Region Mask"

[image6]: ./test_images_intermediate_result/solidYellowCurve2_line_image.jpg "Hough Transform"

[image7]: ./test_images_intermediate_result/solidYellowCurve2_draw_image.jpg "Final Result"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps, please refer to fuction named `find_lane_pipline_fine_tune()` to see implementation details.

* Firstly, as shown below, I apply color selection technique to select only 'Yellow' and 'White' color in the image. In the challenge section, assuming that the lane lines are only yellow or white. I found that this strategy can increase the stability of the algorithm

  ![alt text][image1]

* Secondly, as shown below, I convet the image to grayscale using provided function `grayscale()`.

  ![alt text][image2]

* Thirdly, as shown below, I apply Gaussian smoothing algorithm to the grayscale image using provided function `gaussian_blur()`.

  ![alt text][image3]

* Fourthly, as shown below, I apply Canny edge detection algorithm to get edges of the image. I use the provided fuction `canny()` and use the threshold parameters used in the course quizzes.

  ![alt text][image4]

* Fifthly, as shown below, I apply region mask to the result of edge detection. I finally found a reasonable region after a series of debugging.

  ![alt text][image5]

* Sixthly, as shown below, I apply Hough Transform to the edge detected image. In order to draw a single line on the left and right lanes, I modified the draw_lines() function by
    1. Check the difference between x-coordinates of the endpoints of each straight line segment, to avoid the situation when the denominator is 0, to enhance the stability of the algorithm.
    2. Check the slope of each straight line segment and the x-coordinates of the endpoints when determining which segments belong to the left group or which segments belong to the right group.
    3. Then I use `np.polyfit()` to get left and right straight lines to draw on the original image.

  ![alt text][image6]

* Finally, as shown below, I draw the lines on the original image.

  ![alt text][image7]


### 2. Identify potential shortcomings with your current pipeline


* One potential shortcoming would be the result of detecting the lane line in the video is not very stable.

* Another shortcoming could be, in the challenge section, the lane line of a part of the the video was not recognized successfully, probably because of the shade. I adjusted several rounds of parameters, but failed. If there is a good solution to solve this problem, please let me know:)



### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use the difference method by using **slopes** of several previous frames belongs to the same video to improve the stability of the algorithm. But I am not sure it is a good idea.

Another potential improvement could be to use some tricks to handling the situation when car going across the shade.

### 4. The reason for the homework submission is overdue
Recently the company's work is more busy, I do not have enough time to learn and write code. Next time I will work hard to get homework not overdue.Thanks!
