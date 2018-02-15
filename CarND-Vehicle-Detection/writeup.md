## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog1]: ./output_images/hog_features_ychannel_1.png
[hog2]: ./output_images/hog_features_ychannel_2.png
[hog3]: ./output_images/hog_features_ychannel_3.png
[hog4]: ./output_images/hog_features_ychannel_4.png

[color_vis1]: ./output_images/color_hist_features_1.png
[color_vis2]: ./output_images/color_hist_features_2.png

[spatial_vis1]: ./output_images/spatial_feaures_1.png
[spatial_vis2]: ./output_images/spatial_feaures_2.png

[hog_vis1]: ./output_images/hog_features_1.png
[hog_vis2]: ./output_images/hog_features_2.png

[pred_vis1]: ./output_images/model_predition_1.png
[pred_vis2]: ./output_images/model_predition_2.png

[sliding_window1]: ./output_images/subsampling_window_search_1.png
[sliding_window2]: ./output_images/subsampling_window_search_2.png

[search_result1]: ./output_images/search_result_1.png

[debug_video]: ./output_images/test_video_output.gif

[heatmap]: ./output_images/heatmap.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook(`./SVCClassiferPipeline.ipynb`). The implementation code is as follows:

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, block_norm='L1', vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec, block_norm=block_norm)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm=block_norm)
        return features
```




I started by reading in all the `vehicle` and `non-vehicle` images.  I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are some examples using the `Y-Channel` of `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][hog1]
![alt text][hog2]
![alt text][hog3]
![alt text][hog4]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally chose the parameters of HOG as below:

```python
orient=9
pix_per_cell=(8, 8)
cell_per_block=(2, 2)
hog_channel='ALL' # All color channels of YCrCb color space
block_norm='L1'
```

Moreover, I also employed color histogram features and spatial features which parameters is shown as below:

```python
hist_bins=32
spatial_size=(32, 32)
```
This parameter combinations were obtained by cross-validation technique, which would be described in more detail in the next section.

Also, I have done some visualizations of these features as mentioned above:

* **Color Histogram Features Exploration**

  ![alt text][color_vis1]
  ![alt text][color_vis2]


* **Spatial Features Exploration**

  ![alt text][spatial_vis1]
  ![alt text][spatial_vis2]

* **HOG Features Exploration**

  ![alt text][hog_vis1]
  ![alt text][hog_vis2]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 9th code cell of the IPython notebook(`./SVCClassiferPipeline.ipynb`).

* First of all, A Linear SVM model is a fairly nice choice for this object detection problem since the trade-off of between the performance and speed. Thus there was a only one additional *hyper parameter* `C` to be optimized through cross-validation.

* Secondly, I split my image data into a training and validation set:

  ```python
  # Split up data into randomized training and test sets
  rand_state = np.random.randint(0, 100)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
  ```

* Finally, I trained a linear SVM by using `RandomizedSearchCV` to find a reasonable `C` and test the performance in the validation set:

  ```python
  # Use a linear SVC
  svc = LinearSVC()
  parameters = {
      'C': sp_randint(1, 100)
  }

  # Check the training time for the SVC
  t=time.time()
  # svc.fit(X_train, y_train)
  clf = RandomizedSearchCV(svc, parameters, n_jobs=6, cv=3, verbose=100, n_iter=20)
  clf.fit(X_train, y_train)
  t2 = time.time()
  ```

After several rounds of cross-validation, eventually one combination of feature parameters together with `C` of the Linear SVM model was found, which performed well on the test data set.

Here are some predictions provided by the linear SVM model:

![alt text][pred_vis1]

![alt text][pred_vis2]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented `HOG sub-sampling window search` technique to explore the HOG features more efficiently compared to the `basic sliding window search`. The code is contained in the 8th code cell of the IPython notebook(`./SVCDetectionPipeline.ipynb`).

This fast method only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of `1` would result in a window that's `8 x 8` cells then the overlap of each window is in terms of the cell distance. This means that a `cells_per_step = 2` would result in a search window overlap of `75%`.

Also, It is possible to run this same function multiple times for different scale values to generate multiple-scaled search windows. After several plotting of the test images as shown below, I finally picked two search scales:


``` python
# Top center view for searching the far vehicles
ystart=400
ystop=464
xstart=384
xstop=640
scale=1

# Large view for searching the near vehicles
ystart=400
ystop=592
xstart=0
xstop=1280
scale=1.5
```
![alt text][sliding_window2]

![alt text][sliding_window1]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is one  example image:

![alt text][search_result1]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

I used the [test video](./test_video.mp4) to test my *car vs noncar* classifier and the sliding window search algorithm. Here's a [link to my debug video result](./output_videos/test_video_output.mp4). The left side of the output video of the `test_video.mp4` is final result of the detection pipeline, and the right side is the result of the sliding window search.

![alt text][debug_video]

**Here's a [link to my final video result](./output_videos/project_video_output.mp4). Please check:)**

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a test image provided by this project, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on this test image:

![alt text][heatmap]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The `heatmap` method could give a pretty good result, however, the bounding boxes in the video was not very stable. Inspired by the last project of lane lines detections, the result of current frame and the previous 4 frames are exponentially weighted averaged. This simple strategy could lead a more robust result.

However, the classifier is not very accurate and the feature extraction was very slow to be deployed onto the real vehicles. If there were more time to explore this project, I would do more investigations in object detection and try some deep learning algorithms, such as YOLO or Faster-RCNN algorithms.
