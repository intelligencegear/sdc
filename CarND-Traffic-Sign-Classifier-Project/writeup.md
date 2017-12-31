# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./assets/histogram.png "Visualization"
[image2]: ./assets/loss_accuracy.png "Loss VS Accuracy"
[image3]: ./assets/ten_signs_from_web.png "10 Signs From Web"
[image4]: ./assets/ten_signs_from_web_prediction.png "Predicitons of 10 Signs From Web"
[image5]: ./assets/top_five_predictions.png "Top Five Predicitons"
[image6]: ./assets/conv_0.png "conv_0"
[image7]: ./assets/conv_relu.png "conv_relu"
[image8]: ./assets/conv_max_pool.png "conv_max_pool"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **= (32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The histogram below shows how the training set, validation set, and test set data are distributed among the 43 different traffic classifications.

From the figure we can see that the three sets of data roughly belong to the same distribution.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


I have experimented with 2 methods for data preprocessing that may be not working for my classification model:
* I have experimented with normalizing the data from [0,255] to [0,1] or [-1,1], but the correctness of the validation set did not improve.

* I have experimented with data augmentation by adjusting the brightness of the picture, rotate the picture within a certain range, but the correctness of the validation set also did not improve.

At last, in the final version of the code, I removed both above methods, because they increased the time complexity, but did not bring any benefits for the classification model.

Also, I have test shuffle technique in model training process, it did have been proven a very useful technique for enhance the accuracy of validation set.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					        |										                |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6 			|
| Dropout               | keep_prob=0.9                     |
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					        |										                |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16 			|
| Dropout	      	      | keep_prob=0.9 			              |
| Fully connected		    | Inputs 400, outputs 120        	  |
| RELU					        |										                |
| Dropout               | keep_prob=0.5                     |
| Fully connected		    | Inputs 120, outputs 84        	  |
| RELU					        |										                |
| Dropout               | keep_prob=0.5                     |
| Softmax				        | Inputs 84, outputs 43        			|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model:

* I used an an Adam optimizer with learning rate equals **0.001**.

* I used batch size equals **128**

* I used **300** training epochs and find the best fit round in the validation set.

* I used L2 regularization with lambda coefficient equals **0.01**.

* I used Dropout after every convolution layer, fully connected layer. We can see the detail of **parameters** from the table above.

* I used **Xavier initializer** to initialize the model paramters.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **99.3%**
* validation set accuracy of **97.7%**
* test set accuracy of **95.4%**

![alt text][image2]

I used the well known LeNet-5 architecture with L2-Regularization and Dropout for the traffic sign classification problem since it is has been proven although very simple but very effective for computer vision applications.

In the training process:

* Firstly, I plotted the **Loss VS Accuracy** changing trends of training process, as shown above, to make sure the classification model is not over-fitting or under-fitting. In my opinion, this step was *very effective* for fine tunning models.

* Secondly, I use **Xavier initializer**, **Adam optimizer** and **L2-Regularization** and tune learning rate and lambda coefficient parameters. The first version model, through cross validation result, I believe the model is over-fitting. After several round of parameters tunning processing, through cross validation technique, I already got the accuracy of *meeting requirements in validation set*.

* Thirdly, I add **dropout** layers after every convolution layer, fully connected layer. Through cross validation technique, I got final *keep probabilities* for every dropout layer. This step gave a *huge improvement* and increased the accuracy of validation set *at least 3 points*.

* Finally, I got the number of epochs through cross validation and picked the best fitting one as the final classification model.

Through the accuracy of the training set, the verification set and the test set, we may consider that, although there are many aspects to improve, the final model has neither over-fitting nor under-fitting, and the final classification model shows a fairly well performance for this problem.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![alt text][image3]

The 1-st image might be easy to classify.

The 2-nd image might be difficult to classify because it not very clear.

The 3-rd image might be difficult to classify because it not very clear.

The 4-th image might be difficult to classify because it not very clear.

The 5-th image might be difficult to classify because it not very clear.

The 6-th image might be easy to classify.

The 7-th image might be difficult to classify because it is very blurred.

The 8-th image might be difficult to classify because it is not clear and its background is noisy.

The 9-th image might be easy to classify.

The 10-th image might be difficult to classify because it is very blurred.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image4]


| Image			        |     Prediction	        					| Correct |
|:---------------------:|:---------------------------------------------:|:----:|
| Going Straight or Left      		| Going Straight or Left  	  | Y |
| No passing for vehicles over 3.5 metric tons     			| No passing for vehicles over 3.5 metric tons									| Y |
| Speed limit (100km/h)					| Speed limit (100km/h)        | Y |
| End of all speed and passing limits	      		| *Vehicles over 3.5 metric tons prohibited* |		N		 				|
| Bumpy road			| Bumpy road | Y |
| Go straight or right| Go straight or right | Y |
| Pedestrians | Pedestrians | Y |
| Double curve | *Dangerous curve to the left* | N |
| Dangerous curve to the right | Dangerous curve to the right | Y |
| Speed limit (20km/h) | Speed limit (20km/h) | Y |

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. The accuracy of the model on the test set is much higher (95.4%) than the accuracy of this data set collected from web, the main reason may depend on the following two aspects:
  * The web data set only contains 10 images, thus the accuracy given by this data set may not be trusted.

  * The images that predicting incorrect is very unclear and also hard to classified by human beings.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the *18-th* cell of the Ipython notebook. All top five predictions of web images is shown in below image:

![alt text][image5]

The details of top five predictions belongs to the fist five images is provided below:

1. For the 1-st image, the model is relatively sure that this is a *Going Straight or Left* (probability of 0.964), and the image does contain '*Going Straight or Left*'. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .964         			| Going Straight or Left  									|
| .012     				| Keep left 										|
| .007					| General caution											|
| .003	      			| Turn right ahead					 				|
| .002				    | Ahead only     							|


2. For the 2-nd image, the model is relatively sure that this is a *No passing for vehicles over 3.5 metric tons* (probability of 1.000), and the image does contain '*No passing for vehicles over 3.5 metric tons*'. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.000         			| No passing for vehicles over 3.5 metric tons  									|
| .000     				| Speed limit (80km/h) 										|
| .000					| End of no passing by vehicles over 3.5 metric tons											|
| .000	      			| No passing					 				|
| .000				    | Speed limit (100km/h)     							|

3. For the 3-rd image, the model is relatively sure that this is a *Speed limit (100km/h)* (probability of 0.957), and the image does contain '*Speed limit (100km/h)*'. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .957         			| Speed limit (100km/h)							|
| .038     				| Speed limit (120km/h) 										|
| .002					|  Roundabout mandatory											|
| .001	      			| Speed limit (80km/h)					 				|
| .001				    | Beware of ice/snow     							|


4. For the 4-th image, the prediction of this model is *incorrect*, the image does contain '*End of all speed and passing limits*'. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.000         			| Vehicles over 3.5 metric tons prohibited	|
| .000     				| No passing										|
| .000					|  End of no passing											|
| .000	      			| Roundabout mandatory					 				|
| .000				    | End of no passing by vehicles over 3.5 metric tons   |


5. For the 5-th image, the model is relatively sure that this is a *Bumpy road* (probability of 0.925), and the image does contain '*Bumpy road*'. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .925         			| Bumpy road							|
| .016     				| Bicycles crossing 										|
| .015					|  Road work											|
| .014	      			| Traffic signals					 				|
| .003				    | No vehicles     							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

* Convolution layer: there are some blurred shapes may be related to traffic signs.

![alt text][image6]

* Relu layer: background noise has been significantly reduced and important shapes are kept.

![alt text][image7]

* Max pooling layer: although image size is highly reduced, the useful shapes for classification are still available.

![alt text][image8]
