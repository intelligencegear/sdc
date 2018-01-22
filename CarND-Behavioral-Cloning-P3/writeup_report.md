# **Behavioral Cloning**

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./nvidia_model.png "Nvidia Model Visualization"
[image2]: ./lenet5_model.png "LeNet-5 Model Visualization"
[image3]: ./skewed_distribution.png "Data distribution"
[image4]: ./augmentation.png "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create the model and data preprocessing.
* **end2end_mdoel_demo.ipynb** containing the script to train the model.
* **model_visualization.ipynb** containing the script to plot the model.
* **drive.py** for driving the car in autonomous mode.
* **nvidia_model.h5** containing a trained convolution neural network using Nvidia end to end model.
* **lenet5_model.h5** containing a trained convolution neural network using LeNet-5 model.
* **nvidia_run.mp4** is video recording when in autonomous mode using trained Nvidia model.
* **lenet5_run.mp4** is video recording when in autonomous mode using trainde LeNet-5 model.
* **writeup_report.md** for summarizing the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my **modified** drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py nvida_model.h5
```
**or**

```sh
python drive.py lenet5_model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
In this project, I have implemented two models, including Nvidia end to end model and the LeNet-5 model:

* **Nvidia model**

  My model consists a series of convolutional layers, which including first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers(model.py lines 182-186).  

  The model includes ELU(Exponential Linear Units) layers to introduce nonlinearity(model.py 182-186) , and data is cropped, normalized and resized in model using Keras lambda layer(model.py lines 173-179).

* **LeNet-5 model**

  My model consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 16 (model.py lines 258-263).

  The model includes RELU layers to introduce nonlinearity(model.py lines 258-263),  and data is cropped, normalized and resized in model using Keras lambda layer(model.py lines 249-255).

  Since **LeNet-5 model** is very common and standard, for convenience, we will only describe the **Nvidia model** in deitals when there is no significant difference between the two models.

#### 2. Attempts to reduce overfitting in the model

The **Nvidia model** contains one dropout layer in order to reduce overfitting(model.py line 190).

The **Lenet-5 model** contains two dropout layers in order to reduce overfiting(mode.py lines 269, 272)

Although **Nvidia model** make the car drive more smoothly, however, the **Lenet-5 model** could also make the car stay on the track.

The models was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 221, 298). The models was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

* **Learning rate**

  The models used an adam optimizer, however, as Paul Heraty suggested, I used a 1e-4 as the init learning rate(model.py line 200) in **Nvidia model** training.

* **Early Stopping Strategy**

  I used early stopping strategy and 20 as the maximum of epochs when training. In every epoch the model was evaluated by the validation data set, the best one would be saved through Keras callbacks.

* **Model initialization**

  The convolutional layer is initialized by *normal* and the fully connected layer is initialized by *lecun_uniform*.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. While I have not joystick and the quality of data collected by myself is very hard to guarantee, I just download the data set provided by Udacity.

I used a combination of recorded **images by the center camera**, and recorded **images by the left and right cameras** which can simulate recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As mentioned above, I have implemented two models for this project, including **Nvidia model** and **LeNet-5 model**. These two pipelines is not too complex when using Keras to build up. Compared with the original models, only a few extra dropout layer was added in my implementation. The dropout and early stopping strategy are very common techniques that can help the model avoid the overfitting.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set using tools provided by sklearn package(model.py 221, 298). By the way, the image data in training is augmented while the validation is not, the error of training is little more larger than that of validation since the augmentation was not very perfect compared with the original distribution.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when the car went through in some sharp turn. To improve the driving behavior in these cases, I make a little larger **the correction of images recorded by left and right cameras**(model.py lines 56-58). After several experiments, I chose **0.25** for Nvidia model and **0.5** for LeNet-5 model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

* **Nvidia model**

  The Nvidia model is shown as below. First of all, before passing the images into the network, the images should be converted from RGB color spaces into YUV spaces, just as what did in "End to End Learning for Self-Driving Cars". The network consists of 12 layers(except for the input, dropout, flatten, activation layers):

  * 3 lambda layers which are used to crop, noarmlize and resize the original image to match the input format described by Nvidia paper.

  * 5 convolutional layers which including first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

  * 4 fully connected layers that designed to function as a controller for predicting steering angle.

  ![alt text][image1]

* **LeNet-5 model**

  The LeNet-5 model is shown as below. As mentioned above, the images should be converted from RGB color spaces into YUV spaces. The network actually consists 8 layers(except for the input, dropout, flatten, activation, pooling layers):

  * 3 lambda layers which are used to crop, noarmlize and resize the original image to match the input format for modified LeNet-5(32×32×3).

  * 2 convolutional layers with 5×5 filter sizes and depths between 6 and 16.

  * 3 fully connected layers that designed to function as a controller for predicting steering angle.

  ![alt text][image2]

#### 3. Creation of the Training Set & Training Process

First of all, it is important to emphasize that the creation of a data set is not as easy as imagined. The most important thing in my opion is the image format of training and predicting should be same with each other. However, after tuning of several days, I finally figured out that the **drive.py** gave the picture is in RGB color spaces, while, the image read by **cv2** is in BGR color spaces by default.


Since I have not joystick and the quality of data collected by myself is very hard to guarantee, I just download the data set provided by Udacity to avoid something like *garbage in then garbage out*. After simply plotting, as shown as below, I found that the data distribution is very skewed.

![alt text][image3]

The data set contains 8036 samples from center cameras and the data mainly distributed around zero. Learning through Udacity class, I realized that data augmentation techniques were very effective ways to solve this type of problem. Five data augmentation techniques were employed in this project:

  * Images recorded by left and right cameras were randomly used with a little correction of steering angle(model.py lines 50-60)

  * Images were randomly flipped are used with a negative steering angle(model.py lines 63-68).

  * Images were randomly translated in horizontally or vertically with a little correction of steering angle(model.py lines 71-78)

  * Images were randomly shadowed(model.py lines 83-96)

  * Images were randomly adjusted brightness(model.py lines 99-104)

There is a demo of a image argumentation(flip randomly, translate randomly, add shadow randomly, adjust brightness randomly) as shown below:

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used 20 as the maximum of epochs and used early stopping strategy to save the best model according to the validation loss.

Thanks to Udacity, this is a very challenging and funny project. I think I would complete the track two when time is available.
