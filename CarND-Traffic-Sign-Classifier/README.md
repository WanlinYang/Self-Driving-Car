# **Traffic Sign Recognition** 

## Goals

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

## Writeup

### 1. Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. This image shows an traffic sign example, and the ground truth classification of the traffic sign is given in the data set. Later we will use the traffic sign images and their corresponding labels to train the CNN model.

![dataset_example][./examples/dataset.png]

### 2. Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Image preprocessing

As a first step, I decided to convert the images to grayscale because the shape of input image of LeNet is 32x32x1. This step converts the 32x32x3 GRB image format to 32x32x1 format.

Here is an example of a traffic sign image before and after grayscaling.

![gray_scale][./example/grayscale.jpg]

As a last step, I normalized the image data because we want the process of each image to have a score in a similar range so that our gradients do not go out of control.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

#### Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 gray image   							| 
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 5x5x6						|
| Flatten 				| inputs 5x5x6, outputs 400						|
| Fully connected		| inputs 400, outputs 120						|
| Dropout				| 0.55 keep probability 						|
| Fully connected		| inputs 120, outputs 84						|
| Dropout				| 0.55 keep probability 						|
| Fully connected		| inputs 84, outputs 10							|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

#### Model training

To train the model, I used an Adam optimizer, which is an extension of the classical stochastic gradient descent algorithm. The batch size in this case is 128, number of epoches is 15, and the learning rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

#### Approach

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.952
* test set accuracy of 0.928

Firstly I chose the LeNet, which has two convolutional layers and three fully connected layers. I adopted the network and adjusted the parameters, such as learning rate and training epochs. But the validation accuracy cannot be greater than 0.92. 

Then I observed the accuracy output during a training process. The accuracy did not increase after 5 or 6 epochs, and even decreased. That could be the problem of overfitting. So I inserted two dropout layers after the first two fully connected layers to avoid overfitting, and made the keep probability as 0.55. After several trials, the validation accuracy could be maintained around 0.95.
 

### 3. Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

#### Five German traffic signs

Here are five German traffic signs that I found on the web:

<img src="./test_images/2.ppm" height="100" width="100">  <img src="./test_images/13.ppm" height="100" width="100">  <img src="./test_images/14.ppm" height="100" width="100">
<img src="./test_images/38.ppm" height="100" width="100">  <img src="./test_images/40.ppm" height="100" width="100">

The first image might be difficult to classify because it is slightly blurred because of vehicle speed. The lighting condition of the rest images are not very good, and that may confuse the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

#### Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory	| Roundabout mandatory							| 
| Speed limit (50km/h)  | Speed limit (50km/h)							|
| Stop					| Stop											|
| Yield 	      		| Yield     					 				|
| Keep right			| Keep right         							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This performance is pretty good for these instances.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

#### Top 5 softmax probabilities

Then I output the top 5 softmax probabilities of the above image examples.

For the first image, the model is relatively sure that this is a roundabout mandatory (probability of 0.75), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.75					| Roundabout mandatory   						| 
| 0.10					| Speed limit (30km/h)							|
| 0.05					| End of all speed and passing limits			|
| 0.03					| Priority road					 				|
| 0.01					| Ahead only      								|

The performance of the second image is pretty good. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Speed limit (50km/h)   						| 
| 1.8889788e-14			| Speed limit (30km/h)							|
| 4.1214671e-17			| Speed limit (80km/h)							|
| 1.2793425e-21			| Speed limit (60km/h)			 				|
| 3.8778951e-24			| Wild animals crossing							|

The third image.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9999201e-01			| Stop   										| 
| 7.0643123e-06			| No entry										|
| 5.3659204e-07			| Turn right ahead								|
| 2.4193503e-07			| Yield			 								|
| 9.9716388e-08			| No vehicles									|

The forth image.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Yield   										| 
| 7.3975172e-30			| Ahead only									|
| 2.7682390e-30			| No vehicles									|
| 9.3973059e-34			| Priority road	 								|
| 4.6895273e-34			| Keep right									|

The fifth image.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Keep right									| 
| 7.3975172e-30			| Go straight or right							|
| 2.7682390e-30			| Turn left ahead								|
| 9.3973059e-34			| Speed limit (80km/h)							|
| 4.6895273e-34			| Yield											|

From above image instances, it easy to find that the clearer the image is, the better this model performs. The model is able to produce more accurate results if the light and camera conditions are good.

### 4. Visualizing the Neural Network

Lastly, we can plot the feature map of the first and the second convolutional layer.

![feature_maps][./example/feature_maps.png]