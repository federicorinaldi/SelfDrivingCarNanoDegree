# **Traffic Sign Recognition** 
### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distribution.png "Visualization"
[image2]: ./examples/originals.png "Original images as in the dataset"
[image3]: ./examples/processed.jpg "Images after pre processing"
[image4]: ./examples/sign1.png "6t sign"
[image5]: ./examples/sign2.png "Max 30"
[image6]: ./examples/sign3.png "Turn right ahead"
[image7]: ./examples/sign4.png "Max 100"
[image8]: ./examples/sign5.png "Pedestrian"
[image9]: ./examples/sign6.png "Max 130"
[image10]: ./examples/sign1p.png "6t sign"
[image11]: ./examples/sign2p.png "Max 30"
[image12]: ./examples/sign3p.png "Turn right ahead"
[image13]: ./examples/sign4p.png "Max 100"
[image14]: ./examples/sign5p.png "Pedestrian"
[image15]: ./examples/sign6p.png "Max 130"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one. The submission includes the project code.

You're reading it! and here is a link to my [project code](github.com/federicorinaldi/SelfDrivingCarNanoDegree/blob/master/Term1/Project2/Traffic_Sign_Classifier.ipynb) and to the [HTML generated](github.com/federicorinaldi/SelfDrivingCarNanoDegree/blob/master/Term1/Project2/Traffic_Sign_Classifier.html) 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the different categories. As you can see the data is not normally distributed.

![alt text][image1]

Here is random sample of the images in the data set to better understand the dataset:

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I've created a pre processing function that I've applied to all of the datasets:

1. First we convert the dataset to grayscale as in the [paper suggested in the project readme](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) they end up recommending converting to grayscale. I'm trying to avoid looping through the dataset so I've [found a way](http://www.had2know.com/technology/rgb-to-gray-scale-converter.html) to convert to grayscale as a linear transformation of the input matrix
2. I remove the unneeded dimensions as we are only using grayscale we can reshape the input to 1 dimension
3. Last I normalize the images on the [0;1] range with the formula Xi = (Xi - min(x))/(max(x) - min(x)) as resulting in the linear transformation X/255 

Here is an example of some traffic signs images after processing.

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5      	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU          		|           									|
| Max pooling			| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| 120 outputs									|
| RELU 					|												|
| Dropout       		| we keep 0.5									|
| Fully connected		| 84 outputs									|
| Dropout       		| we keep 0.5									|
| RELU 					|												|
| Fully connected		| 43 outputs									|
 
As you can see the layers basically represents the LeNet ConvNet but with dropout added for the training set. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.955
* test set accuracy of 0.937

At first I tried to follow the paper from Pierre Sermanet and Yann LeCun but I couldn't get good results as I couldn't get the dimensions correctly and also it was really slow. After a few tries I started from scratch with the LeNet ConvNet after the added preprocessing I was able to get more than 90% on the validation set. Then I figured that my model was over fitting so I included the dropout, I've played with the values and found that if I put something higher as the keep probability the validation test would be higher but the test set would be lower and as I already had more than 93% on the validation test I was aiming for a higher test set score. Finally I played with the epochs and the learning rate, I had no luck with other learning rates but the LeNet default but I've found that with 100 epochs and a batch of 128 some times I would get a validation set score of 97%. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The first and the last images might be difficult to classify because they weren't in any of the trained categories. The others should have been detected correctly. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 80%. This compares negative to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top five soft max probabilities for each image are the following:

Image 1:

| Probability         	|     Prediction	  | 
|:---------------------:|:-------------------:| 
| Speed limit (70km/h)  |   0.77              |
| Speed limit (80km/h)  |   0.09              |
| Speed limit (50km/h)  |   0.06              |
| Road work             |   0.04              |
| Keep left             |   0.01              |

Image 2:

| Probability         	|     Prediction	 | 
|:---------------------:|:------------------:| 
| Speed limit (30km/h)  | 0.99               |
| Speed limit (50km/h)  | 0.1                |
| Speed limit (20km/h)  | 0                  |
| Keep right            | 0                  |
| General caution       | 0                  |

Image 3:

| Probability         	|     Prediction	| 
|:---------------------:|:-----------------:| 
| Turn right ahead      |    1.0            |
| No vehicles           |    0              |
| Stop                  |    0              |
| Priority road         |    0              |
| Ahead only            |    0              |

Image 4:

| Probability         	|     Prediction	| 
|:---------------------:|:-----------------:| 
| Speed limit (50km/h)  | 0.75              |
| Speed limit (80km/h)  | 0.11              |
| Wild animals crossing | 0.09              |
| Speed limit (60km/h)  | 0.02              |
| Double curve          | 0.01              |

Image 5:

| Probability                           |     Prediction	| 
|:-------------------------------------:|:-----------------:| 
| General caution                       |   0.98            |
| Pedestrians                           |   0.01            |
| Road narrows on the right             |   0.00            |
| Right-of-way at the next intersection |   7.37            |
| Beware of ice/snow                    |   0               |

Image 6:

| Probability         	|     Prediction	| 
|:---------------------:|:-----------------:| 
| Pedestrians           | 0.71              |
| Roundabout mandatory  | 0.15              |
| Speed limit (30km/h)  | 0.05              |
| General caution       | 0.03              |
| Speed limit (100km/h) | 0.03              |