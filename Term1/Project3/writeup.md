# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: CNN-Diagram.png "Model Visualization"
[image2]: test_image.jpg "Original image taken from driving"
[image3]: test_image_flipped.jpg "Flipped version of the image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of an implementation of the [NVidia convolutional neural network](https://arxiv.org/pdf/1604.07316v1.pdf) but with added RELU activation layer for 3 of the 4 fully-connected layers  (model.py lines 104-106) 

The model also includes a layer for Cropping the image and get rid of the top and bottom parts as shown in one of the course lessons (code line 97), and the data is normalized in the model using a Keras lambda layer (code line 96). 

#### 2. Attempts to reduce overfitting in the model

I've tried introducing Dropout layers on the Convolutional Layers but after doing some empirical tests I found out that it wasn't improving the model at all. So my next move to try to reduce overfitting was to improve the data, see point 4 for more details.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 83). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. At first I tried using a combination of center lane driving, recovering from the left and right sides of the road and driving the track on the opposite direction but after some try and error and speaking with my mentor Minith, he told me that the Udacity Sample data provided should be enough so I focused on improving the pre processing of that dataset. For more details, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the basics so I followed closely the course videos, I first started with a linear model and then added LeNet instead. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that this model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Even with higher MSE on the validation, this proven to be Ok until I hit the bridge part of the track at that point I knew I had to try something more powerful.

I've implemented the NVidia CNN and then the model had a lower MSE for validation but not small enough

To combat the overfitting, I modified the model introducing a generator that would take an image randomly, inside this process the image would be flipped randomly as well over the Y axis along with it's steering angle this proved to be beneficial as it normalized the test set.

Then I increased the number of trainings (a batch of 2000 over 10 epochs) as now I had the generator baking up the test data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 95-108) consisted of the aforementioned convolution neural network with the following layers:


![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I started capturing my own dataset but I ended up using Udacity Test Dataset for this project.

I had 8036 data points but as each included the images for the 3 cameras I preprocessed this data by following the udacity course, so I've applied the steering correction (code lines 33-35) and ended up with a dataset of 24108 elements. I've also randomly flipped the image over the Y axis on the generator (code lines 48-53) so that gives me a total dataset of 48216.

Here is an example of one of the images feeded to the network and it's flipped counterpart:

![alt text][image2] ![alt text][image3]

I finally randomly shuffled the data set and put 33% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the training loss that was almost 0% and that the validation loss started going up after that epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Simulation

I've provided 2 videos (video.mp4 and video2.mp4) were it shows 2 successful laps being navigated correctly by the algorithm