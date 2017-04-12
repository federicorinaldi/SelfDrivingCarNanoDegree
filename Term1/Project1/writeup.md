# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidYellowCurve.jpg "Original"
[image2]: ./test_images_output/solidYellowCurveGray.jpg "Grayscale"
[image3]: ./test_images_output/solidYellowCurveCanny.jpg "Canny"
[image4]: ./test_images_output/solidYellowCurveRegion.jpg "Region"
[image5]: ./test_images_output/solidYellowCurveHough.jpg "Hough transform"
[image6]: ./test_images_output/solidYellowCurveFinal.jpg "Final"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps:

1. Convert the images to grayscale
2. Applied gaussian blur
3. Applied the Canny transformation to get the Canny edges
4. Calculated the vertices of the region of interest based on the width and height of the image
5. Apply that mask so we only analize the region of interest
6. Apply Hough transform to get all the lines that matched the given parameters
7. Merge the lines with the original image and return that image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculating the slope of each line and looking at its sign, I determined that the positive slopes corresponded to the right lane and the negative slopes corresponded to the left lanes. I've also gathered the intercept for each line at this stage.

After all the lines have been processed, I found the mean of both the slope and the intercepts for each side (left and right) and proceed to calculate the points that would allow me to draw these two averaged lines.

The first point I've calculated rely on the intersection of the two lines (they are going to share this point) so for that I first calculate the x intersection and then the y one.

The second point I had to calculate is per line and it's equal to the intersection of each line and the bottom of the image given by the function y = image.shape[1]

Once I have those 3 points (the intersection of the two lines and the intersection of each line with the bottom) I draw those lines on the image given

Here is a snapshot on how the pipeline works: 

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is that I'm not being able to successfully find the lines in some frames of the solidYellowLeft and the challenge videos. I tried to tweak the parameters of the hough transform and I lowered those frames a lot but there are 7 frames aprox lost on each of those two videos.  

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to store the information of the previous frame in the case of the video so we can do a smoother transition and the lines wouldn't feel that jumpy. Also this would allow me to fix the cases where the lanes are not found for certain frames: having the information on the previous frame I can use that as a good prediction.

Another potential improvement could be to not work on straight lines but try to determine curves.
