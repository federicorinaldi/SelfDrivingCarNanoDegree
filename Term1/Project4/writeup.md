## Writeup Template

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

[image1]: ./output_images/sample_chess.png "Sample image"
[image2]: ./output_images/undistorted_chess.png "Undistorted Image"
[image3]: ./output_images/undistorted.png "Undistorted Image"
[image4]: ./output_images/unwarped_chess.png "Unwarped Image"
[image5]: ./output_images/thresholded_gradient_x.png "Thresholded Gradient over X"
[image6]: ./output_images/thresholded_gradient_y.png "Thresholded Gradient over Y"
[image7]: ./output_images/thresholded_magnitude.png "Thresholded Magnitude"
[image8]: ./output_images/thresholded_grad_dir.png "Thresholded Grad. Dir."
[image9]: ./output_images/combined_image.png "Combined image"
[image10]: ./output_images/gray_binary.png "Gray binary"
[image11]: ./output_images/s_binary.png "S Binary"
[image12]: ./output_images/h_binary.png "H Binary"
[image13]: ./output_images/thresholded_s.png "Thresholded S"
[image14]: ./output_images/pipeline_applied.png "Pipeline applied image"
[image15]: ./output_images/warped_image.png "Warped image"
[image16]: ./output_images/warp_binary.png "Warped image"
[image17]: ./output_images/lane_detection.png "Lane detection"
[image18]: ./output_images/result.png "Final result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the four first code cells of the IPython notebook located in "./Advanced Lane Finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

And here the final warped image:

![alt text][image4]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The pipeline is found on the cell 14 of the notebook and it's a class named ImageProcessor. The first step of the pipeline is to apply the distortion correction:

![alt text][image3]

After this I've followed by applying a gaussian blur to smooth a little bit the image

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The next step in the pipeline is to convert to YUV and HLS color spaces. There are more examples of the benefits of using this color spaces on the notebook. Then I select the U and V (YUV) and S (HLS) channels and convert to grayscale, I didn't know how to convert to grayscale in this mixed color space but I found the solution on [one of the students submission](https://github.com/GeoffBreemer/SDC-Term1-P4-Advanced-Lane-Finding/blob/master/BinaryThresholder.py#L17).

After that I followed by putting the grayscaled image through the functions abs_sobel_thresh, abs_sobel_thresh, mag_thresh, dir_threshold and hls_select all of those discussed during the course. I've combined all of those images and ended up with my binary image:

![alt text][image9]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_image()` inside the ImageProcessor class, which appears on the cell 14 of the the IPython notebook.  The `transform_image()` function takes as inputs an image (`img`). The source (`src`) and destination (`dst`) points are inside the function. I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[ 20, 690],      # bottom left
                    [ 550, 460],     # top left
                    [ 730, 460],     # top right
                    [ 1260, 690]])     # bottom right

dst = np.float32([
    (0, img.shape[0]),
    (0, 0),
    (img.shape[1], 0),
    (img.shape[1], img.shape[0])])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 20, 690      | 0, 720        | 
| 550, 460      | 0, 0      |
| 730, 460     | 1280, 0      |
| 1260, 690      | 1280, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image15]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I've followed the course videos and fit my lane lines with a 2nd order polynomial and used sliding window the code is inside the `find_lanes` method of the class ImageProcessor in the 14th cell of the notebook. I've also created a helper method `plot_lanes` to show how the innner workings of the find lane algorithm works. This is the result:

![alt text][image17]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To find the curvature of the lane and other metrics I've used [the work of the user ctsuu](https://github.com/ctsuu/Advanced-Lane-Finding/blob/master/Advanced-Lane-Finding-Subimission.ipynb) that led me to create the method `__get_rad_po` that is used by the `draw_lanes` method and it returns camera_pos, turning_radius, steering_angle. You can see those values on the next point.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the cell 14 of the jupyter notebook in the function `draw_lanes`.  Here is an example of my result on a test image:

![alt text][image18]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](.output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took was to follow closely the videos on the course, you can tell as a lot of the code in the notebook was part of the quizzes. And I really struggled when I had to do entirely on my own for example one of the biggest issues I had was to find the right source and destination points for the transform, I've spend hours on this. The good thing is that there is a lot of information from other students and that helped me to carry on.

About the implementation, I've only did some basic fail-recovery scenarios, for example if I don't find parallel lines in a given frame I default to the previous one but this can be heavily improved by using some of the other properties exposed on the Lane object, like best_fit or recent_xfitted. I think also the parameters and thresholds for the compound binary image can be improved as some time the resulting image lacks some of the lanes.

One thing that needs further investigation is the performance as I've noticed that for a 50 seconds video took more than 2 minutes of processing discarding it as a fit for real time data.