# **Finding Lane Lines on the Road**


## Goals

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect work in a written report


[//]: # (Image References)

[blur]: ./examples/blur.jpg "Blur"
[canny]: ./examples/canny.jpg "Canny"
[masked_canny]: ./examples/masked_canny.jpg "Masked Canny"
[hough]: ./examples/hough_lines.jpg "Hough Lines"
[long_lines]: ./examples/long_lines.jpg "Long Lines"
[long_lines_mask]: ./examples/long_lines_mask.jpg "Masked Long Lines"
[overlapped]: ./examples/overlapped.jpg "Overlapped"


## Reflection

### 1. Pipline

My pipeline consisted of 8 steps.

1. Convert the image to grayscale
<img src="./examples/gray.jpg" height="50%" width="50%">

2. Gaussian blur to wipe out noise. The size of kernel is 5 in this case.
<img src="./examples/blur.jpg" height="50%" width="50%">

3. Canny edge detection.
<img src="./examples/canny.jpg" height="50%" width="50%">

4. Mask the image of edge outputed from the previous step.
<img src="./examples/masked_canny.jpg" height="50%" width="50%">

5. Houghline detection. This step outputs a series of coordinates of lines in the form
of [x1, y1, x2, y2].
<img src="./examples/hough_lines.jpg" height="50%" width="50%">

6. Separate lines can calculate the mean value of lines' coordinates. Then elongate the mean lines
and plot them on a blank (np.zeros) image.
<img src="./examples/long_lines.jpg" height="50%" width="50%">

7. Mask the images with elongated line from the previous step.
<img src="./examples/long_lines_mask.jpg" height="50%" width="50%">

8. Overlap over the original image
<img src="./examples/overlapped.jpg" height="50%" width="50%">

### 2. Potential shortcomings with the current pipeline

One potential shortcoming would be tunning the parameters of OpenCV functions. The pipeline
includes several steps and many parameters. Although each step was tested carefully,
not every step can be ideal with tuning by human.

### 3. Possible improvements the pipeline

A possible improvement would be to calculate the mean values of coordinates among frames
in a video. This may reduce the oscillation of plots when some noise appears in one frame.

