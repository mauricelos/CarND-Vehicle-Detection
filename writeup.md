##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle_non_vehicle.jpg
[image2]: ./output_images/comparison_hog.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/original_heat_hot.jpg
[image6]: ./output_images/labels_map.jpg
[image7]: ./output_images/original_heat.jpg
[video1]: ./result.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Extracting Images from `CrowdAI` and `Autti` dataset

I extracted images from both dataset to get some addional training data for the cars class. You can find my code for that in the `extract_images.py`! First I initialzed all parameters of the labels.csv and added the path to the images (line 11-21). Thenn I took every images with the label "car" and checked if the bounding boxes were bigger than 32x32 to reduce the amount of really tiny images, which would help in the training (line 25-35). After that's done the remaining images are cropped from the raw frames and saved in a seperate folder (line 37-58).

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 63 through 74 of the file called `training.py`). Actually these lines just send all image paths to `functions.py`. In `functions.py` all features get extracted with "extract_features"(line 64- 112). The HOG features are then extracted by "get_hog_features"(line 24-41), which takes in the image, orient, pixel per cell and cell per block, which can all be changed in the `training.py`. The HOG features are then processed by skimage.feature (hog).

My training's dataset contains the `vehicle` and `non-vehicle` image dataset as well as images created from the `CrowdAI` and `Autti` dataset.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(4, 4)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and came to the conclusion that I want a little bit more detail so I used 4 pixels_per_cell, which gives me a little bit more detail than with 8. But it's also not to detailed, which would make classification harder. For Orientations I choose 9, which again seems to be a good compromise. More orientation would make classification harder less would generalize to much so that the difference between car and road would be to small and false positive would appear more often in testing.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearns LinearSVC. I split my data into training and testing. The testing dataset is 20% of the whole dataset (line 86- 103 in `training.py`). Color features were used as well. Also I had to reduce my dataset to 5000 samples per class, because my memory storage was maxed out with the whole dataset.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

I decided to search in a set region of the image, which was set with y_start , y_stop. In this area I use a 64 pixels big image grid. I also use overlap so that every inch of the selected area gets searched for vehicles. My overlap is defined by the amount of cell I go per step, which I set to 1 (which is not good for performance, 2 would be better perfoming). My scale is set to 1.5 and it's able to catch both vehicle in the video. Increasing the scale would increase the area in which boung boxes are combined to one big one and that could lead to unprecise detection. Code for sliding windows: (line 22-85 in `testing.py`)

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I also reduced the steps per cell to one, because it increases the amount of windows around the vehicles. This hurts performance, but makes the detection a bit more solid (my pipeline should work with 2 steps per cell as well!). Here are some example images (`more images of my final result on all 6 test images can be found in the output_images folder`):

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The video can be found within the folder (result.mp4)!


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (line 82-85 in `testing.py`). From the positive detections I created a heatmap (line 101-114 in `testing.py`)and then thresholded that map to identify vehicle positions (line 110-114 in `testing.py`). I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected (line 118-133 in `testing.py`).  

Here's an example result showing the heatmap on a test image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the original image:

### Here is the image and their corresponding heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:

![alt text][image6]

### Here the resulting bounding boxes are drawn onto the image:

![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This kind of object identification needs a lot of data from different places to work good in many situations. If the situation is different from the situation in the video the identification most likely wouldn't be as solid. So more data would make the pipeline more robust. Also multiple scaling values for the windows could improve the result of the pipeline, as objects that as far away or close would be detected better, this would on the other hand also take away some performance and eventually false positives could increase as well as real detection. 

