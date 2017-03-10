import numpy as np
import pandas
import os
import matplotlib.image as mpimg
from skimage import transform
from PIL import Image
from random import randint

#### Here I extracted vehicle images from the CrowdAI and Autti dataset ####

dir = 'object-dataset/'             #local path to dataset and csv
csv = 'object-dataset/labels.csv'

dataframe = pandas.read_csv(csv, header=None)   #structure the frames, bounding boxes and labels
dataset = dataframe.values
xmaxi = dataset[0:,3]      # this one is for Autti dataset, CrowdAI has different label.csv
ymini = dataset[0:,2]      # if someone really reads this, the CrowdAI bounding box coordinates are
xmini = dataset[0:,1]      # kind of awkward. One can not crop images in the normal manner: [y1:y2, x1:x2]
ymaxi = dataset[0:,4]      # or: [ymin:ymax, xmin:xmax]. For CrowdAI it's: [xmin:ymin, xmax:ymax]
frame = dataset[0:,0]      # which is really weird, because than y-values are sometimes over 1200 and
labels = dataset[0:,6]     # y-values in a 1920x1200 frame should, by common sense, not be bigger than 1200.

#### Taking every frame from the label Car and check if they are at least 32x32 pixels big  #### 

for xmin, xmax, ymin, ymax, image, label in zip(xmini, xmaxi, ymini, ymaxi, frame, labels):
    
    if label == 'car':            
        
        if abs(int(ymax)-int(ymin)) < 32:
            continue
        else:
            if abs(int(xmax)-int(xmin)) < 32:
                continue
            else:
                print("Creating vehicle images from data and bounding boxes. This may take a while!")
    
                path, image_file = os.path.split(image)
                                    
                src_fname, ext = os.path.splitext(image)    # get the names of the frames
                
                image_raw = mpimg.imread(dir + image)
            
                roi = image_raw[int(ymin):int(ymax), int(xmin):int(xmax)]    # cropping out the actually car/cars of the frame
            
                roi_reshaped = transform.resize(roi, (64, 64))   # reshape it to 64x64 pixels
                
                rescaled = (255.0 / roi_reshaped.max()*(roi_reshaped- roi_reshaped.min())).astype(np.uint8)  # converting numpy array back to image
                    
                im = Image.fromarray(rescaled)
                
                random = randint(0,100)  # adding a random number to every image file name, because there are multiple images per frame
                
                save_fname = os.path.join('extracted_imgs/', os.path.basename(src_fname)+str(random)+'.jpg')
                
                im.save(save_fname)  # save extracted image to folder, so I can load it when ever I want
            
    else:
        continue
    
print("Done!")