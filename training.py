import pickle
import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from functions import *
from sklearn.model_selection import train_test_split

#### Loading data from folders ####    

images1 = glob.glob('non_vehicles_new/*.jpg')
images2 = glob.glob('vehicles_new/GTI_Far/*.jpg')
images3 = glob.glob('vehicles_new/GTI_Left/*.jpg')
images4 = glob.glob('vehicles_new/GTI_MiddleClose/*.jpg')
images5 = glob.glob('vehicles_new/GTI_Right/*.jpg')
images6 = glob.glob('test/*.jpg')

#### Putting all image paths in the cars/notcars list ####

cars = []
notcars = []

for image in images1:
    notcars.append(image)
    
for image in images2:
    cars.append(image)
    
for image in images3:
    cars.append(image)
    
for image in images4:
    cars.append(image)
    
for image in images5:
    cars.append(image)
    
for image in images6:
    cars.append(image)

#### Restricting the amount of data used, because of memory capacity ####
    
cars = cars[0:5000]        # I have about 125,000 images of vehicles, but using all would make no sense as non vehicles would be underrepresented then
notcars = notcars[0:5000]  # there are 8900+ images in my nonvehicle folder, so for best result all should be used 

#### Setting my parameters #### 

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb, Gray
orient = 9  # HOG orientations
pix_per_cell = 4 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

#### Sending everything to functions.py where images get converted to features ####

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)  # stacking the cars and notcars features                       

X_scaler = StandardScaler().fit(X)   # applying a normalization on the stacked features 

scaled_X = X_scaler.transform(X)


y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))  # stack the labels as well (cars are ones, non cars are zeros)


rand_state = np.random.randint(0, 100)      # splitting the data in are random manner and use 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

svc = LinearSVC()   # applying a linear SVC

t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

t=time.time()

#### Saving everything in a pickle file ####

with open('svc_pickle5.p', 'wb') as f:    
    
    pickle.dump([svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, hog_channel, spatial_feat, hist_feat, hog_feat], f)