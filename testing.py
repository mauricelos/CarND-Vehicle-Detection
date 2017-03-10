import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from functions import *
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

### Reading in images or videos and run them trough pipeline ####

image1 = mpimg.imread('test_images/test1.jpg')

def pipeline(image):

    with open('svc_pickle.p', 'rb') as f:    #loading previously saved pickle with all neccessary parameters
        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, hog_channel, spatial_feat, hist_feat, hog_feat = pickle.load(f)

    
    box_list = []    # creating a box list for all detected objects in find cars
    
    def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        
        draw_img = np.copy(img)
        #img = img.astype(np.float32)/255  # not neccessary I converted all my training images to .jpg
        
        img_tosearch = img[ystart:ystop,:,:]   # searching only in the selected area
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')  # converting image to colorspace used in training
        if scale != 1:
            imshape = ctrans_tosearch.shape  # scaling image, if scale other than 1
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]   # individual channels of the 3-channel image
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        
        window = 64  #window size (8x8)
        nblocks_per_window = (window // pix_per_cell)-1  #block per window
        cells_per_step = 1  #cells to step per loop (I had this on 1 when I processed the video and images 2 shopuld be okay as well!)
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step  #total steps
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # compute HOG feature for every channel, get_hog_features definition is found in functions.py
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # Extract the image patch and resizing it
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features, all computed i function.py
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
    
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:    # if prediction == 1 (1 == Car) creat bounding box
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                    #box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                    if xbox_left > 400:   # I did this to eliminate false positives from the left road in the video, but it works without it as well
                        box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                    else:
                        continue
                    
        return draw_img
        
    ystart = 400  # my set y start and stop and scale
    ystop = 550
    scale = 1.5
        
    out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    plt.imshow(out_img)   # shows the image with raw bounding boxes
    
    #print(box_list)   #shows the box_list
     
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    def add_heat(heatmap, bbox_list):   # applying heat
        
        for box in bbox_list:    # Iterate through list of box_list
            # Add += 1 for all pixels inside each box
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
        return heatmap
        
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        
        return heatmap

    hot_box = []   # contains the final bounding boxes around the vehicles
    
    def draw_labeled_bboxes(img, labels):
        
        for car_number in range(1, labels[1]+1):   # Iterate through all detected vehicles
            
            nonzero = (labels[0] == car_number).nonzero()  # Find pixels with each car_number label value
            
            nonzeroy = np.array(nonzero[0])    # Identify x and y values of those pixels
            nonzerox = np.array(nonzero[1])
            
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))) # define bounding boxes based on min/max x and y-values
            
            hot_box.append(bbox)   # append bounding boxes to list 
            
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)  # draw bounding boxes on image
        
        return img
    
#### Applying heat and create labels for final result ####
        
    heat = add_heat(heat,box_list)
        
    heat = apply_threshold(heat,4)
      
    heatmap = np.clip(heat, 0, 255)
    
    labels = label(heatmap)
    
    result = draw_labeled_bboxes(np.copy(image), labels)
    
### Print hot_ box and show Result and Heat map ####    
    
    print(hot_box)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(result)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='binary_r')
    plt.title('Heat Map')
    fig.tight_layout()
    #plt.savefig('output_images/labels_map.jpg')
    #cv2.imwrite('output_images/test2_done.jpg', result)
    
    return result


pipeline(image1)

#result_output = 'result_1.mp4'
#clipl = VideoFileClip("project_video.mp4")
#input_clip = clipl.fl_image(pipeline)
#input_clip.write_videofile(result_output, audio=False)



    
