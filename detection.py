# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections import deque
from skimage.feature import hog
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

from util import plot

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    return hog(img, orientations=orient,
              pixels_per_cell=(pix_per_cell, pix_per_cell),
              cells_per_block=(cell_per_block, cell_per_block),
              transform_sqrt=True,
              visualise=vis, feature_vector=feature_vec)

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features
    
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features
    
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2RGB':
        return np.copy(img)
    return cv2.cvtColor(img, eval('cv2.COLOR_%s' % conv))
    
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        feature_image = convert_color(image, 'RGB2%s' % color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))

    return features

heatmap_history = deque(maxlen = 5)
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, 
              spatial_size, hist_bins, hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Find Cars.
    1. preprocess image(normalize, crop, scale)
    2. extract image hog features
    3. sliding window with multi-scale to classifiy the window image is car or not
    4. use heatmap to filter false positive
    5. draw rectangle when it is a car
    """
    def preprocess(img, ystart, ystop, scale):
        # normalize
        img = img.astype(np.float32)/255
        # crop sky
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        # scale
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        return ctrans_tosearch
    
    def get_image_hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel):
        if hog_channel == 'ALL':
            hog_features = []
            for i in range(3):
                hog_feature = get_hog_features(img[:,:,i], orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog_features.append(hog_feature)
        else:
            hog_features = get_hog_features(img[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)
        return hog_features
    
    def get_window_hog_features(image_hog_features, row_start, row_end, col_start, col_end, hog_channel):
        if hog_channel == 'ALL':
            window_hog_features = []
            for i in range(3):
                window_hog_feature = image_hog_features[i][row_start:row_end, col_start:col_end].ravel()
                window_hog_features.append(window_hog_feature)
            window_hog_features = np.hstack(window_hog_features)
        else:
            window_hog_features = image_hog_features[hog_channel][row_start:row_end, col_start:col_end].ravel()

        return window_hog_features
    
    def get_window_image(img, row_start, row_end, col_start, col_end):
        return cv2.resize(ctrans_tosearch[row_start:row_end, col_start:col_end], (64,64))
    
    def get_heatmap(box_list):
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        for box in box_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap = np.clip(heatmap, 0, 255)
        return heatmap        
    
    def apply_threshold(heatmap, threshold):
        heatmap[heatmap < threshold] = 0
        return heatmap    
        
    def draw_labeled_bboxes(img, labels):
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
    
    box_list = []
    draw_img = np.copy(img)

    for scale in scales:
        ctrans_tosearch = preprocess(img, ystart, ystop, scale)
            
        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - 1
        nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - 1
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    

        if hog_feat:
            # Compute individual channel HOG features for the entire image
            image_hog_features = get_image_hog_features(ctrans_tosearch, orient, pix_per_cell, cell_per_block, hog_channel)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                               
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                    
                # Extract the image patch
                subimg = get_window_image(ctrans_tosearch, ytop, ytop+window, xleft, xleft+window)
    
                features = []
                if spatial_feat:
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    features.append(spatial_features)

                if hist_feat:
                    # Get color features
                    hist_features = color_hist(subimg, nbins=hist_bins)
                    features.append(hist_features)
                    
                if hog_feat:
                    # Extract HOG for this patch
                    window_hog_features = get_window_hog_features(image_hog_features, ypos, ypos+nblocks_per_window, xpos, xpos+nblocks_per_window, hog_channel)
                    features.append(window_hog_features)
                    
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack(features).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
    
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
    
                    box_list.append(( (xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart) ))

    heatmap = get_heatmap(box_list)
    heatmap_history.append(heatmap)
    sum_heatmap = np.zeros([heatmap.shape[0], heatmap.shape[1]])
    for i in range(len(heatmap_history)):
        sum_heatmap = np.add(sum_heatmap, heatmap_history[i])
    average_heatmap = sum_heatmap / len(heatmap_history)
    heatmap = apply_threshold(average_heatmap, 5)
    
    #plot(heatmap, 'heat map', cmap='gray')

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img

