# -*- coding: utf-8 -*-
import time
import glob
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from detection import extract_features

def load_train_images():
    cars = []
    notcars = []

    car_images = glob.glob("vehicles/*/*.png", recursive=True)
    for image in car_images:
        cars.append(image)

    notcar_images = glob.glob("non-vehicles/*/*.png", recursive=True)
    for image in notcar_images:
        notcars.append(image)

    return cars, notcars

def train_classifier():
    """
    1.Extract features from images
    2.Get training set and validation set
    3.Train svc classifier
    4.Test on the validation set
    5.Serialize the svc and params to disk
    """
    # Load train images
    cars, notcars = load_train_images()

    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size =  (32, 32)
    hist_bins = 32
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    # feature prepare
    t=time.time()
    car_features = extract_features(cars, color_space=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=False, hist_feat=True, hog_feat=True)

    notcar_features = extract_features(notcars, color_space=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=False, hist_feat=True, hog_feat=True)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # combined-features normalize
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # training set & validation set
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    # use svc 
    svc = LinearSVC()

    # training
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    
    # validating
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Save results
    dist = {
        'svc': svc, 'scaler': X_scaler, 'orient': orient,
        'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block,
        'spatial_size': spatial_size, 'hist_bins': hist_bins}
    pickle.dump(dist, open('svc_pickle.p', 'wb'))

    return svc

train_classifier()

