# -*- coding: utf-8 -*-
"""
Data & Feature Exploration
"""

import cv2
import random
import matplotlib.image as mpimg
from detection import get_hog_features
from util import plot
    
# original image
car_image = mpimg.imread('./train_images/vehicles/KITTI_extracted/%s.png' % (random.randint(1,100)))
notcar_image = mpimg.imread('./train_images/non-vehicles/Extras/extra%s.png' % (random.randint(1,100)))

plot(car_image, 'car image')
plot(notcar_image, 'notcar image')

# YCrCb
car_ycrcb = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
notcar_ycrcb = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YCrCb)

# hog image
orient = 9
pix_per_cell = 8
cell_per_block = 2

_, car_hog_image = get_hog_features(car_image[:,:,0], orient, pix_per_cell, cell_per_block, True)
_, notcar_hog_image = get_hog_features(notcar_image[:,:,0], orient, pix_per_cell, cell_per_block, True)

plot(car_hog_image, 'car rgb hog features', cmap='gray')
plot(notcar_hog_image, 'notcar rgb hog features ', cmap='gray')

title = ['Y', 'Cr', 'Cb']
for i in range(3):
    _, car_ycrcv_hog = get_hog_features(car_ycrcb[:,:,i], orient, pix_per_cell, cell_per_block, True)
    plot(car_ycrcv_hog, 'car ycrcb hog features %s' % title[i], cmap='gray')

for i in range(3):
    _, notcar_ycrcv_hog = get_hog_features(notcar_ycrcb[:,:,i], orient, pix_per_cell, cell_per_block, True)
    plot(notcar_ycrcv_hog, 'notcar ycrcb hog features %s' % title[i], cmap='gray')






