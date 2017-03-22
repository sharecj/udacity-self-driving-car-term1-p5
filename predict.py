# -*- coding: utf-8 -*-
import glob
import pickle

import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from detection import find_cars
from util import plot
    
# Load svc pickle
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )

svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

ystart = 400
ystop = 656
scales = [1, 1.5]
hog_channel = 'ALL'

# Test on test images
test_images = glob.glob("test_images/*")
out_images = []
box_list = []

#for file in test_images:
#    image = mpimg.imread(file)
#    out_image = find_cars(image, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, 
#                        spatial_size, hist_bins, hog_channel, spatial_feat=False, hist_feat=True, hog_feat=True)
#    out_images.append(out_image)

#for idx, image in enumerate(out_images):
#    plot(image, 'Test Image %s' % idx)

def process_video(image):
    out_image = find_cars(image, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, 
                          spatial_size, hist_bins, hog_channel,spatial_feat=False, hist_feat=True, hog_feat=True)
    return out_image
    
output_video = 'my_project_video2.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_video)
output_clip.write_videofile(output_video, audio=False)