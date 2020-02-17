# Copyright © 2019 Southern Company Services, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import numpy as np
import pandas as pd
import cv2 as cv




# cell_segmentation
#
# This script divides a rectangularized module image into individual cell images
#
# Inputs:
#   1. module_image (uint8, uint16, int array): module image to subdivide
#   2. cell_output_dir (str): directory to save the cell images to
#   3. cell_counts (tuple): number of cells along each direction of the module, listed (num cells high, num cells wide)
#   4. cell_output_size (tuple): resolution of the scaled output cell images


def cell_segmentation(module_im, cell_output_dir, cell_counts = (12, 6), cell_aspect_ratio = (1,1),  cell_output_size = (150,150)):
    
    #Define approximate module aspect ratio and total cell count
    module_aspect_ratio = cell_counts(2)/cell_counts(1)
    cell_count = cell_counts(1) * cell_counts(2)
    
    #Get dimensions of the image
    im_rows, im_cols, im_channels = module_im.size
    im_aspect_ratio = im_cols/im_rows
    
    
    #Determine module orientation
    #if (im_rows > im_cols):
        #orient = 
    
    #Divide image into cells
    #for (i in range(
        
    return cell_count    
        
        
        
#def edgeDetection():
        
        
        
        
#def lineDetection():
        
        
        
def calculateProjectiveTransform(vert_lines, hori_lines, gray_smooth, resize_rows, image_scalar, debug = False):
    
    #Define variables
    proj_transforms = []
        
    #Iterate through all pairs
    it = 0
    for a in range(0, len(vert_lines)):
        l1 = vert_lines[a][0]
        for b in [x for x in range(0, len(vert_lines)) if x != a]:
            l2 = vert_lines[b][0]
            for c in range(0, len(hori_lines)):
                l3 = hori_lines[c][0]
                for d in [x for x in range(0, len(hori_lines)) if x != c]:
        
                    #Pull second horizontal line
                    l4 = hori_lines[d][0]
                    
                    #Redefine them to span entire image
                        #l1 = vert_lines_samples[0][0]
                    l1_m = (l1[3] - l1[1])/(l1[2] - l1[0])
                    l1_b = l1[1] - (l1_m * l1[0])
                    l1_func = lambda y: (y - l1_b)/l1_m
                    l1 = [int(l1_func(resize_rows)), resize_rows, int(l1_func(0)),0]

                        #l2 = vert_lines_samples[1][0]
                    l2_m = (l2[3] - l2[1])/(l2[2] - l2[0])
                    l2_b = l2[1] - (l2_m * l2[0])
                    l2_func = lambda y: (y - l2_b)/l2_m
                    l2 = [int(l2_func(resize_rows)), resize_rows, int(l2_func(0)),0]

                        #l3 = hori_lines_samples[0][0]
                    l3_m = (l3[3] - l3[1])/(l3[2] - l3[0])
                    l3_b = l3[1] - (l3_m * l3[0])
                    l3_func = lambda y: (y - l3_b)/l3_m
                    l3 = [int(l3_func(resize_rows)), resize_rows, int(l3_func(0)),0]

                        #l4 = hori_lines_samples[1][0]
                    l4_m = (l4[3] - l4[1])/(l4[2] - l4[0])
                    l4_b = l4[1] - (l4_m * l1[0])
                    l4_func = lambda y: (y - l4_b)/l4_m
                    l4 = [int(l4_func(resize_rows)), resize_rows, int(l4_func(0)),0]

                    #Plot sampled perspective lines
                    if debug:
                        gray_sample_lines = cv.cvtColor(gray_smooth, cv.COLOR_GRAY2BGR)
                        gray_sample_lines = cv.line(gray_sample_lines, (l1[0], l1[1]), (l1[2], l1[3]), (0,255,128), 3, cv.LINE_8)
                        gray_sample_lines = cv.line(gray_sample_lines, (l2[0], l2[1]), (l2[2], l2[3]), (0,255,128), 3, cv.LINE_8)
                        gray_sample_lines = cv.line(gray_sample_lines, (l3[0], l3[1]), (l3[2], l3[3]), (0,255,128), 3, cv.LINE_8)
                        gray_sample_lines = cv.line(gray_sample_lines, (l4[0], l4[1]), (l4[2], l4[3]), (0,255,128), 3, cv.LINE_8)

                    #Sample line geometric relationship calculations
                    l1_ang = np.arctan(l1_m) * 180/np.pi
                    l2_ang = np.arctan(l2_m) * 180/np.pi
                    l3_ang = np.arctan(l3_m) * 180/np.pi
                    l4_ang = np.arctan(l4_m) * 180/np.pi
                    l1l3_x = (l3_b - l1_b)/(l1_m - l3_m)
                    l1l4_x = (l4_b - l1_b)/(l1_m - l4_m)
                    l2l3_x = (l3_b - l2_b)/(l2_m - l3_m)
                    l2l4_x = (l4_b - l2_b)/(l2_m - l4_m)
                    l1l3_y = l1_m * l1l3_x + l1_b
                    l1l4_y = l1_m * l1l4_x + l1_b
                    l2l3_y = l2_m * l2l3_x + l2_b
                    l2l4_y = l2_m * l2l4_x + l2_b
                    l1l3_pt = [l1l3_x, l1l3_y]
                    l1l4_pt = [l1l4_x, l1l4_y]
                    l2l3_pt = [l2l3_x, l2l3_y]
                    l2l4_pt = [l2l4_x, l2l4_y]
                    l1_len = np.sqrt((l1l3_pt[1] - l1l4_pt[1])**2 + (l1l3_pt[0] - l1l4_pt[0])**2)
                    l2_len = np.sqrt((l2l3_pt[1] - l2l4_pt[1])**2 + (l2l3_pt[0] - l2l4_pt[0])**2)
                    l3_len = np.sqrt((l1l3_pt[1] - l2l3_pt[1])**2 + (l1l3_pt[0] - l2l3_pt[0])**2)
                    l4_len = np.sqrt((l1l4_pt[1] - l2l4_pt[1])**2 + (l1l4_pt[0] - l2l4_pt[0])**2)
                    l1l3_ang = np.arctan((l1l3_pt[1] - l1l4_pt[1])/(l1l3_pt[0] - l1l4_pt[0])) * 180/np.pi
                    l1l4_ang = np.arctan((l2l3_pt[1] - l2l4_pt[1])/(l2l3_pt[0] - l2l4_pt[0])) * 180/np.pi
                    l2l3_ang = np.arctan((l1l3_pt[1] - l2l3_pt[1])/(l1l3_pt[0] - l2l3_pt[0])) * 180/np.pi
                    l2l4_ang = np.arctan((l1l4_pt[1] - l2l4_pt[1])/(l1l4_pt[0] - l2l4_pt[0])) * 180/np.pi

                    if debug:
                        gray_sample_lines = cv.circle(gray_sample_lines, (int(l1l3_x), int(l1l3_y)), 10, (255,255,255), thickness=-1)
                        gray_sample_lines = cv.circle(gray_sample_lines, (int(l1l4_x), int(l1l4_y)), 10, (0,128,255), thickness=-1)
                        gray_sample_lines = cv.circle(gray_sample_lines, (int(l2l3_x), int(l2l3_y)), 10, (0,128,255), thickness=-1)
                        gray_sample_lines = cv.circle(gray_sample_lines, (int(l2l4_x), int(l2l4_y)), 10, (0,128,255), thickness=-1)

                    #Redefine lines in terms of just the box that is being reshapen
                    l1_box = l1l3_pt + l1l4_pt
                    l2_box = l2l3_pt + l2l4_pt
                    l3_box = l1l3_pt + l2l3_pt
                    l4_box = l1l4_pt + l2l4_pt
                    if debug:
                        gray_sample_box = cv.cvtColor(gray_smooth, cv.COLOR_GRAY2BGR)
                        gray_sample_box = cv.line(gray_sample_box, (int(l1_box[0]), int(l1_box[1])), (int(l1_box[2]), int(l1_box[3])), (0,255,128), 3, cv.LINE_4)
                        gray_sample_box = cv.line(gray_sample_box, (int(l2_box[0]), int(l2_box[1])), (int(l2_box[2]), int(l2_box[3])), (0,255,128), 3, cv.LINE_4)
                        gray_sample_box = cv.line(gray_sample_box, (int(l3_box[0]), int(l3_box[1])), (int(l3_box[2]), int(l3_box[3])), (0,255,128), 3, cv.LINE_4)
                        gray_sample_box = cv.line(gray_sample_box, (int(l4_box[0]), int(l4_box[1])), (int(l4_box[2]), int(l4_box[3])), (0,255,128), 3, cv.LINE_4)


                    #Define the moving (reference) points    
                    moving_pts = np.float32([[l1l3_pt], [l1l4_pt], [l2l3_pt], [l2l4_pt]])

                    #Make second vertical line parallel to first
                    l2_new_m = l1_m
                    l2_new_b = l2l3_y - l2_new_m * l2l3_x
                    l2l4_new_x = (l2_new_b - l4_b)/(l4_m - l2_new_m)
                    l2l4_new_y = l2_new_m * l2l4_new_x + l2_new_b
                    l2l4_new_pt = [l2l4_new_x, l2l4_new_y]
                    l2_new_box = l2l3_pt + l2l4_new_pt
                    if debug:
                        gray_sample_box = cv.line(gray_sample_box, (int(l2_new_box[0]), int(l2_new_box[1])), (int(l2_new_box[2]), int(l2_new_box[3])), (0,128,256), 3, cv.LINE_4)
                        gray_sample_box = cv.line(gray_sample_box, (int(l2l4_x), int(l2l4_y)), (int(l2_new_box[2]), int(l2_new_box[3])), (0,128,256), 3, cv.LINE_4)

                    #Make second horizontal line parallel to first
                    l4_new_m = l3_m
                    l4_new_b = l1l4_y - l4_new_m * l1l4_x
                    l2l4_new_x = (l2_new_b - l4_new_b)/(l4_new_m - l2_new_m)
                    l2l4_new_y = l2_new_m * l2l4_new_x + l2_new_b
                    l2l4_new_pt = [l2l4_new_x, l2l4_new_y]
                    l4_new_box = l1l4_pt + l2l4_new_pt
                    l2_new_box = l2l3_pt + l2l4_new_pt
                    if debug:
                        gray_sample_box = cv.line(gray_sample_box, (int(l2_new_box[0]), int(l2_new_box[1])), (int(l2_new_box[2]), int(l2_new_box[3])), (256,128,0), 3, cv.LINE_4)
                        gray_sample_box = cv.line(gray_sample_box, (int(l4_new_box[0]), int(l4_new_box[1])), (int(l4_new_box[2]), int(l4_new_box[3])), (256,128,0), 3, cv.LINE_4)

                    #Make all angles right - convert rhombus to rectangle
                    if (l3_m == 0):   #Case where no rotation will be necessary at the end
                        l2l4_new_x = l2l3_x
                        l2l4_new_pt = [l2l4_new_x, l2l4_new_y]
                        l1l4_new_x = l1l3_x
                        l1l4_new_pt = [l1l4_new_x, l2l4_new_y]       
                    else:
                        l1_new_m = np.tan((l3_ang - 90) * (np.pi/180))
                        l1_new_b = l1l3_y - l1_new_m * l1l3_x
                        l1l4_new_x = (l1_new_b - l4_new_b)/(l4_new_m - l1_new_m)
                        l1l4_new_y = l1_new_m * l1l4_new_x + l1_new_b
                        l1l4_new_pt = [l1l4_new_x, l1l4_new_y]
                        l2_new_m = l1_new_m
                        l2_new_b = l2l3_y - l2_new_m * l2l3_x
                        l2l4_new_x = (l2_new_b - l4_new_b)/(l4_new_m - l2_new_m)
                        l2l4_new_y = l2_new_m * l2l4_new_x + l2_new_b
                        l2l4_new_pt = [l2l4_new_x, l2l4_new_y]
                        #l1l4_new_pt
                        #l2l4_new_pt
                    l1_new_box = l1l3_pt + l1l4_new_pt
                    l2_new_box = l2l3_pt + l2l4_new_pt
                    l3_new_box = l3_box #NO CHANGE
                    l4_new_box = l1l4_new_pt + l2l4_new_pt
    
    
                    if debug:
                        gray_sample_box = cv.line(gray_sample_box, (int(l1_new_box[0]), int(l1_new_box[1])), (int(l1_new_box[2]), int(l1_new_box[3])), (128,0,256), 3, cv.LINE_4)
                        gray_sample_box = cv.line(gray_sample_box, (int(l2_new_box[0]), int(l2_new_box[1])), (int(l2_new_box[2]), int(l2_new_box[3])), (128,0,256), 3, cv.LINE_4)
                        gray_sample_box = cv.line(gray_sample_box, (int(l4_new_box[0]), int(l4_new_box[1])), (int(l4_new_box[2]), int(l4_new_box[3])), (128,0,256), 3, cv.LINE_4)  
        
                    #Projective Transformation
                    moving_pts = (1/image_scalar) * moving_pts
                    fixed_pts = (1/image_scalar) * np.float32([[l1l3_pt], [l1l4_new_pt], [l2l3_pt], [l2l4_new_pt]]) 
                    proj_trans = cv.getPerspectiveTransform(moving_pts, fixed_pts)
                    proj_transforms.append(proj_trans)
        
        
        
                    #Iteration counter
                    it += 1
        
    #Calculate mean projective transform
    return proj_transforms    
        
#def correctPerspective():
    