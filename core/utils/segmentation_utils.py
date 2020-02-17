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
import math



# cell_segmentation
#
# This script divides a rectangularized module image into individual cell images
#
# Inputs:
#   1. module_image (uint8, uint16, int array): module image to subdivide
#   2. cell_output_dir (str): directory to save the cell images to
#   3. cell_counts (tuple): number of cells along each direction of the module, listed (num cells high, num cells wide)
#   4. cell_output_size (tuple): resolution of the scaled output cell images


def cell_segmentation(module_im, cell_output_dir, cell_counts = (12, 6), cell_aspect_ratio = (1,1),  cell_output_size = (150,150),\
                      debug = True):
    
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


def processImage(image, debug=True):
        
      
    
    
    
    #Adjust hue to remove influence of wrapping (hue is specified as 360 deg around a cylinder - 180 for opencv for uint8 type)
    hue_filtered = hue
    #hue_filtered[hue >= 170] = 0
    #hue_filtered[val < 48] = 0

    
# edgeDetection
#
# Scrip applying contextual information to refine the Canny edge detection algorithm
#
# Inputs:
#   1. rgb
#   2. rgb_min: single channel image where the value for each pixel was defined as the minimum of all rgb channels 
#   3. gray: standard grayscale image
#   4. hue: hue channel from the image after converting from rgb to hsv
#   5. image_pixels (int): total number of pixels in the image
#   6. debug (bol) (OPTIONAL): specifies if debug outputs should be generated
#
# Returns:
#   1. edges: binary image showing the final detected edges
#   2. outputs (dict): contains all other required and debug outputs
    

def edgeDetection(rgb, rgb_min, gray, hue, image_pixels, debug=True):
   
    #Print update
    if debug:
        print('   Edge Detection:')
        
    #Initial edge detection parameters
    gauss_size = (5,5)
    gauss_std = 2;
    canny_thresh1 = 100
    canny_thresh2 = 200
    edge_it = 0
    edge_ratio_goal = 0.005 # 0.5% of total image area
    edge_ratio_range = (0.0025, 0.0075)
    edge_errors = []
    edge_repeat = True
    
    #Iteratively refine edge detection until the proportion of edges falls into a prespecified range
    while edge_repeat & (edge_it < 10):
        
        #Apply Gaussian Blur
        hue_smooth = cv.GaussianBlur(hue, gauss_size, gauss_std)
        rgb_min_smooth = cv.GaussianBlur(rgb_min, gauss_size, gauss_std)
        gray_smooth = cv.GaussianBlur(gray, gauss_size, gauss_std)

        #Canny edge detection
        edges_rgb_min = cv.Canny(rgb_min_smooth, 0.5 * canny_thresh1, 0.5 * canny_thresh2)
        edges_gray = cv.Canny(gray, canny_thresh1, canny_thresh2)
        edges_hue = cv.Canny(hue_smooth, 25, 50)
        
        #Determine best edge source
        edge_ratio_rgb_min = np.sum(edges_rgb_min)/(255 * image_pixels)
        edge_ratio_gray = np.sum(edges_gray)/(255 * image_pixels)
        edge_ratio_hue = np.sum(edges_hue)/(255 * image_pixels)
            #edge_rndmns_rgb_min = 
            #edge_rndmns_gray
            #edge_rndmns_hue
        edges = edges_gray  #TEMPORARY
        edge_ratio = np.sum(edges)/(255 * image_pixels)
        edge_percent = edge_ratio * 100
        
        
        #Determine if the edge detection should be repeated and how the parameters should change
        if (edge_ratio > edge_ratio_range[1]) | (edge_ratio < edge_ratio_range[0]):
            
            #Specify repeat and log error
            edge_repeat = True
            edge_error = edge_ratio - edge_ratio_goal
            edge_errors.append(edge_error)
            
            #Update Parameters
            edge_P = edge_ratio/edge_ratio_goal
            edge_I = 0            
            edge_D = 0
            gauss_std = (edge_ratio/edge_ratio_goal) * gauss_std 
            canny_thresh1 = (np.sqrt(edge_ratio/edge_ratio_goal) * canny_thresh1)
            canny_thresh2 = (np.sqrt(edge_ratio/edge_ratio_goal) * canny_thresh2)
            
            #Print Update
            if debug:
                print(f'      ({edge_it}): Ratio = {edge_percent:.2f}%   -   ',\
                      f'Gauss STD = {gauss_std:.2f}; Canny Thresh = {canny_thresh1:.2f}')
                
        else:
            
            #Specify not to repeat
            edge_repeat = False
            
            #Print update
            if debug:
                print(f'      ({edge_it}): Ratio = {edge_percent:.2f}%')
            
        #Iterate iteration counter
        edge_it = edge_it + 1
    
    #Define debug outputs
    outputs = {'rgb_min_smooth': rgb_min_smooth, 'gray_smooth': gray_smooth, 'hue_smooth': hue_smooth, 'edges_rgb_min': edges_rgb_min, 'edges_gray': edges_gray, 'edges_hue': edges_hue}
    
    
    #Define outputs
    return edges, outputs



def extendLine(line_points, xs, ys, debug=True):

    #Rename line_points to l to simplify appearance
    l = line_points
    
    #Calculate line coefficients and define lambda functions 
    l_m = (l[3] - l[1])/(l[2] - l[0])
    l_b = l[1] - (l_m * l[0])
    x_func = lambda y: (y - l_b)/l_m
    y_func = lambda x: x * l_m + l_b
    
    #Define new points - Edge case of vertical line
    if (math.isinf(l_b)):                     
        line_points_extended = [l[0], ys[0], l[2], ys[1]]
    else:
        if (xs is None) and (ys is not None):
            line_points_extended = [int(x_func(ys[0])), ys[0], int(x_func(ys[1])),ys[1]]
        elif (ys is None) and (xs is not None):
            line_points_extended = [xs[0], int(y_func(xs[0])), xs[1], int(y_func(xs[1]))]
        else:
            raise ValueError('Inputs not valid. Must only specify the points for one axis (x or y)')
                        
    #Debug outputs
    if False:
        print(f'Line {l} extended to {line_points_extended}. Func is y = {l_m:.4f} * x + {l_b:.2f}')
        
    #Return output
    return line_points_extended

# lineDetection
#
# Scrip applying the hough transform to detect lines and store them in various forms for further processing
#
# Inputs:
#   1. edges: binary image showing detected edges in the image
#   2. rho (float32): rho parameter for hough transform
#   3. theta
#   4. threshold
#   5. min_line_length
#   6. max_line_gap
#   7. debug (bol) (OPTIONAL): specifies if debug outputs should be generated
#
# Returns:
#   1. lines
#   2. line_points
#   3. line_angles

def lineDetection(edges, rho, theta, threshold, min_line_length, max_line_gap, resize_rows, resize_cols, debug=True):

    #Apply probabilistic hough transform to detect lines for the given parameters
    line_points = cv.HoughLinesP(edges, rho = rho, theta = theta, threshold = threshold, \
                                   minLineLength = min_line_length, maxLineGap = max_line_gap)
        
    #Calculate additional line parameters and store them in a list of dicts
    lines = []
    line_angles = []
    if line_points is not None:
        for i in range(0, len(line_points)):
            
            #Calculate orientation
            points = line_points[i][0]
            angle_deg = np.arctan((points[3] - points[1])/(points[2]-points[0])) * 180/np.pi
            length = np.sqrt((points[2] - points[0])**2 + (points[3] - points[1])**2)
            
            #Calculate edge intercepts for the lines
            if (angle_deg <= 45) and (angle_deg >= -45):
                points_edge = extendLine(points, [0, resize_rows], None, debug=True)
            else:
                points_edge = extendLine(points, None, [0, resize_cols], debug=True) 
            
            
            
            #Store in dict and then list
            line = {'points': points, 'length': length, 'angle_deg': angle_deg, 'points_edge': points_edge}
            lines.append(line)
            line_angles.append(angle_deg)

    #Define output
    return lines, line_points, line_angles

def filterLines(lines, line_points, line_angls, debug = True):
    
    #Convert to nparray for boolean indexing
    line_angles = np.array(line_angles)
    
    #Identify horizontal lines
    hori_angles = (line_angles >= -45) * (line_angles <= 45)
    hori_lines = line_points[hori_angles]
    hori_angles = line_angles[hori_angles]
    
    #Identify vertical lines and convert negative angles to positive ones
    vert_angles = (line_angles < -45) + (line_angles > 45)
    vert_lines = line_points[vert_angles]
    vert_angles = line_angles[vert_angles]
    vert_angles[vert_angles <=0] = vert_angles[vert_angles <=0] + 180  
    
    #Calculate statistics
    vert_angles_mean = np.mean(vert_angles)
    vert_angles_std = np.std(vert_angles)
    hori_angles_mean = np.mean(hori_angles)
    hori_angles_std = np.std(hori_angles)
    
    #Remove extreme outliers (i.e. std > 3)
    max_std = 3
    vert_angles_valid = (vert_angles <= (vert_angles_mean + max_std*vert_angles_std)), \
        * (vert_angles >= (vert_angles_mean - max_std*vert_angles_std))
    vert_angles = vert_angles[vert_angles_valid]
    reject_lines = vert_lines[~vert_angles_valid]
    vert_lines = vert_lines[vert_angles_valid]
    hori_angles_valid = (hori_angles <= (hori_angles_mean + max_std*hori_angles_std)), \
        * (hori_angles >= (hori_angles_mean - max_std*hori_angles_std))
    hori_angles = hori_angles[hori_angles_valid]
    reject_lines = np.concatenate((reject_lines, hori_lines[~hori_angles_valid]), 0)
    hori_lines = hori_lines[hori_angles_valid]




    #Remove redundant lines (i.e. those that are essentially overlapping)
        #print(hori_lines)
    angles_unique_thresh = 0.5 * hough_theta * 180/np.pi
    hori_angles_unique = []
    hori_lines_unique = []
    hori_angles_groups = []
    hori_lines_groups = []
    for h in range(0, len(hori_lines)):

        #Determine if the angle belongs to an existing group
        hori_angle = hori_angles[h]
        hori_angle_diff = [ang - hori_angle for ang in hori_angles_unique]
        unique_angles = np.any([abs(diff) < angles_unique_thresh for diff in hori_angle_diff])
        if not unique_angles:
            hori_angles_unique.append(hori_angle)
            hori_lines_unique.append(hori_lines[h])
            hori_angles_groups.append([hori_angle])
            hori_lines_groups.append([hori_lines[h]])
        else:                         #Place the line in an existing group
            min_ind = np.argmin(abs(np.asarray(hori_angle_diff)))
            hori_angles_groups[min_ind].append(hori_angle)
            hori_lines_groups[min_ind].append(hori_lines[h])


    #print(hori_angles_unique)
    #print(hori_angles_groups)
    #x=y
    #hori_angles = hori_angles_unique
    #hori_lines = hori_lines_unique
    #print(hori_lines)
    
    
 
 
    
    

    
        
        
def calculateProjectiveTransform(vert_lines, hori_lines, gray_smooth, resize_rows, image_scalar, debug = True):
    
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
    