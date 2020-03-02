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
import matplotlib.pyplot as plt
from random import sample

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


def plotLines(image,vert_lines, hori_lines, reject_lines, all_lines=None):
    
    #Two plotting modes
    if (all_lines is None):
        
        for i in range(0, len(reject_lines)):
            l = reject_lines[i]
            image = cv.line(image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv.LINE_8)          
        for i in range(0, len(hori_lines)):
            l = hori_lines[i]
            image = cv.line(image, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv.LINE_8)
        for i in range(0, len(vert_lines)):
            l = vert_lines[i]
            image = cv.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_8)
        
    else:
        
        for i in range(0, len(all_lines)):
            l = all_lines[i]
            image = cv.line(image, (l[0], l[1]), (l[2], l[3]), (255,0,255), 3, cv.LINE_8)  
        
        
    #Return statement
    return image


def calculateInterlineDistance(debug=True):

    x=y
    
def extendLine(line_points, xs, ys, debug=True):

    #Rename line_points to l to simplify appearance
    l = line_points
    
    #Define new points - Edge case of vertical line
    if (l[2] == l[0]):                     
        line_points_extended = [l[0], ys[0], l[2], ys[1]]
    else:
        l_m = (l[3] - l[1])/(l[2] - l[0])
        l_b = l[1] - (l_m * l[0])
        x_func = lambda y: (y - l_b)/l_m
        y_func = lambda x: x * l_m + l_b
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
    line_points_edge = []
    if line_points is not None:
        for i in range(0, len(line_points)):
            
            #Calculate orientation
            points = line_points[i][0]
            if points[2] == points[0]:
                angle_deg = 90
                slope = float('inf') 
                y_intercept = None
            else:
                angle_deg = np.arctan((points[3] - points[1])/(points[2]-points[0])) * 180/np.pi
                slope = np.tan(angle_deg)
                y_intercept = points[1] - (slope * points[0])
            
            #Calculate additional parameters
            length = np.sqrt((points[2] - points[0])**2 + (points[3] - points[1])**2) 
            
            #Calculate edge intercepts for the lines
            if (angle_deg <= 45) and (angle_deg >= -45):
                points_edge = extendLine(points, [0, resize_cols], None, debug=True)
                orientation = 'horizontal'
            else:
                points_edge = extendLine(points, None, [0, resize_rows], debug=True) 
                orientation = 'vertical'
                if (angle_deg <= -45):
                    angle_deg += 180
            
            
            #Store in dict and then list
            line = {'points': points, 'length': length, 'angle_deg': angle_deg, 'slope': slope, 'y_intercept': y_intercept, \
                    'orientation': orientation, 'points_edge': points_edge}
            
            #Update lists
            lines.append(line)
            line_angles.append(angle_deg)
            line_points_edge.append(points_edge)

    #Define output
    return lines, line_points, line_points_edge, line_angles


def groupLines():
    
    x=y



def calculateVanishingPoints(debug=True):
    
    #Iterate through all line pairs and determine approximate vanishing points
    vert_vanish_pts = []
    vert_vanish_xs = []
    vert_vanish_ys = []
    for a in range(0, len(vert_lines)):
        line1 = vert_lines[a]
        for b in range(0, len(vert_lines)):
            if (b != a) and (abs(vert_angles[a] - vert_angles[b]) > 0.1):
                line2 = vert_lines[b]
                print(line2)
                line1_m = (line1[3] - line1[1])/(line1[2] - line1[0])
                line1_b = line1[1] - (line1_m * line1[0])
                    #print(line1), print(line1_m), print(line1_b)
                line2_m = (line2[3] - line2[1])/(line2[2] - line2[0])
                line2_b = line2[1] - (line2_m * line2[0])
                    #print(line2), print(line2_m), print(line2_b)
                if (line1_m != line2_m) and (line1_b != line2_b):
                    vert_vanish_xs.append((line2_b - line1_b)/(line1_m - line2_m))
                    vert_vanish_ys.append(line1_m * vert_vanish_xs[-1] + line1_b)
                    vert_vanish_pts.append([vert_vanish_xs[a], vert_vanish_ys[a]])
                    #print(vp_x), print(vp_y)
                #POSSIBLE DIVIDE BY ZERO ERROR
    vert_vanish_pt = [np.nanmean(vert_vanish_xs), np.nanmean(vert_vanish_ys)]
    hori_vanish_pts = []
    hori_vanish_xs = []
    hori_vanish_ys = []
    for a in range(0, len(hori_lines)):
        line1 = hori_lines[a][0]
        for b in range(0, len(hori_lines)):
            if (b != a) and (abs(hori_angles[a] - hori_angles[b]) > 0.1):
                line2 = hori_lines[b][0]
                line1_m = (line1[3] - line1[1])/(line1[2] - line1[0])
                line1_b = line1[1] - (line1_m * line1[0])
                    #print(line1), print(line1_m), print(line1_b)
                line2_m = (line2[3] - line2[1])/(line2[2] - line2[0])
                line2_b = line2[1] - (line2_m * line2[0])
                    #print(line2), print(line2_m), print(line2_b)
                if (line1_m != line2_m) and (line1_b != line2_b):
                    hori_vanish_xs.append((line2_b - line1_b)/(line1_m - line2_m))
                    hori_vanish_ys.append(line1_m * hori_vanish_xs[-1] + line1_b)
                    hori_vanish_pts.append([hori_vanish_xs[a], hori_vanish_ys[a]])
                    #print(vp_x), print(vp_y)
                #POSSIBLE DIVIDE BY ZERO ERROR
    hori_vanish_pt = [np.nanmean(hori_vanish_xs), np.nanmean(hori_vanish_ys)]
        #print(hori_vanish_pt)
        
    #Return outputs
    return x

def filterOutlierLines(lines, vert_inds, hori_inds, reject_inds, hough_theta, debug = True):
    
    x=y


def filterRedundantLines(lines, vert_inds, hori_inds, reject_inds, hough_theta, debug = True):
 
    #Convert index arrays to lists
    vert_inds = list(vert_inds[0])
    hori_inds = list(hori_inds[0])
    reject_inds = list(reject_inds[0])
    
    #Look for outliers based on axis intercept and angle
    x_intercepts = []
    y_intercepts = []
    vert_angles = []
    hori_angles = []
    vert_lines = []
    hori_lines = []
    for v in vert_inds:
        
        #Pull line and parameters
        line = lines[v]
        x_intercepts.append(line['points_edge'][0])
        vert_angles.append(line['angle_deg'])
        vert_lines.append(line['points'])
        
    for h in hori_inds:
        
        #Pull line and parameters
        line = lines[h]
        y_intercepts.append(line['points_edge'][1])
        hori_angles.append(line['angle_deg'])
        hori_lines.append(line['points'])
    
    #fig = plt.figure()
    #plt.plot(hori_angles, y_intercepts, 'r.')
    #plt.draw()
    #fig.canvas.manager.window.raise_()
    #plt.pause(0.01)
    
    #Filter out lines which diverge considerably from the fit curve.
    
    
    #Identify distinct vertical angles and group lines based on their angle
    angles_thresh = 0.5 * hough_theta * 180/np.pi
    hough_angle_ct = round(np.pi/hough_theta) + 1
    hough_angles = np.linspace(0, 180, hough_angle_ct)
    vert_angles_unique = []
    vert_lines_unique = []
    vert_angles_groups = []
    vert_inds_groups = []
    vert_intercept_groups = []
    vert_lines_groups = []
    vert_lines_lengths = []
    for v in range(0, len(vert_angles)):

        #Determine which discrete hough angle the line is associated with
        vert_ind = vert_inds[v]
        vert_angle = vert_angles[v]
        vert_angle_diff = [ang - vert_angle for ang in hough_angles]
        min_diff_ind = np.argmin(abs(np.asarray(vert_angle_diff)))
        hough_angle = hough_angles[min_diff_ind]
        x_intercept = x_intercepts[v]
        
        #Place the line into its corresponding group
        angle_present = np.any(np.asarray(vert_angles_unique) == hough_angle) 
        if angle_present:
            group_ind = np.where(np.asarray(vert_angles_unique) == hough_angle)
            group_ind = group_ind[0][0]
            vert_angles_groups[group_ind].append(vert_angle)
            vert_lines_groups[group_ind].append(vert_lines[v])
            vert_lines_lengths[group_ind].append(lines[vert_ind]['length'])
            vert_inds_groups[group_ind].append(vert_ind)
            vert_intercept_groups[group_ind].append(x_intercept)
        else:
            vert_angles_unique.append(hough_angle)
            vert_angles_groups.append([vert_angle])
            vert_lines_groups.append([vert_lines[v]])
            vert_lines_lengths.append([lines[vert_ind]['length']])  
            vert_inds_groups.append([vert_ind])
            vert_intercept_groups.append([x_intercept])
    
    
    #print(vert_angles_groups), print(vert_inds_groups), print(vert_intercept_groups), z=y

    #Choose lines which are most representive for unique angles
    vert_angles = []
    vert_lines = []
    vert_inds = []
    for g in range(0, len(vert_angles_unique)):
        
        #Pull line groups and their corresponding lengths
        inds = vert_inds_groups[g]
        angles = vert_angles_groups[g]
        group_lines = vert_lines_groups[g]
        lengths = vert_lines_lengths[g]
        max_length = np.argmax(np.asarray(lengths))
        max_ind = inds[max_length]
        
        #Define the line to use
        vert_inds.append(max_ind)
        vert_lines.append(group_lines[max_length].tolist())
        vert_angles.append(angles[max_length])
        
        #Store the reject lines
        for ind in inds:
            if ind != max_ind:
                reject_inds.append(ind)

     

    #Calculate inter-line distance for remaining valid lines
    valid_inds = vert_inds + hori_inds
    
    
    #Look for overlapping lines
    if len(hori_lines) > 10:
        
        #Group y intercepts of the horizontal lines using a histogram (to ensure adequate representation)
        y_intercept_hist, hist_edges = np.histogram(y_intercepts, bins=15)
        
        #Iterate through bins and compile the properties of the lines in each bin
        bin_angles = []
        bin_intercepts = []
        bin_lengths = []
        bin_inds = []
        bin_lines = []
        for b in range(0, len(hist_edges) - 1):
            
            #Pull lines with y_intercepts in the bin
            upper = hist_edges[b+1]
            lower = hist_edges[b]
            lines_within = np.where((y_intercepts >= lower) * (y_intercepts < upper))
            lines_within = lines_within[0]            
            bin_lines.append([hori_lines[ind] for ind in lines_within])
            
            #Define corresponding line properties
            bin_inds.append([hori_inds[ind] for ind in lines_within])
            bin_lengths.append([lines[ind]['length'] for ind in bin_inds[b]])
            bin_intercepts.append([y_intercepts[ind] for ind in lines_within])
            bin_angles.append([hori_angles[ind] for ind in lines_within])
       
    
        #Calculate aggregates for the line groups
        bin_mean_lengths = []
        bin_std_lengths = []
        bin_mean_intercepts = []
        bin_std_intercepts = []
        bin_mean_angles = []
        bin_std_angles = []
        bin_lengths_diff = []
        bin_intercepts_diff = []
        bin_angles_diff = []
        bin_lengths_order = []
        bin_intercepts_order = []
        bin_angles_order = []
        for i in range(0, len(bin_inds)):
            
            #Calculate aggregates
            bin_mean_lengths.append(np.mean(bin_lengths[i]))
            bin_std_lengths.append(np.std(bin_lengths[i]))
            bin_mean_intercepts.append(np.mean(bin_intercepts[i]))
            bin_std_intercepts.append(np.std(bin_intercepts[i]))
            bin_mean_angles.append(np.mean(bin_angles[i]))
            bin_std_angles.append(np.std(bin_angles[i]))
            
            #Calculate difference of each line from the aggregate for the group
            bin_lengths_diff.append([np.abs(bin_mean_lengths[i] - length) for length in bin_lengths[i]]) 
            bin_intercepts_diff.append([np.abs(bin_mean_intercepts[i] - intercept) for intercept in bin_intercepts[i]]) 
            bin_angles_diff.append([np.abs(bin_mean_angles[i] - angle) for angle in bin_angles[i]])
            
            #Calculate order of lines for each parameter
            bin_lengths_order.append(np.argsort(-1 * np.asarray(bin_lengths[i])))   #NEED TO UPDATE
            bin_intercepts_order.append(np.argsort(bin_intercepts_diff[i]))
            bin_angles_order.append(np.argsort(bin_angles_diff[i]))
        
        
        
        #print(-1 * np.asarray(bin_lengths[i])), print(bin_lengths_order), print(bin_angles_order), print(bin_intercepts_order)
        
        #Calculate rating of each line to selet the representative line
        bin_line_rankings = []
        hori_lines = []
        hori_inds = []
        for i in range(0, len(bin_inds)):
            
            if len(bin_angles_order[i] > 0):
            
                #Calculate line rankings
                bin_line_ranking = 0.45 * bin_angles_order[i] + 0.35 * bin_intercepts_order[i]
                                   #+ 0.0 * bin_lengths_order[i]
                bin_line_rankings.append(list(bin_line_ranking))

                #Define the horizontal output lines to be the line with the lowest ranking 
                best_line_ind = np.argmin(bin_line_ranking)
                hori_inds.append(bin_inds[i][best_line_ind])
                hori_lines.append(bin_lines[i][best_line_ind])
                
                #Record rejected indecies
                for ind in bin_inds[i]:
                    if ind != bin_inds[i][best_line_ind]:
                        reject_inds.append(ind)
                   
        
        
        
        
        #print(bin_mean_intercepts), print(bin_mean_angles)
        
        #print(bin_lengths_diff), print(bin_angles_diff), print(bin_intercepts_diff)
        
        
        
        
        
        
      
        
        #Select the most representative line from the group
    #    for b in range(0,len(bin_inds)):
     #       bin_length_diff = [bin_mean_intercepts[b] - 
     #       bin_intercept_diff
   #         bin_angle_diff = 1
        
        
        
        
    #Compile all values
    reject_lines = []
    for ind in reject_inds:
        reject_lines.append(lines[ind]['points'])
    
    
    #Define outputs
    outputs = {'x_intercepts': x_intercepts, 'vert_angles': vert_angles, 'y_intercepts': y_intercepts, 'hori_angles': hori_angles}
    
    #Return statement
    return vert_inds, vert_lines, vert_angles, hori_inds, hori_lines, hori_angles, reject_inds, reject_lines, outputs



def filterLines(lines, line_points, line_angles, hough_theta, debug = True):
    
    #Convert to nparray for boolean indexing
    line_angles = np.array(line_angles)
    
    #Identify horizontal lines
    hori_inds = np.where((line_angles >= -45) * (line_angles <= 45))
    hori_lines = line_points[hori_inds]
    hori_angles = line_angles[hori_inds]
    hori_lengths = [lines[ind]['length'] for ind in hori_inds[0]]
    
    #Identify vertical lines and convert negative angles to positive ones
    vert_inds= np.where((line_angles < -45) + (line_angles > 45))
    vert_lines = line_points[vert_inds]
    vert_angles = line_angles[vert_inds]
    vert_angles[vert_angles <=0] = vert_angles[vert_angles <=0] + 180  
    vert_lengths = [lines[ind]['length'] for ind in vert_inds[0]]
    
    #Calculate statistics
    vert_angles_mean = np.mean(vert_angles)
    vert_angles_std = np.std(vert_angles)
    hori_angles_mean = np.mean(hori_angles)
    hori_angles_std = np.std(hori_angles)
    vert_lengths_mean = np.mean(vert_lengths)
    vert_lengths_std = np.std(vert_lengths)
    hori_lengths_mean = np.mean(hori_lengths)
    hori_lengths_std = np.std(hori_lengths)
    
    #Remove extreme outliers (i.e. std > 2)
    max_std = 2
    vert_angles_valid = (vert_angles <= (vert_angles_mean + max_std*vert_angles_std)) * (vert_angles >= (vert_angles_mean - \
                                                                                                         max_std*vert_angles_std))
    vert_angles = vert_angles[vert_angles_valid]
    reject_lines = vert_lines[~vert_angles_valid]
    reject_inds = np.where(~vert_angles_valid)
    vert_lines = vert_lines[vert_angles_valid]
    hori_angles_valid = (hori_angles <= (hori_angles_mean + max_std*hori_angles_std)) * (hori_angles >= (hori_angles_mean - \
                                                                                                         max_std*hori_angles_std))
    hori_angles = hori_angles[hori_angles_valid]
    reject_lines = np.concatenate((reject_lines, hori_lines[~hori_angles_valid]), 0)
    reject_inds = np.concatenate((reject_inds, np.where(~hori_angles_valid)), 1)
    hori_lines = hori_lines[hori_angles_valid]

    #Remove redundant lines
    vert_inds, vert_lines, vert_angles, hori_inds, hori_lines, hori_angles, reject_inds, reject_lines, redundant_outputs \
        = filterRedundantLines(lines, vert_inds, hori_inds, reject_inds, hough_theta, debug = debug)
    
    #Define additional outputs
    outputs = {'vert_angles_mean': vert_angles_mean, 'vert_angles_std': vert_angles_std, 'hori_angles_mean': hori_angles_mean, \
               'hori_angles_std': hori_angles_std, 'redundant_outputs': redundant_outputs}
    
    
    #Define outputs
    return vert_inds, vert_lines, vert_angles, hori_inds, hori_lines, hori_angles, reject_inds, reject_lines, outputs 
 
 
    
    

    
        
        
def calculateProjectiveTransform(lines, vert_inds, hori_inds, gray_smooth, resize_rows, image_scalar, sample_ct=3, debug = True):
    
    #TEMPORARY
    debug = False
    
    #Randomly sample from the lists
    vert_lines = [lines[ind] for ind in sample(vert_inds,sample_ct)]
    hori_lines = [lines[ind] for ind in sample(hori_inds, sample_ct)]

    
    #Define variables
    proj_transforms = []
        
    #Iterate through all pairs
    it = 0
    for a in range(0, len(vert_lines)):
        vl1 = vert_lines[a]
        for b in [x for x in range(0, len(vert_lines)) if x != a]:
            vl2 = vert_lines[b]
            for c in range(0, len(hori_lines)):
                hl1 = hori_lines[c]
                for d in [x for x in range(0, len(hori_lines)) if x != c]:
        
                    #Pull second horizontal line
                    hl2 = hori_lines[d]
                    
                    #Pull the edge intercept points (where lines span full lenth of image)
                    l1 = vl1['points_edge']
                    l2 = vl2['points_edge']
                    l3 = hl1['points_edge']
                    l4 = hl2['points_edge']
                    
                    
                    
                    #Redefine them to span entire image
                    if (l1[2] == l1[0]):
                        l1_m = float('inf')
                        l1_b = None
                        l1_func = lambda y: l1[0]
                        l1 = [l1[0], resize_rows, l1[2], 0]
                        print(f'      Line 1 - Coords = {l1}')
                    else:
                        l1_m = (l1[3] - l1[1])/(l1[2] - l1[0])
                        l1_b = l1[1] - (l1_m * l1[0])
                        l1_func = lambda y: (y - l1_b)/l1_m
                        l1 = [int(l1_func(resize_rows)), resize_rows, int(l1_func(0)),0]
                        print(f'      Line 1 - Coords = {l1}')
                        #l2 = vert_lines_samples[1][0]
                    if (l2[2] == l2[0]):
                        l2_m = float('inf')
                        l2_b = None
                        l2_func = lambda y: l1[0]
                        l2 = [l2[0], resize_rows, l2[2], 0]
                    else:
                        l2_m = (l2[3] - l2[1])/(l2[2] - l2[0])
                        l2_b = l2[1] - (l2_m * l2[0])
                        l2_func = lambda y: (y - l2_b)/l2_m
                        l2 = [int(l2_func(resize_rows)), resize_rows, int(l2_func(0)),0]

                    
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
                    if l1[2] == l1[0]:
                        l1l3_x = l1[0]
                        l1l4_x = l1[0]
                        l2l3_x = (l3_b - l2_b)/(l2_m - l3_m)
                        l2l4_x = (l4_b - l2_b)/(l2_m - l4_m)
                    if l2[2] == l2[0]:
                        l2l3_x = l2[0]
                        l2l4_x = l2[0]   
                        l2l3_x = (l3_b - l2_b)/(l2_m - l3_m)
                        l2l4_x = (l4_b - l2_b)/(l2_m - l4_m)
                    else:
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
                    proj_trans = np.asarray(proj_trans)
                    
                    
                    #Store Transform
                    if it == 0:
                        proj_transforms.append(proj_trans)
                    else:
                        
                        proj_transforms.append(proj_trans)
                    
                    
                    #proj_transforms.append(proj_trans)
        
        
        
                    #Iteration counter
                    it += 1
        
    #Calculate mean projective transform
    
    
    #Define outputs
    proj_transforms = proj_trans
    rot_ang = l3_ang
    return proj_transforms, rot_ang    
        
#def correctPerspective():
    