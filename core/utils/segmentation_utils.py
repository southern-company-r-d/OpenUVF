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
    

def edgeDetection(rgb, rgb_min, gray, hue, image_pixels, params=None, debug=True):
   
    #Print update
    if debug:
        print('   Edge Detection:')
    
    #Pull or define parameters
    if params is None:
        gauss_size = (9,9)
        gauss_std = 3;
        canny_thresh_min = 100
        canny_thresh_max = 200        
        edge_ratio_goal = 0.005 # 0.5% of total image area
        edge_ratio_range = (0.0025, 0.025)
    else:
        gauss_size = params['gauss_size']
        gauss_std = params['gauss_std']
        canny_thresh_min = params['canny_thresh_min']
        canny_thresh_max = params['canny_thresh_max']
        edge_ratio_goal = params['edge_ratio_goal']
        edge_ratio_range = params['edge_ratio_range']
       
    #Calculate image histogram and cumulative distribution function
    gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    gray_hist_cumsum = np.cumsum(gray_hist)/np.sum(gray_hist)
    #print(gray_hist), print(gray_hist_cumsum)
    #plt.figure(figsize=[18, 9.5])
    #plt.plot(range(0,256), gray_hist)   
    #plt.figure(figsize=[18, 9.5])
    #plt.plot(range(0,256), gray_hist_cumsum) 
    
    #Calculate threshold
    
    
    #Calculate laplacians
    
    
    #Calculate edge detection threshhold 
    image_mean = np.mean(gray.ravel())
    peak_val = np.argmax(gray_hist)
    peak_left = gray_hist_cumsum[peak_val]
    peak_right = 1 - peak_left
    #canny_thresh1 = np.min(np.where(gray_hist_cumsum > 0.50))
    #canny_thresh2 = 200   #np.max(np.where(gray_hist_cumsum <= 0.99)) + 1
    #print(canny_thresh1), print(canny_thresh2), print(image_mean), print(peak_val), print(peak_left), print(peak_right)
    
    
    #Iteratively refine edge detection until the proportion of edges falls into a prespecified range
    edge_errors = []
    edge_repeat = True
    edge_it = 0    
    while edge_repeat & (edge_it < 10):
        
        #Apply Gaussian Blur
        hue_smooth = cv.GaussianBlur(hue, gauss_size, sigmaX=gauss_std, sigmaY=gauss_std)
        rgb_min_smooth = cv.GaussianBlur(rgb_min, gauss_size, sigmaX=gauss_std, sigmaY=gauss_std)
        gray_smooth = cv.GaussianBlur(gray, gauss_size, sigmaX=gauss_std, sigmaY=gauss_std)

        #Canny edge detection
        edges_rgb_min = cv.Canny(rgb_min_smooth, 0.5 * canny_thresh_min, 0.5 * canny_thresh_max)
        edges_gray = cv.Canny(gray, canny_thresh_min, canny_thresh_max)
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
            
            #Handle case of no edges detected
            
            
            #Update Parameters
            edge_P = edge_ratio/edge_ratio_goal
            edge_I = 0            
            edge_D = 0
            gauss_std = np.sqrt(edge_ratio/edge_ratio_goal) * gauss_std 
            #canny_thresh_min = (np.sqrt(edge_ratio/edge_ratio_goal) * canny_thresh_min)
            #canny_thresh_max = (np.sqrt(edge_ratio/edge_ratio_goal) * canny_thresh_max)
            
            #Print Update
            if debug:
                print(f'      ({edge_it}): Ratio = {edge_percent:.2f}%   -   ',\
                      f'Gauss STD = {gauss_std:.2f}; Canny Thresh = {canny_thresh_min:.2f}')
                
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


def correctDistortion(debug=True):
    
    x=y



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
                slope = (points[3] - points[1])/(points[2]-points[0])
                angle_deg = np.arctan(slope) * 180/np.pi
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
    
    
def calculateLineIntercept(l1, l2, hough_theta, debug = True):
    
    #Pull parameters
    l1m = l1['slope']
    l1b = l1['y_intercept']
    l2m = l2['slope']
    l2b = l2['y_intercept']
    
    #Calculate intercepts
    if (l1m != l2m):         #Account for parallel lines
        
        #Break solution into 5 cases
        if (l1b is None):
            l1l2_x = l1['points'][0]
            l1l2_y = l2m * l1l2_x + l2b            
        elif (l2b is None):
            l1l2_x = l2['points'][0]
            l1l2_y = l1m * l1l2_x + l1b             
        else:
            l1l2_x = (l2b - l1b)/(l1m - l2m)
            l1l2_y = l1m * l1l2_x + l1b
            
            
    else: #No intercept
        l1l2_x = None
        l1l2_y = None
    
    #Return outputs
    return l1l2_x, l1l2_y


def calculateVanishingPoints(vanish_lines, inds, hough_theta, debug=True):
    
    #Pull lines
    lines = [vanish_lines[ind] for ind in inds]
    angles = [vanish_lines[ind]['angle_deg'] for ind in inds]
    
    #Calculate Vertical Vanishing Point
    vanish_pts = []
    vanish_xs = []
    vanish_ys = []
    vanish_inds = []
    for a in range(0, len(lines)):
        l1 = lines[a]
        for b in range(0, len(lines)):
            if (b != a) and (abs(angles[a] - angles[b]) > 0.1):
                
                #Pull comparison line
                l2 = lines[b]
                vanish_inds.append([inds[a],inds[b]])
                
                #Calculate line intersection
                if (abs(l1['angle_deg'] - l2['angle_deg']) > hough_theta):
                    x, y = calculateLineIntercept(l1, l2, hough_theta, debug=debug)
                
                    #Log the intersection if they exist
                    if (x is not None) and (y is not None):
                        vanish_xs.append(x)
                        vanish_ys.append(y)
                        vanish_pts.append([x, y])

    #Calculate mean vanishing point and filter outliers
    mean_vanish_pt = [np.nanmean(vanish_xs), np.nanmean(vanish_ys)]
    vanish_dists = []
    for pt in vanish_pts:
        
        #Calculate distance of each vanishing point from the first order mean
        dist = np.sqrt((pt[0] - mean_vanish_pt[0])**2 + (pt[1] - mean_vanish_pt[1])**2)
        vanish_dists.append(dist)
        
    vanish_dists_mean = np.nanmean(vanish_dists)
    vanish_dists_std = np.nanstd(vanish_dists)
    valid_pairs = np.where((vanish_dists - vanish_dists_mean) < (1.5 * vanish_dists_std))
    valid_xs = [vanish_xs[ind] for ind in valid_pairs[0]]
    valid_ys = [vanish_ys[ind] for ind in valid_pairs[0]]
    mean_vanish_pt = [np.nanmean(valid_xs), np.nanmean(valid_ys)]
    
 
    
    #Determine if the horizontal lines will reasonably converge
    
    
    
    #Calculate Horizontal Vanishing Point (First Determine if it can be reasonably found)
#    hori_vanish_pts = []
#    hori_vanish_xs = []
#    hori_vanish_ys = []
#    for a in range(0, len(hori_lines)):
#        line1 = hori_lines[a][0]
#        for b in range(0, len(hori_lines)):
#            if (b != a) and (abs(hori_angles[a] - hori_angles[b]) > 0.1):
#                line2 = hori_lines[b][0]
#                line1_m = (line1[3] - line1[1])/(line1[2] - line1[0])
#                line1_b = line1[1] - (line1_m * line1[0])
#                    #print(line1), print(line1_m), print(line1_b)
#                line2_m = (line2[3] - line2[1])/(line2[2] - line2[0])
#                line2_b = line2[1] - (line2_m * line2[0])
#                    #print(line2), print(line2_m), print(line2_b)
#                if (line1_m != line2_m) and (line1_b != line2_b):
#                    hori_vanish_xs.append((line2_b - line1_b)/(line1_m - line2_m))
#                    hori_vanish_ys.append(line1_m * hori_vanish_xs[-1] + line1_b)
#                    hori_vanish_pts.append([hori_vanish_xs[a], hori_vanish_ys[a]])
#                    #print(vp_x), print(vp_y)
#                #POSSIBLE DIVIDE BY ZERO ERROR
#    hori_vanish_pt = [np.nanmean(hori_vanish_xs), np.nanmean(hori_vanish_ys)]
        #print(hori_vanish_pt)
        
    #Return outputs  
    return mean_vanish_pt, vanish_pts, vanish_xs, vanish_ys, vanish_inds 













def filterOutlierLines(lines, vert_inds, hori_inds, reject_inds, hough_theta, debug = True):
    
    x=y

def vanishingPointLineFilter(lines, inds, reject_inds, hough_theta, pair_ct=1, debug=True):
    
    #Calculate vanishing points
    vanish_pt, vanish_pts, vanish_xs, vanish_ys, vanish_inds = calculateVanishingPoints(lines, inds, hough_theta, debug=debug)
    
    #Calculate distance of the from the mean vanishing point to all others
    vanish_dists = []
    for pt in vanish_pts:
        
        #Calculate distance of each vanishing point from the first order mean
        dist = np.sqrt((pt[0] - vanish_pt[0])**2 + (pt[1] - vanish_pt[1])**2)
        vanish_dists.append(dist)
    
    #Select the line pair
    try:   #TEMPORARY
        repr_inds = vanish_inds[np.argmin(vanish_dists)]
        repr_lines = [lines[ind]['points'].tolist() for ind in repr_inds]
    except Exception:
        repr_inds = vanish_inds
        repr_lines = [lines[ind]['points'].tolist() for ind in repr_inds]
    
    #Update reject_inds
    for ind in inds:
        if ind not in repr_inds:
            reject_inds.append(ind)
    
    #Define outputs
    return repr_inds, repr_lines, vanish_pt, vanish_pts, vanish_xs, vanish_ys, vanish_inds, reject_inds 
    
    
def histogramLineFilter(lines, line_points, line_angles, line_inds, ax_intercepts, reject_inds, bin_ct, debug=True):
                        
    #Group axis intercepts of the lines using a histogram (to ensure adequate representation)
    ax_intercept_hist, hist_edges = np.histogram(ax_intercepts, bins=bin_ct)

    #Iterate through bins and compile the properties of the lines in each bin
    bin_angles = []
    bin_intercepts = []
    bin_lengths = []
    bin_inds = []
    bin_lines = []
    for b in range(0, bin_ct):

        #Pull lines with y_intercepts in the bin
        upper = hist_edges[b+1]
        lower = hist_edges[b]
        lines_within = np.where((ax_intercepts >= lower) * (ax_intercepts < upper))
        lines_within = lines_within[0]
        bin_lines.append([line_points[ind] for ind in lines_within])

        #Define corresponding line properties
        bin_inds.append([line_inds[ind] for ind in lines_within])
        bin_lengths.append([lines[ind]['length'] for ind in bin_inds[b]])
        bin_intercepts.append([ax_intercepts[ind] for ind in lines_within])
        bin_angles.append([line_angles[ind] for ind in lines_within])

        
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
    for i in range(0, bin_ct):

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
        bin_lengths_order.append(np.argsort(np.asarray(bin_lengths[i])))   #NEED TO UPDATE
        bin_intercepts_order.append(np.argsort(bin_intercepts_diff[i]))
        bin_angles_order.append(np.argsort(bin_angles_diff[i]))



    #print(-1 * np.asarray(bin_lengths[i])), print(bin_lengths_order), print(bin_angles_order), print(bin_intercepts_order)


    
    
    #Calculate rating of each line to selet the representative line
    bin_line_rankings = []
    output_points = []
    output_inds = []
    output_intercepts = []
    for i in range(0, len(bin_inds)):

        if len(bin_angles_order[i] > 0):

            #Pull orders and values and convert to float
            angles_order = np.asarray(bin_angles_order[i]) + 1
            intercepts_order = np.asarray(bin_intercepts_order[i]) + 1
            lengths = np.asarray(bin_lengths[i])
            
            #Calculate line rankings
            bin_line_ranking = 0.3 * angles_order + 0.35 * intercepts_order + 0.35 * (1 - lengths/np.max(lengths))
            bin_line_rankings.append(list(bin_line_ranking))

            
            #Define the horizontal output lines to be the line with the lowest ranking 
            best_line_ind = np.argmin(bin_line_ranking)
            output_inds.append(bin_inds[i][best_line_ind])
            output_points.append(bin_lines[i][best_line_ind])
            output_intercepts.append(bin_intercepts[i][best_line_ind])

            #Record rejected indecies
            for ind in bin_inds[i]:
                if ind != bin_inds[i][best_line_ind]:
                    reject_inds.append(ind)
                   
        
    return output_points, output_inds, output_intercepts, reject_inds
    
    
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
    x_intercepts = []
    for g in range(0, len(vert_angles_unique)):
        
        #Pull line groups and their corresponding lengths
        inds = vert_inds_groups[g]
        angles = vert_angles_groups[g]
        intercepts = vert_intercept_groups[g]
        group_lines = vert_lines_groups[g]
        lengths = vert_lines_lengths[g]
        max_length = np.argmax(np.asarray(lengths))
        max_ind = inds[max_length]
        
        #Define the line to use
        vert_inds.append(max_ind)
        vert_lines.append(group_lines[max_length].tolist())
        vert_angles.append(angles[max_length])
        x_intercepts.append(intercepts[max_length])
        
        #Store the reject lines
        for ind in inds:
            if ind != max_ind:
                reject_inds.append(ind)

     

    #Calculate inter-line distance for remaining valid lines
    valid_inds = vert_inds + hori_inds
    
    
    #Histogram filter to remove final overlapping lines
    hist_bin_ct = 20
    if len(hori_lines) > 10:
        
        #Filter horizontal lines - First round 
        hori_lines, hori_inds, y_intercepts, reject_inds = histogramLineFilter(lines, hori_lines, hori_angles, hori_inds, y_intercepts,\
                                                                 reject_inds, hist_bin_ct, debug=debug)
        
        #Filter vertical lines - First round histogram based
        vert_lines, vert_inds, x_intercepts, reject_inds = histogramLineFilter(lines, vert_lines, vert_angles, vert_inds, x_intercepts,\
                                                                 reject_inds, hist_bin_ct, debug=debug)  
    
    #Define outputs
    outputs = {'x_intercepts': x_intercepts, 'vert_angles': vert_angles, 'y_intercepts': y_intercepts, 'hori_angles': hori_angles}
    
    #Return statement
    return vert_inds, vert_lines, vert_angles, hori_inds, hori_lines, hori_angles, reject_inds, y_intercepts, outputs



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
    vert_inds, vert_lines, vert_angles, hori_inds, hori_lines, hori_angles, reject_inds, y_intercepts, redundant_outputs \
        = filterRedundantLines(lines, vert_inds, hori_inds, reject_inds, hough_theta, debug = debug)
    
    #Select representative horizontal lines
    hori_lines, hori_inds, y_intercepts, reject_inds = histogramLineFilter(lines, hori_lines, hori_angles, hori_inds, y_intercepts,\
                                                                 reject_inds, 2, debug=debug) 
    
    #Select representative vertical lines
    vert_inds, vert_lines, vert_vanish_pt, vert_vanish_pts, vert_vanish_xs, vert_vanish_ys, vert_vanish_inds, reject_inds = \
    vanishingPointLineFilter(lines, vert_inds, reject_inds, hough_theta, debug=debug)
    
    #Compile reject lines
    reject_lines = []
    for ind in reject_inds:
        reject_lines.append(lines[ind]['points'])    
    
    #Define additional outputs
    outputs = {'vert_angles_mean': vert_angles_mean, 'vert_angles_std': vert_angles_std, 'hori_angles_mean': hori_angles_mean, \
               'hori_angles_std': hori_angles_std, 'vert_vanish_pt': vert_vanish_pt, 'vert_vanish_pts':vert_vanish_pts, \
               'vert_vanish_xs':vert_vanish_xs, 'vert_vanish_ys':vert_vanish_ys, 'vert_vanish_inds':vert_vanish_inds, \
               'redundant_outputs': redundant_outputs}    
    
    #Define outputs
    return vert_inds, vert_lines, vert_angles, hori_inds, hori_lines, hori_angles, reject_inds, reject_lines, outputs 
   
    
    
        
def projectiveTransform(image, lines, vert_inds, hori_inds, gray_smooth, resize_rows, image_scalar, hough_theta, sample_ct=2, debug = True):
    
    #Check that there are sufficient lines
    #if (len(vert_inds) < 2) or (len(hori_inds) < 2):
        
    
        
    #Encapsulate in try except statement to handle errors
    try:
        #Randomly sample from the lists
        vert_lines = [lines[ind] for ind in vert_inds]
        hori_lines = [lines[ind] for ind in hori_inds]

        #Redefine lines
        vl1 = vert_lines[0]
        vl2 = vert_lines[1]
        hl1 = hori_lines[0]
        hl2 = hori_lines[1]

        #Pull the edge intercept points (where lines span full lenth of image)
        l1 = vl1['points_edge']
        l2 = vl2['points_edge']
        l3 = hl1['points_edge']
        l4 = hl2['points_edge']

        #Pull line parameters
        l1_m = vl1['slope']
        l1_b = vl1['y_intercept']
        l2_m = vl2['slope']
        l2_b = vl2['y_intercept']
        l3_m = hl1['slope']
        l3_b = hl1['y_intercept']
        l4_m = hl2['slope']
        l4_b = hl2['y_intercept']

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
        l1l3_x, l1l3_y = calculateLineIntercept(vl1, hl1, hough_theta, debug)
        l1l4_x, l1l4_y = calculateLineIntercept(vl1, hl2, hough_theta, debug)
        l2l3_x, l2l3_y = calculateLineIntercept(vl2, hl1, hough_theta, debug)
        l2l4_x, l2l4_y = calculateLineIntercept(vl2, hl2, hough_theta, debug)
        l1l3_pt = [l1l3_x, l1l3_y]
        l1l4_pt = [l1l4_x, l1l4_y]
        l2l3_pt = [l2l3_x, l2l3_y]
        l2l4_pt = [l2l4_x, l2l4_y]
        #l1_len = np.sqrt((l1l3_pt[1] - l1l4_pt[1])**2 + (l1l3_pt[0] - l1l4_pt[0])**2)
        #l2_len = np.sqrt((l2l3_pt[1] - l2l4_pt[1])**2 + (l2l3_pt[0] - l2l4_pt[0])**2)
        #l3_len = np.sqrt((l1l3_pt[1] - l2l3_pt[1])**2 + (l1l3_pt[0] - l2l3_pt[0])**2)
        #l4_len = np.sqrt((l1l4_pt[1] - l2l4_pt[1])**2 + (l1l4_pt[0] - l2l4_pt[0])**2)
        #l1l3_ang = np.arctan((l1l3_pt[1] - l1l4_pt[1])/(l1l3_pt[0] - l1l4_pt[0])) * 180/np.pi
        #l1l4_ang = np.arctan((l2l3_pt[1] - l2l4_pt[1])/(l2l3_pt[0] - l2l4_pt[0])) * 180/np.pi
        #l2l3_ang = np.arctan((l1l3_pt[1] - l2l3_pt[1])/(l1l3_pt[0] - l2l3_pt[0])) * 180/np.pi
        #l2l4_ang = np.arctan((l1l4_pt[1] - l2l4_pt[1])/(l1l4_pt[0] - l2l4_pt[0])) * 180/np.pi

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
            gray_sample_box = cv.line(gray_sample_box, (int(l1_box[0]), int(l1_box[1])), (int(l1_box[2]), int(l1_box[3])),\
                                      (0,255,128), 3, cv.LINE_4)
            gray_sample_box = cv.line(gray_sample_box, (int(l2_box[0]), int(l2_box[1])), (int(l2_box[2]), int(l2_box[3])),\
                                      (0,255,128), 3, cv.LINE_4)
            gray_sample_box = cv.line(gray_sample_box, (int(l3_box[0]), int(l3_box[1])), (int(l3_box[2]), int(l3_box[3])),\
                                      (0,255,128), 3, cv.LINE_4)
            gray_sample_box = cv.line(gray_sample_box, (int(l4_box[0]), int(l4_box[1])), (int(l4_box[2]), int(l4_box[3])),\
                                      (0,255,128), 3, cv.LINE_4)

        #Define the moving (reference) points    
        moving_pts = np.float32([[l1l3_pt], [l1l4_pt], [l2l3_pt], [l2l4_pt]])

        #Make second vertical line parallel to first
        l2_new_m = l1_m
        if np.isinf(l2_new_m):
            l2l4_new_x = l2l3_x
            l2l4_new_y = l4_m * l2l4_new_x + l4_b
            l2l4_new_pt = [l2l4_new_x, l2l4_new_y]
        else:
            l2_new_b = l2l3_y - l2_new_m * l2l3_x
            l2l4_new_x = (l2_new_b - l4_b)/(l4_m - l2_new_m)
            l2l4_new_y = l2_new_m * l2l4_new_x + l2_new_b
            l2l4_new_pt = [l2l4_new_x, l2l4_new_y]

        l2_new_box = l2l3_pt + l2l4_new_pt
        if debug:

            gray_sample_box = cv.line(gray_sample_box, (int(l2_new_box[0]), int(l2_new_box[1])), (int(l2_new_box[2]), int(l2_new_box[3])),\
                                      (0,128,256), 3, cv.LINE_4)
            gray_sample_box = cv.line(gray_sample_box, (int(l2l4_x), int(l2l4_y)), (int(l2_new_box[2]), int(l2_new_box[3])),\
                                      (0,128,256), 3, cv.LINE_4)

        #Make second horizontal line parallel to first
        l4_new_m = l3_m
        l4_new_b = l1l4_y - l4_new_m * l1l4_x
        if np.isinf(l2_new_m):
            l2l4_new_x = l2l4_x
            l2l4_new_y = l4_new_m * l2l4_new_x + l4_new_b
        else:
            l2l4_new_x = (l2_new_b - l4_new_b)/(l4_new_m - l2_new_m)
            l2l4_new_y = l2_new_m * l2l4_new_x + l2_new_b

        l2l4_new_pt = [l2l4_new_x, l2l4_new_y]
        l4_new_box = l1l4_pt + l2l4_new_pt
        l2_new_box = l2l3_pt + l2l4_new_pt
        if debug:
            gray_sample_box = cv.line(gray_sample_box, (int(l2_new_box[0]), int(l2_new_box[1])), (int(l2_new_box[2]), int(l2_new_box[3])),\
                                      (256,128,0), 3, cv.LINE_4)
            gray_sample_box = cv.line(gray_sample_box, (int(l4_new_box[0]), int(l4_new_box[1])), (int(l4_new_box[2]), int(l4_new_box[3])),\
                                      (256,128,0), 3, cv.LINE_4)

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
            gray_sample_box = cv.line(gray_sample_box, (int(l1_new_box[0]), int(l1_new_box[1])), (int(l1_new_box[2]), int(l1_new_box[3])),\
                                      (128,0,256), 3, cv.LINE_4)
            gray_sample_box = cv.line(gray_sample_box, (int(l2_new_box[0]), int(l2_new_box[1])), (int(l2_new_box[2]), int(l2_new_box[3])),\
                                      (128,0,256), 3, cv.LINE_4)
            gray_sample_box = cv.line(gray_sample_box, (int(l4_new_box[0]), int(l4_new_box[1])), (int(l4_new_box[2]), int(l4_new_box[3])),\
                                      (128,0,256), 3, cv.LINE_4)  

        #Projective Transformation
        moving_pts = (1/image_scalar) * moving_pts
        fixed_pts = (1/image_scalar) * np.float32([[l1l3_pt], [l1l4_new_pt], [l2l3_pt], [l2l4_new_pt]]) 
        proj_trans = cv.getPerspectiveTransform(moving_pts, fixed_pts)
        proj_trans = np.asarray(proj_trans)


        #Store Transform
        #proj_transforms.append(proj_trans)



        #Select representative Projective Transform
        #proj_transform_means = np.zeros((3,3))
        #proj_transform_stds = np.zeros((3,3))
        #for r in range(0, 3):
        #    for c in range(0, 3):
        #        values = []
        #        for i in range(0, it):
        #            values.append(proj_transforms[i][r][c])
        #            
        #        #Calculate statistics for the list of projective transforms
        #        proj_transform_means[r][c] = np.mean(values) 
        #        proj_transform_stds[r][c] = np.std(values)

        #print(proj_transform_means), print(proj_transform_stds)
        #proj_transform = proj_transform_means

        #Perform Projective Transform
        proj_transform = proj_trans
        image_rows, image_cols, channels = image.shape
        transed = cv.warpPerspective(image, proj_trans, (image_cols, image_rows))

        #Rotate image
        rot_ang = l3_ang    
        rotate_trans = cv.getRotationMatrix2D(tuple(np.array([image_rows, image_cols])/2), rot_ang, 1.0)
        transed = cv.warpAffine(transed, rotate_trans, (image_cols, image_rows))
        
        #Create empty images if not created earlier
        if not debug: 
            gray_sample_lines = []
            gray_sample_box = []
        
        #Print update
        print('      Perspective correction successful!')
        
    except Exception:
        
        transed = image
        proj_transform = []
        rot_ang = 0
        gray_sample_lines = []
        gray_sample_box = []       
        
        #Print update
        print(f'      Perspective correction failed with {Exception}')
    
     
    #Return outputs
    return transed, proj_transform, rot_ang, gray_sample_lines, gray_sample_box    
        
#def correctPerspective():
    