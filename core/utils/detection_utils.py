# pipeline_utils.py
#
# Braden Gilleland - Southern Company Services R&D 2018
#
# This script collects utilities used for OpenUVF's detection pipeline.

import os
import re
import csv
import numpy as np
import pandas as pd
import array as arr
import cv2 as cv
from matplotlib import pyplot as plt
from core.object_detection.utils import visualization_utils as vis_util

# image_pipeline
#
# This script develops an ordered list of UVF images to process in a detection pipeline. 
#
# Inputs:
#   1. pipeline_type (str): specifies what type of inputs are expected. 
#        - Options:
#             1. 'modules-cells' - inputs are cells, from which the pipeline assembles images #                    belonging to the same modules for processing and later assembly.
#             2. 'modules' - inputs are images of modules
#             3. 'cells' - inputs are images of cells. expected output are cells
#   2. images_dir (str): specifies directory where images are stored
#   3. cell_counts (int array) (OPTIONAL): array specifying number of cells in the module such as [vertical, horizontal]
#   4. max_batch_size (int): specifies the maximum amount of images that can be processed in a batch



def image_pipeline(pipeline_type, images_dir, shard_images_count, cell_counts=[12, 6], max_batch_size = 72, module_spec='A\d\d\d\d\d-\d', image_ext='.PNG', inc_incmplt_modules=False):
    
    #List contents of directory
    images_list = os.listdir(images_dir)
    
    #Handling 'cells' type
    if pipeline_type == 'cells' or pipeline_type == 'modules':
        
        #Iterate through directory
        i = 0
        b = 0
        batch = []
        output_list = []
        for image_path in images_list:
            
            #Only consider if it is an image of the specified type
            if image_path.endswith(image_ext):
                
                #Adding image to batch
                if not image_path in batch:
                    batch.append(image_path)
                    
                #Adding batch to list and clearing batch if it reaches max    
                if len(batch) >= (max_batch_size - 1):
                    #Add batch to list
                    output_list.append(batch)
                    
                    #Clear batch
                    batch = []
                elif i == len(images_list) - 2:
                    #Add batch to list
                    output_list.append(batch)
            
            #Tracking which index we are on inside list
            i+=1
        
        return output_list
    
    elif pipeline_type == 'modules-cells':
                    
        #Identify modules represented in directory
        modules = []
        for image_path in images_list:

            #Pull matching module label
            try:
                #Identify module name
                module = re.findall(module_spec, image_path)[0]

                #Check if module name is already stored, otherwise add it
                if not module in modules:
                    modules.append(module)

            except Exception:
                #Ignore path if it does not follow image label spec
                continue
        
        print(modules)
        #Identify cells that belong to modules
        output_list = []
        module_cell_counts = []
        for module in modules:
            
            
            module_cells = [image_path for image_path in images_list if module in image_path];

            module_cell_counts.append(len(module_cells))
            
            #Only output if it has the prerequisite cell count 
            if len(module_cells) == (cell_counts[1] * cell_counts[0]):
                output_list.append(module_cells)
            elif inc_incmplt_modules:
                 output_list.append(module_cells)                  
            
        
        print(module_cell_counts)
        return output_list
    
    
    
def assemble_module(cell_images, image_list, outputs, cell_counts=[12,6], cell_spec='_C'):
    
    #Check if cell_images is complete
    if len(cell_images) == (cell_counts[1] * cell_counts[0]):
        
        ## IMAGES ##
        
        #Define output image size 
        cell_rows = cell_images[1].shape[0]
        cell_cols = cell_images[1].shape[1]
        cell_depth = cell_images[1].shape[2]
        image_rows = cell_rows * cell_counts[1]
        image_cols = cell_cols * cell_counts[0]
        cell_counts_total = cell_counts[0]*cell_counts[1]
        
        #Defining image extension
        image_ext = os.path.splitext(image_list[0])[-1]
        
        #Define module name - assumes only one module worth of images are input
        module_name = os.path.splitext(image_list[0])[0]
        module_name = module_name.split(cell_spec)[0]        
        
        #Index cells
        cell_list = []
        for image_filename in image_list:
            
            #Remove file extension
            image_filename = os.path.splitext(image_filename)[0]
            
            #Isolate cell label - assumes it is the entire element after the cell_spec
            cell_label = image_filename.split(cell_spec)[-1]
            cell_label = cell_spec + cell_label
            
            #Store cell label in list
            cell_list.append(cell_label)
        
        #Iterate through rows and columns, building rows w. hstacking and cols w/ vstacking 
        num_detections_per_output = outputs[0]['num_detections']
        output_detections = dict(
            num_detections=(cell_counts[0]*cell_counts[1]*num_detections_per_output),
            detection_boxes=[],
            detection_scores=[],
            detection_classes=[])
            
            
        for r in range(cell_counts[0]):
            current_row = []
            for c in range(cell_counts[1]):
                
                #Define Cell Number - Assumes numbered from top left going right row-by-row
                cell_number = r * cell_counts[1] + c + 1
                cell_label = cell_spec + str(cell_number)
                
                #Identify index of cell
                cell_ind = cell_list.index(cell_label)
                
                #Pull corresponding cell image
                cell_image = cell_images[cell_ind]
                
                #Build Image row
                current_row.append(cell_image)
                
                #Pull outputs
                output = outputs[cell_ind]              
                
                #Define cell position in module
                cell_ymin = c * cell_cols
                cell_xmin = r * cell_rows
                
                #Define crack position
                i=0
                for detection_box in output['detection_boxes']:

                    #Defining approximate normal values of bounding box corners
                    xmin = (cell_xmin + (detection_box[0] * cell_cols))/image_cols
                    ymin = (cell_ymin + (detection_box[1] * cell_rows))/image_rows
                    xmax = (cell_xmin + (detection_box[2] * cell_cols))/image_cols
                    ymax = (cell_ymin + (detection_box[3] * cell_rows))/image_rows
                    
                    #Update detection_boxes - REVERSED 
                    detection_box_new = [xmin, ymin, xmax, ymax]
                    output['detection_boxes'][i] = detection_box_new
                    
                    #Tracking index
                    i+=1
                
                #Append elements of the output to the sum output
                if r == 0 and c == 0:
                    output_detections['detection_boxes'] = output['detection_boxes']
                    output_detections['detection_scores'] = output['detection_scores']
                    output_detections['detection_classes'] = output['detection_classes']
                else:
                    output_detections['detection_boxes'] = np.vstack((output_detections['detection_boxes'], output['detection_boxes']))
                    output_detections['detection_scores'] = np.concatenate((output_detections['detection_scores'], output['detection_scores']))
                    output_detections['detection_classes'] = np.concatenate((output_detections['detection_classes'], output['detection_classes']))
                
                    
                
                
                #output_detections['detection_boxes'].append(output['detection_boxes'].tolist())
                #print(output_detections)
                
                
            #Convert current_row (list) to tuple
            current_row = tuple(current_row)
            
            #Build current row image
            current_row = np.hstack(current_row)
            
            #Build image
            if r == 0:
                output_image = current_row
            else:
                output_image = np.vstack((output_image, current_row))
            
        
    
        ## OUTPUTS ##
        image_list = [module_name + image_ext]
        output_image = [output_image]
        output_detections = [output_detections]
        return output_image, image_list, output_detections
    
    else:
        return cell_images, image_list, outputs
                
                
def build_visualization(np_images, outputs, category_index, probability_thresh):
    
    #Prepare output/visualization images
    np_images_labeled = []
    np_im = 0
    max_boxes = len(outputs[0]['detection_classes'])
    for image_np in np_images:

        #Pulling output
        output = outputs[np_im]

        #Add visualization boxes and labels
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output['detection_boxes'],
            output['detection_classes'],
            output['detection_scores'],
            category_index,
            instance_masks=output.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=probability_thresh,
            max_boxes_to_draw=max_boxes)

        #Append image 
        np_images_labeled.append(image_np)  
        np_im+=1
    
    return np_images
    
    
# initialize_statistics_dict
#
# This function initializes a dictionary that stores 
    
def initialize_statistics_dict(cell_counts=[12,6]):
    
    #Define statistics
    statistics = dict(
        plant_num_cells=0,
        plant_num_modules=0,
        plant_num_cracks=0,
        plant_num_cracks_per_cell=0,
        plant_num_cracked_cells=0,
        plant_num_cracks_per_cracked_cell=0,
        plant_num_cracks_per_module=0,
        plant_num_cracks_per_cell_index=[0]*(cell_counts[0]*cell_counts[1]),
        modules={},
        cell_num_cracks=[],
        cell_crack_area=[]
        )
    return statistics
    
    
def log_statistics(statistics, probability_thresh, outputs, image_list, np_images, cell_spec='_C', cell_counts=[12,6]):
    
    #Iterate through outputs
    modules_seen = []
    images_seen = []
    for o in range(len(outputs)):
        
        #Pull output and image_filename
        output = outputs[o]
        image_filename = image_list[o]    
                        
        #Pull values
        num_detections = output['num_detections']
        detection_classes = output['detection_classes']
        detection_boxes = output['detection_boxes']
        detection_scores = output['detection_scores']
        
        #Dervice valid values
        valid_detections = detection_scores >= probability_thresh
        valid_detections = np.where(valid_detections)[0]
        
        #Derive statistics
        num_cracks = len(valid_detections)
        crack_areas = []
        for detection_box in detection_boxes[valid_detections]:
            
            #Pull section of image 
            width = detection_box[2] - detection_box[0]
            height = detection_box[3] - detection_box[1]
            crack_areas.append(width*height)
        
        #Classify detected cracks
        #if any(valid_detections):
        #    crack_labels = classify_cracks(np_images[o], detection_boxes[valid_detections])
        
        #Update explicit statistics
        statistics['plant_num_cracks'] += num_cracks
        
        #Update module level statistics - MAY ERROR ON MODULE IMAGES
        image_name = os.path.splitext(image_filename)[0]
        module_name = image_name.split(cell_spec)[0]
        if not module_name in modules_seen:
            modules_seen.append(module_name)
            statistics['plant_num_modules'] = statistics['plant_num_modules'] + 1
            statistics['modules'][module_name] = dict(
                num_cells=0,
                num_cracks=0,
                num_cracks_per_cell_index=[0]*(cell_counts[0]*cell_counts[1])
            )
        
        #Update cell level statistics
        if cell_spec in image_filename:
            
            #Define cell index
            cell_index = int(image_name.split(cell_spec)[-1]) - 1
            
            #Cell tracking
            statistics['plant_num_cells'] += 1
            
            #Crack tracking
            if any(valid_detections):
                
                #Plant Statistics
                statistics['plant_num_cracks_per_cell_index'][cell_index] += num_cracks 
                statistics['plant_num_cracked_cells'] += 1
                statistics['plant_num_cracks_per_module'] = statistics['plant_num_cracks']/statistics['plant_num_modules']
                statistics['plant_num_cracks_per_cracked_cell'] = statistics['plant_num_cracks']/statistics['plant_num_cracked_cells']
                statistics['plant_num_cracks_per_cell'] = statistics['plant_num_cracks']/statistics['plant_num_cells']
                statistics
                
                #Module Statistics
                statistics['modules'][module_name]['num_cells']+=1
                statistics['modules'][module_name]['num_cracks']+=len(valid_detections)
                statistics['modules'][module_name]['num_cracks_per_cell_index'][cell_index] = len(valid_detections)
                

            #Cell Statistics
            statistics['cell_num_cracks'].append(num_cracks)
            statistics['cell_crack_area'].append(crack_areas)
            
            
        
        #Update only module level statistics
        else:
            x=1
            
            
        #Update implicit statistics
    
    #Return
    return statistics
        
def save_statistics_to_csv(statistics, mode='plant'):
    x=y
    
def create_statistics_table(statistics):
    x=y
    #utilize pandas data frames
    
def classify_cracks(np_image, crack_boxes):
    
    #Pull image size
    rows = np_image.shape[0]
    cols = np_image.shape[1]
    
    #Predefine crack labels
    crack_labels=[]
    
    #Iterate through crack boxes and classify each subimage
    for crack_box in crack_boxes:
        
        print(crack_box)
        
        #Calculate box area
        area = (crack_box[2]-crack_box[0]) * (crack_box[3] - crack_box[1])
        
        #Convert bounding box to indecies
        xmin = int(np.round(crack_box[0] * rows))
        xmax = int(np.round(crack_box[2] * rows))
        ymin = int(np.round(crack_box[1] * cols))
        ymax = int(np.round(crack_box[3] * cols))
        
        #Pull subimage
        subimage = np_image[xmin:xmax, ymin:ymax]
        subimage_hist, bins = np.histogram(subimage.ravel(), 256, [0, 256])
        plt.figure(figsize=[5, 5])
        plt.imshow(subimage)
        plt.figure(figsize=[5, 5])
        plt.hist(subimage.ravel(), 256, [0, 256]); plt.show()
        
        #Threshholding to determine fraction that is 
        blurred = cv.GaussianBlur(subimage, (3,3), 0)
        nope, fluor = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        plt.figure(figsize=[5, 5])
        plt.imshow(fluor)
                                           
        x=y
        
    
    return crack_labels
    
def get_image_size_from_pipeline(pipeline_config):
    x=y
    
