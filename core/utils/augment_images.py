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

from PIL import Image
import numpy as np
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET

import os
import tensorflow as tf
from object_detection.utils import dataset_util

from matplotlib import pyplot as plt




def augment_images(images_dir, annotations_dir, image_ext, rotations, mirrored, ignore_difficult):
    
    #Create Output Directory
    images_out_dir = images_dir + '/augmented'
    if not os.path.isdir(images_out_dir):
        os.mkdir(images_out_dir)
  
    an = 0;
    for annotation_filename in os.listdir(annotations_dir):
        
        if annotation_filename.endswith('.xml'):
                       
            #Pull Example Name
            name = os.path.splitext(annotation_filename)[0]
            
            #Pull Corresponding Image
            image_filename = name + image_ext
            image = Image.open(os.path.join(images_dir, image_filename))

            #Counting augmentations
            try:
                aug = 0
            
            
                #INFINITE LOOP PROBLEM
                while aug < len(rotations):


                    #Redefining image
                    modified_image = image

                    #Mirroring image
                    if mirrored[aug]:
                        modified_image = modified_image.transpose(Image.FLIP_TOP_BOTTOM)

                    #Rotating image
                    rotated_image = modified_image.rotate(rotations[aug], expand=True)
                    modified_image = rotated_image

                    #Pull/Update Original Annotation
                    annotation_path = os.path.join(annotations_dir, annotation_filename)
                    annotation_tree = ET.parse(annotation_path)
                    annotation_root = annotation_tree.getroot()

                    if not annotation_root.find('object') == None:      

                        #Calculating Image Size  
                        width, height = modified_image.size

                        #Iterating through objects (i.e. labels) in image
                        objct = 0
                        for obj in annotation_root.iter('object'):    

                            #Handling Difficult Instances - Difficult images removed from augmented set
                            difficult = bool(obj.find('difficult').text)
                            if ignore_difficult and difficult:
                                continue

                            #Pulling Object Location
                            bndbox = obj.find('bndbox')
                            xmin = int(bndbox.find('xmin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymin = int(bndbox.find('ymin').text)
                            ymax = int(bndbox.find('ymax').text)

                            #Converting bounding box to mask
                            mask = bounding_box_to_mask(image, xmin, xmax, ymin, ymax)
                            
                            #Mirroring
                            if mirrored[aug]:
                                transposed_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)                            
                                mask = transposed_mask

                            #Rotating
                            rotated_mask = mask.rotate(-rotations[aug], expand=True)
                            mask = rotated_mask

                            #Getting new coordinates 
                            mask = np.asarray(mask)
                            xmin, xmax, ymin, ymax = mask_to_bounding_box(mask)


                            #Updating Annotation
                            bndbox.find('xmin').text = str(xmin)
                            bndbox.find('xmax').text = str(xmax)
                            bndbox.find('ymin').text = str(ymin)
                            bndbox.find('ymax').text = str(ymax)

                            #obj['bndbox']['xmin'] = xmin
                            #obj['bndbox']['xmax'] = xmax
                            #obj['bndbox']['ymin'] = ymin
                            #obj['bndbox']['ymax'] = ymax

                    #Define Image and Annotation Names
                    image_name_modified = name + '-A' + str(aug) + '.PNG'
                    annotation_name_modified = name + '-A' + str(aug) + '.xml'

                    #Update annotation filename
                    annotation_root.find('filename').text = image_name_modified

                    #Save Image and Annotation               
                    modified_image.save(os.path.join(images_out_dir, image_name_modified))
                    annotation_tree.write(os.path.join(images_out_dir, annotation_name_modified))


                    #Increment augmentation count
                    aug+=1
            except IndexError:
                print('Indexing Error' + image_filename)
                break
                
    
def bounding_box_to_mask(image, xmin, xmax, ymin, ymax):
    
    width, height = image.size
    
    #Initialize mask
    mask = np.zeros([width, height])
    
    #Defining Subscripts
    xs = np.arange(xmin,xmax)
    ys = np.arange(ymin,ymax)
    
    #Avoiding indexing errors
    xs = xs[xs < width-2]
    ys = ys[ys < height-2]
    
    #Creating broadcast arrays       
    xs = np.transpose(np.resize(xs, (len(ys), len(xs))))
    ys = np.resize(ys, (len(xs), len(ys))) + 1


    #Add bounding box 
    mask[xs, ys] = 255
    mask = Image.fromarray(mask)
    
    #Output mask
    return mask
    

def mask_to_bounding_box(mask):
    
    #Pull locations of nonzero values
    xs, ys = np.nonzero(mask)
    
    #Identify extrema
    xmin = np.amin(xs)
    xmax = np.amax(xs)
    ymin = np.amin(ys)
    ymax = np.amax(ys)
    
    #Return extrema
    return xmin, xmax, ymin, ymax
