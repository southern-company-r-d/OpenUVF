#   Copyright 2019 Southern Company. 
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from xm.dom.minidom import parseString
from PIL import Image
import xml.etree.ElementTree as ET
import os
import csv


def refactor_set(images_dir, annotations_dir, replace_what, replace_with, mode)

    #Check that redact_what and redact_with are same length
    if not len(redact_what) == len(redact_with):
        raise Exception('redact_what and redact_with must be string tuples of the same length') 
        
    #Creating output 'map' csv that shows the label    
        
        
        

    
    
    #Define output directory names
    if images_dir == annotations_dir:
        redacted_images_dir = images_dir + '/redacted'
        redacted_annotations_dir = redacted_images_dir
    else if images_and_annotations
        redacted_images_dir = images_dir + '/redacted'
        redacted_annotations_dir = annotations_dir + '/redacted'
    else
        redacted_images_dir = images_dir + '/redacted'
    
    #Create output directory
    if not os.path.isdir(redacted_images_dir):
        os.mkdir(redacted_images_dir)
    if not os.path.isdir(redacted_annotations_dir) & not images_dir == annotations_dir & images_and_annotations:
        os.mkdir(redacted_annotations_dir)
        
        
    #Iterate through images and redact name
    im = 0
    for image_filename in os.listdir(images_dir):
        
        #Pulling image name and extension
        name = os.path.splitext(image_filename)[0]
        image_ext = os.path.splitext(image_filename)[1]
        
        #Redact name
        redacted_name = name
        for r = range(len(redact_what)):
            redacted_name = redacted_name.replace(redact_what[r], redact_with[r])        
        
        #Defining redacted filename
        redacted_image_filename = redacted_name + image_ext
        
        #Load Image
        image = Image.open(image_filename);
        
        #Save Image with redacted name
        image.save(os.path.join(redacted_images_dir, redacted_image_filename))        
        
        #Modifying corresponding annotation
        if images_and_annotations:
            
            #Define annotation filename
            annotation_filename = name + '.xml'
            
            #Catching exceptions
            try:
            
                #Load annotation xml
                annotation_path = os.path.join(annotations_dir, annotation_filename)
                annotation_tree = ET.parse(annotation_path)
                annotation_root = annotation_tree.getroot()
                
                #Changing corresponding image filename
                
            
            
        im+=1
            
        
        
        
        