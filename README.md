# OpenUVF
Copyright © 2019 Southern Company Services, Inc. All rights reserved. OpenUVF is [licensed](https://github.com/southern-company-r-d/OpenUVF/blob/master/LICENSE) under the Apache License, Version 2.0.

OpenUVF uses portions of [TensorFlow](http://www.tensorflow.org) ([TensorFlow on GitHub](https://github.com/tensorflow/tensorflow)), also licensed under [Apache 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE), copyright 2018 The TensorFlow Authors.

## Overview

OpenUVF is a software utility developed by Southern Company R&D, a subdivision of Southern Company Services, Inc., to automatically analyze Ultraviolet Fluorescence (UVF) images of solar PV modules. It is imagined as a foundation, upon which UVF can be developed as a practical inspection technique complementing existing PV inspection techniques like Electroluminescence and Thermography.

OpenUVF is intended for research and development purposes. It is a work in progress, and may not work for your application or on your system. We hope you find it useful, though!

This software requires the following items:
* MATLAB (and the Image Processing Toolbox)
* [LabelImg](https://github.com/tzutalin/labelImg) or a similar annotation tool with PASCAL VOC format
* Python 3, Anaconda, and several packages described in our documentation [here](https://github.com/southern-company-r-d/OpenUVF/tree/master/docs).
* [TensorFLow](https://github.com/tensorflow/tensorflow), and for training new models, NVIDIA GPU hardware and described in our documentation [here](https://github.com/southern-company-r-d/OpenUVF/tree/master/docs).
* And additional items described in our documentation [here](https://github.com/southern-company-r-d/OpenUVF/tree/master/docs).

A portion of TensorFlow, specifically the [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), is redistributed with OpenUVF. It is included in our code [here](https://github.com/southern-company-r-d/OpenUVF/tree/master/core/object_detection).

## Details
This section is a work in progress. For now, check out what we have [here](https://github.com/southern-company-r-d/OpenUVF/tree/master/docs).

Ultraviolet Fluorescence (UVF) is an emerging field inspection technique which enables the detection of cracks and other faults in some types of solar PV modules. Chemical changes in the encapsulating layer between the cells and the front glass can lead to fluorescence when it is exposed to UV light. Cracks in solar cells are often indicated by dark lines in the UVF images. OpenUVF is designed to automatically analyze UVF images and detect and locate cracks. 

The first step in using OpenUVF is to train a model to detect cracks. Cracks can show up differently in different PV modules depending on their age, material content, and other factors. Several examples of UVF images of individual cells are shown below. We have trained and tested OpenUVF on "type A" cells/modules. Types C, E, and F, are examples of modules that only show a ring of fluorescence, but we expext that models could be trained to detect the gaps in the ring that indicate cell cracks. 

![UVF examples](https://github.com/southern-company-r-d/OpenUVF/blob/master/docs/images/UVF_cell_examples_sm.jpg)

MATLAB is used to pre-process images to make modules rectangular and to segment out images of individual cells (based on known module and cell relative geometries). A training dataset then has to be manually annotated using LabelImg or a similar tool, where a person identifies bounding boxes of cracks in cells. A TensorFlow F-RCSS model is then trained on that dataset.

Once a model has been trained, the processing workflow steps include: 1) acquiring UVF images of solar panels; 2) binarization, closing, and module segmentation; 3) perspective correction and cell segmentation; 4) TensorFlow F-RCNN crack detection; 5) detected cracks; and 6) module re-stitching.

![Workflow diagram](https://github.com/southern-company-r-d/OpenUVF/blob/master/docs/images/processing_workflow2.jpg)

## Disclaimer
As set forth in the [license](https://github.com/southern-company-r-d/OpenUVF/blob/master/LICENSE), Southern Company Services, Inc., provides OpenUVF on an “as is” basis, without warranties or conditions of any kind, either express or implied, including, without limitation, any warranties or conditions of title, non-infringement, merchantability, or fitness for a particular purpose. You (the individual or legal entity exercising permissions granted by this license) are solely responsible for determining the appropriateness of using or redistributing OpenUVF and assume any risks associated with your exercise of permissions under this license.
