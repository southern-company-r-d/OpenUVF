% Copyright © 2019 Southern Company Services, Inc.  All rights reserved.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function segment_images(inputs)

%% Function Setup

    %Function Cleaning
    clear; clc; close all;
    tic
    
    %File Directory Inputs
    inputDirectory = 'inputs/';%'S:/Workgroups/SCS Research and Environmental Affairs/RTM/BHGilleland/Projects/UV Fluorescence/Images/APEX - Site Revisit with 1032 Module Speedrun - (10-15-2018)/Night UVF/Single Row Speed Run (80 strings)/';%inputs/';
    inputFileType = '.JPG';
    outputDirectoryCellBinary = 'outputs/Cells - Binary/';
    outputDirectoryCellColor = 'outputs/Cells - Color/';
    outputDirectoryModBinary = 'outputs/Modules - Binary/';
    outputDirectoryModColor = 'outputs/Modules - Color/';
    outputDirectoryModData = 'outputs/Modules - Data/';
    
    %Location Inputs
    site = 'A'; %name of location where the pictures were taken - for saving files
    
    %Module Inputs
    nCellsWide = 6; %Provided as seen in an unmodified image
    nCellsTall = 12;
    nModsPerImApprox = 1;
    
    %Module Inputs
    nCells = 72;
    nCellsWide = 6;
    imageBuffer = 4;
    aspectRatio = [2 1];
    moduleType = 1; %fluorescence profile
    
%% Initialization

    % Determine where your m-file's folder is.
    folder = fileparts(which(mfilename)); 

    % Add that folder plus all subfolders to the path.
    addpath(genpath(folder));

    % Boolean variable to determine if plots should be shown or not
    plots = true;

    % Error handling variables 
    errorCount = 0;
    errorFiles = {};

    % Directory of input image files (jpg)
    files = dir([inputDirectory, '*', inputFileType]);
    
    %Performance tracking
    images.total = length(files);
    images.processed = 0; %images completely processed
    images.failed = {}; %images that fail to process
    images.avgmodules = 0; %number of modules per average image
    
    %Initializing time tracking and estimation
    time.val = tic;
    time.imtimes = zeros(images.total, 1); %log time to process each image

    % Process Images
    for z = 1:length(files)

        % Clear workspace of non-essential variables to free up memory    
        clearvars -except plots errorCount errorFiles files z moduleType ...
            numCells numCellsWide imageBuffer aspectRatio nCells...
            nCellsWide nCellsTall ...
            site outputDirectoryCellBinary outputDirectoryCellColor ...
            outputDirectoryModBinary outputDirectoryModColor outputDirectoryModData
        
        close all
        try    

        % Display current image counter
        fprintf('Image %d\n', z)

        %Encapsulating function in a try catch statement for continuity
        %try
        % Get image file absolute path from file components 
        imageFile = fullfile(files(z).folder, files(z).name);

        % Read image from file
        I = imread(imageFile);

        % Convert to grayscale
        imageGray = rgb2gray(I);

        % Get size of image
        [rows,cols] = size(imageGray);

        %% 1 - Binarization
        % Convert to binary image to decrease information density and provide input
        % for panel segmentation.

        imgType = moduleType;
        modOrCell = 'm';
        bw = BinaryConversion(I, imgType, modOrCell, plots);

    %% 2 - Panel Segmentation
        % Find each full panel in the image and get an accurate estimate of
        % each panel's corners for perspective correction.
        
        % Contextual Image Closing
        [closed, I] = close_image(bw, I, plots);        

        
        % Find corners of each full panel
        panels = FindCorners(closed, plots);

        % Redraw label matrix from panel corners
        labelMatrix = zeros(rows, cols);
        for j = 1:length(panels)
            c = [panels(j).Corners.bottomLeft(:,1), panels(j).Corners.bottomRight(:,1), panels(j).Corners.topRight(:,1), panels(j).Corners.topLeft(:,1)];
            r = [panels(j).Corners.bottomLeft(:,2), panels(j).Corners.bottomRight(:,2), panels(j).Corners.topRight(:,2), panels(j).Corners.topLeft(:,2)];
            roi = roipoly(I, c, r);
            labelMatrix = labelMatrix + roi*j;
        end
        if plots, figure; imshow(label2rgb(uint16(labelMatrix), @jet, [98 98 98]./255), 'Border', 'tight'); title('Label Matrix - Adjusted'); end

        % Flash Correction
        flashFolder = 'fuji-x100f wide angle';
        %flashCorrected = FlashCorrection(I, flashFolder, plots);
        flashCorrected = I;
        
    %% 3 - Perspective Correction
        % Find and apply projective transformation to correct panel aspect ratio 
        % to ensure uniform cell properties for cell segmentation and
        % classification.

        % Iterating through each module and calculating its own correction
        for m = 1:length(panels)
            
            % Crop Image to reduce memory use
            corners = [panels(m).Corners.topLeft; panels(m).Corners.bottomLeft;
                       panels(m).Corners.topRight; panels(m).Corners.bottomRight];
            buffer = 10;
            Xmin = round(min(corners(:,1))) - buffer;
            Xmax = round(max(corners(:,1))) + buffer;
            Ymin = round(min(corners(:,2))) - buffer;
            Ymax = round(max(corners(:,2))) + buffer;
            rect = [Xmin Ymin (Xmax-Xmin) (Ymax - Ymin)];
            panels(m).croppedim = imcrop(flashCorrected, rect);
            panels(m).croppedlabels = imcrop(labelMatrix, rect);
            panels(m).Corners.topLeft = [corners(1,1) - Xmin, corners(1,2) - Ymin];
            panels(m).Corners.bottomLeft = [corners(2,1) - Xmin, corners(2,2) - Ymin];
            panels(m).Corners.topRight = [corners(3,1) - Xmin, corners(3,2) - Ymin];
            panels(m).Corners.bottomRight = [corners(4,1) - Xmin, corners(4,2) - Ymin];
            
            % Calculate projective transformation matrix for middle panel
            movingPoints = [panels(m).Corners.bottomLeft; panels(m).Corners.bottomRight; panels(m).Corners.topRight; panels(m).Corners.topLeft];
            fixedPoints = [0 aspectRatio(1)*1000; aspectRatio(2)*1000 aspectRatio(1)*1000; aspectRatio(2)*1000 0; 0 0];
            transformationType = 'projective';
            tform = fitgeotrans(movingPoints, fixedPoints, transformationType);
            
            % Apply transformation to original image and fixed label matrix
            [warpedImage, warpedImageRef] = imwarp(panels(m).croppedim, tform, 'linear');
            [warpedLabelMatrix, warpedLabelMatrixRef] = imwarp(panels(m).croppedlabels, tform, 'nearest');
            if plots, figure; imshow(warpedImage, 'Border', 'tight'); title('Perspective Corrected Image'); end
            if plots, figure; imshow(label2rgb(uint16(warpedLabelMatrix), @jet, [98 98 98]./255), 'Border', 'tight'); title('Perspective Corrected Label Matrix'); end
            
            % Use fixed label matrix to mask the original image and isolate panels from background
            mask = warpedLabelMatrix > 0;
            maskedImage = bsxfun(@times, warpedImage, cast(mask, class(warpedImage)));
            if plots, figure; imshow(maskedImage, 'Border', 'tight'); title('Perspective Corrected Masked Image'); end
            
            %Defining Module Boundary
            boundaries =  regionprops(warpedLabelMatrix, 'BoundingBox');
            BoundingBox = boundaries(m).BoundingBox;
            
            %Isolating Module
            panels(m).Image = imcrop(maskedImage, BoundingBox);
            
            %Log transformation information for output
            panels(m).croprect = rect;
            panels(m).movingPoints = movingPoints;
            panels(m).fixedPoints = fixedPoints;
            panels(m).tform = tform;
            panels(m).warpedImageRef = warpedImageRef;
            panels(m).warpedLabelMatrixRef = warpedLabelMatrixRef;
            panels(m).mask = mask;
            
        end

    %% 4 - Cell Segmentation

    % Segment cells and get cell properties
        [panels] = GetCellProperties(panels, imgType, 'c', nCells, nCellsWide, imageBuffer, plots);

    %% 5- Output Preparation
        
        moduleCt = 1;
        for m = 1:length(panels)
            
            %Updating Module
            module(moduleCt).imageFile = imageFile;
            module(moduleCt).imageFileName = files(z).name(1:(strfind(files(z).name, '.')-1));
            module(moduleCt).image = panels(m).Image;
            module(moduleCt).binary = panels(m).Binary;
            module(moduleCt).indexFromLeft = m;
            module(moduleCt).indexFromRIght = length(panels) - m + 1;
            module(moduleCt).cells(1:nCells) = struct;
            module(moduleCt).panel = panels(m);
            
            %Creating cell counter 
            cellCt = 1;
            
            %Ierating through cells to create data structures and output
            %   label images
            for c = 1:nCells
%                 cellind = unique(modules(m).Cells(c).CellImage); %different indexing systems
%                 cellind = cellind(cellind > 0); %Attempt to number it as
%                 MATLAB linearly indexes array - instead transition to IEC
%                 standard per Joey
                cellImage = panels(m).Cells(c).CellImage;
                cellImage = cellImage > 0;
                cellImage = imresize(cellImage, [150, 150]);
                cellImage = uint8(256 .* double(cellImage));
                module(moduleCt).cells(c).image = cellImage;
                module(moduleCt).cells(c).color = panels(m).Cells(c).CellColorImage;
                module(moduleCt).cells(c).info = panels(m).Cells(c).Blobs;
                module(moduleCt).cells(c).num = cellCt;
                module(moduleCt).cells(c).label = panels(m).Cells(c).Label;
                module(moduleCt).cells(c).name = sprintf('%s_M%s-%d_C%d.PNG',...
                    site, module(moduleCt).imageFileName, moduleCt, cellCt);
                
                %Outputting cell image file
                cellBinaryFn = [outputDirectoryCellBinary, module(moduleCt).cells(c).name];
                cellBinaryIm = module(moduleCt).cells(c).image;
                cellColorFn = [outputDirectoryCellColor, module(moduleCt).cells(c).name];
                cellColorIm = imresize(module(moduleCt).cells(c).color, [150, 150]);
                imwrite(cellBinaryIm, cellBinaryFn);
                imwrite(cellColorIm, cellColorFn);
                
                %Iterating cell counter
                cellCt = cellCt + 1;
        
            end
            
            
            %Outputting Module Binary Image
            modBinaryFn = sprintf('%s_M%s-%d.PNG', site, module(moduleCt).imageFileName, moduleCt);
            modBinaryFn = [outputDirectoryModBinary, modBinaryFn];
            modBinaryIm = module(moduleCt).binary;
            imwrite(modBinaryIm, modBinaryFn);
            
            %Outputting Module Color Image
            modColorFn = sprintf('%s_M%s-%d.PNG', site, module(moduleCt).imageFileName, moduleCt);
            modColorFn = [outputDirectoryModColor, modColorFn];
            modColorIm = module(moduleCt).image;
            imwrite(modColorIm, modColorFn);
            
            %Iterating module counter
            moduleCt = moduleCt + 1;
            
            
        end
        
        %Outputting Module Structure
        modulefn = sprintf('%s_Ms%s.mat', site, module(1).imageFileName);
        modulefn = [outputDirectoryModData, modulefn];
        save(modulefn, 'module', '-v7.3')
        
        
        catch ME
            fprintf('\nImage %d Failed. %s\n    Continuing...\n\n', z, ME.message)            
        end
    end
    
    %Saving Output Files
    
end
