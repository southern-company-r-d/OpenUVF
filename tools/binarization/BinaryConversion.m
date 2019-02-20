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

function bwmodule = BinaryConversion(I, imgType, modOrCell, plots, numCells, numCellsWide, ib)
%% BinaryConversion.m 
% Joey Richardson - Southern Company Services 2018
% This function uses various processing techniques to convert an input
% image to a binary image to decrease information density for further
% processing. The function was designed to work with the CrackIdentification
% script in order to prepare images for both panel and cell segmentation

% Inputs:
% I           - (image) Input image for binary conversion
% imgType     - (integer) Determines subset of images being processed.
%                Allows function to be used for multiple different types of
%                input images (i.e. UVF images with different properties
%                like fluorescence color)
% modOrCell   - (char or string) Determines whether processing is being
%                applied to prepare for panel segmentation or cell
%                segmentation as they require different levels of precision
% plots       - (optional boolean default false) Determines whether plots 
%                will be shown or not
%
% Required Functions:
% Image Processing Toolbox - Mathworks

% Variable Input Handling
if nargin < 4
    plots = false;
    numCells = 72;
    numCellsWide = 6;
elseif nargin < 5
    numCells = 72;
    numCellsWide = 6;
elseif nargin < 6
    numCellsWide = 6;
end


% Initializew binary image
bwmodule = false(size(I));
bwmodule = bwmodule(:,:,1);

% Defining Vertical Cell Count
numCellsHigh = numCells/numCellsWide;

%Image Preprocessing - Uniform between types




% Switch statement for processing different image cases.
%    Add additional cases for different image types (EL, different looking UVF images, etc)
switch imgType
    case 1 %Site 1 - Pale Green Fluorescence
        if strcmpi(modOrCell, 'mod') || strcmpi(modOrCell, 'module') || strcmpi(modOrCell, 'm')
            
            %Store input image for reference
            Iin = I;
            
%             %Normalize I
%             I = double(I);
%             I = uint8(I .* (255./max(I(:))));
%             
%             %Headlamp Handling - PROBLEM: Removes faint fluorescence and
%             %   hotspots
%             r = I(:,:,1);
%             g = I(:,:,2);
%             b = I(:,:,3);
%             toored = r> (1.25 .* g) | r > (1.25 .* b); 
%             toored = repmat(toored, 1,1,3); %cat(3, toored, false(size(I,1), size(I,2)));
%             I(toored) = .25 .* I(toored); %mute reds
%             
%             % Initial Blur
%             I = imguidedfilter(I);
%             
%             %Convert to HSV color space
%             hsv = rgb2hsv(I);
%             
%             %Apply adaptive histogram adjustment to value layer
%             %hsv(:,:,3) = adapthisteq(hsv(:,:,3));
%             
%             %Define hue based mask
%             huemask = hsv(:,:,1) > (1/6) & hsv(:,:,1) < (5/6); %exclude reds
%             satmask = hsv(:,:,2) < .5; %for site 1 images, the fluorescence is grayish
%             valmask = hsv(:,:,3) > mean(hsv(:,:,3)) & hsv(:,:,3) < .98;
%             hsvmask = huemask & valmask;
%             hsvmask = imdilate(hsvmask, strel('disk', 15));
%             hsvmask = repmat(hsvmask, 1, 1, 3);            
%             hsv(~hsvmask) = .25 .* hsv(~hsvmask);
%             
%             %Convert back to rgb
%             I = uint8(256 .* hsv2rgb(hsv));
            
            %Uneven brightness correction
            comp = imcomplement(I);
            haze = imreducehaze(comp);
            haze = imcomplement(haze);
            
            %Level filtering using Otsu's method
            hazeT = graythresh(rgb2gray(haze));
            hazemask = imbinarize(rgb2gray(haze), hazeT);
            if plots, figure('Name', 'Hazemask'); imshow(hazemask, 'Border', 'tight'); title('Binarized haze'); end      
            
            %Consider where the blue or green channels are brighter than
            %red
            propercolor = haze(:,:,3) > haze(:,:,1) | haze(:,:,2) > haze(:,:,1);
            
            
            % Get the Green Channel from RGB colorspace
            g = haze(:,:,2);
            if plots, figure('Name', 'Green Channel'); imshow(g, 'Border', 'tight'); title('Green Channel of RGB Space'); end      

            % Get the Saturation Channel from HSV colorspace
            hsv = rgb2hsv(haze);
            sat = uint8(hsv(:,:,2)*255);
            if plots, figure; imshow(sat, 'Border', 'tight'); title('Saturation Channel of HSV Space'); end      

            % Subtract the Saturation Channel from the Green Channel
            sub = g - sat;
            if plots, figure; imshow(sub, 'Border', 'tight'); title('Green - Saturation'); end      

            % Apply Median Filter to eliminate salt and pepper noise
            filtered = medfilt2(sub, [8,8]);
            if plots, figure; imshow(filtered, 'Border', 'tight'); title('Filtered Image'); end      

            % Apply Adaptive Threshold to binarize image
            % *Can fail if image contains large dark regions
            % *Could be changed from adaptive to discrete threshold if
            % flash correction is used 
            T = adaptthresh(filtered, 0.2, 'Statistic', 'gaussian');
            bwmodule = imbinarize(filtered, T);
            
            % Headlamp Handling
            [rows, cols] = size(bwmodule);
            bwmodule = bwareafilt(bwmodule, [2000, .1 .* (rows*cols)]);
            
            %Filters
%             props = regionprops(bwmodule, 'all');
%             labels = bwlabel(bwmodule);
%             eulers = [props(:).EulerNumber];
%             solidities = [props(:).Solidity];           
%             holefilt = eulers > (mean(eulers)- (5*std(eulers)));
%             solidityfilt = solidities > .6;
%             cells = find(holefilt & solidityfilt);
%             bwmodule = ismember(labels, cells);
            if plots, figure; imshow(bwmodule, 'Border', 'tight'); title('Binary Image'); end      

        elseif strcmpi(modOrCell, 'cell') || strcmpi(modOrCell, 'c')
            
            %FUNCTIONS TO CONSIDER: Imguidedfilt (edge preservation with
            %gaussian imnlmfilt (non local means filtering
            
            %Noise removal
%             hsv = rgb2hsv(I);
%             hsv(:,:,1) = imgaussfilt(medfilt2(hsv(:,:,1), [3 3]), 1); %color noise
%             hsv(:,:,3) = medfilt2(wiener2(hsv(:,:,3), [5 5]), [3 3]); %luminance noise
%             I = hsv2rgb(hsv);
            
            %Define the bounding boxes of each cell
            widthBounds = round(linspace(1, size(I,2), numCellsWide + 1));
            heightBounds = round(linspace(1, size(I,1), numCellsHigh + 1));
            
            %Prepare figure for debug outputs
            if plots figure('units', 'normalized', 'outerposition', [0 0 1 1]); end
            
            %Drawing Module Image
            if plots subplot(2,4,[1 5]); imshow(I); axis equal; end
            
            %Iterate through cells
            for h = 1:numCellsHigh
                
                %Define vertical indecies
                hi = heightBounds(h);
                he = heightBounds(h+1);
                
                for w = 1:numCellsWide
                    
                    %Defining horizontal indecies
                    wi = widthBounds(w);
                    we = widthBounds(w+1);
                    
                    %Pulling cell image
                    cell = double(I(hi:he, wi:we, 2)) + double(I(hi:he, wi:we, 3));% - double(I(hi:he, wi:we, 1));
                    if plots subplot(2,4,2); imshow((cell)./max(cell(:))); title('Cell Image'); end
                    
                    %Normalizing cell
                    cell = cell./max(cell(:));
                    
                    %Removing Noise
                    cell = medfilt2(cell, [2, 2]);
                    
                    %Adjusting contrast
                    cell = imadjust(cell, [.15 .85],[0 1]);
                    
                    %Smoothing image
                    cell = imgaussfilt(cell, 1);
                                        
                    %Masking the cell area to remove edge noise
                    [cr, cc] = size(cell);
                    disksize = 3; %round(mean([cr, cc])/25);
%                     opened = imopen(cell,strel('disk',disksize));
%                     openedThresh = adaptthresh(opened, .8, 'NeighborhoodSize', (2*floor(mean([cr, cc])./128) + 1));
%                     opened = imbinarize(openedThresh);
%                     opened = imfill(opened, 'holes');
%                     opened = imdilate(opened, strel('disk',disksize));
                    mask = imclose(cell, strel('disk', disksize));
                    maskT = adaptthresh(mask, .8, 'NeighborhoodSize', (2*floor(mean([cr, cc])./2) + 1));

                    mask = imbinarize(mask, maskT);
                    mask = bwareafilt(mask, [round(.05 .* (cr .* cc)), (cr .*cc)]);
                    holes = ~mask;
                    holes = bwareafilt(holes, [.01 .* (cr*cc), .1 .* (cr*cc)]); 
                    
                    mask = imfill(mask, 'holes');
                    %mask(holes) = false;
                    cell(~mask) = 0; %preserving large holes that denote cracks within the cell
                    if plots subplot(2,4,3); imshow(cell); title('Masked Cell'); end
                    
                    
                    
                    
%                     blur = imgaussfilt(cell, 10);
%                     maskT = graythresh(blur) - .05
%                     mask = blur > maskT;
%                     cell(~mask) = 0;
%                     if plots subplot(2,4,3); imshow(uint8(256.*cell)); title('Cell Image - Contrast Adjusted'); end
                    
                    %Blurring to remove noise
                    %cell = imbilatfilt(cell, 50, 1);
                    %cell = imguidedfilter(cell, 'DegreeOfSmoothing', 100);
                    cell = imbilatfilt(cell, 200, 2);
                    if plots subplot(2,4,6); imshow(cell); axis equal; end   
                    
                    % Convert to binary
                    %cellT = graythresh(cell(mask))
                    cellmean = mean(cell(mask));
                    cellstd = std(cell(mask));
                    cellT = .25; %abs(cellmean - 1.*cellstd);
                    bwcell = cell > cellT;
                    if plots subplot(2,4,7); imshow(bwcell); end
                    holes = ~bwcell;
                    holes = bwareafilt(holes, [.01 .* (cr*cc), .1 .* (cr*cc)]);
                    holes = imdilate(holes, strel('disk',4));
                    bwcell = imfill(bwcell, 'holes');
                    bwcell(holes) = false; %preserving large holes and removing noisy holes.
                    if plots subplot(2,4,7); imshow(bwcell); end
                    
                    %Dilate Cracks/Holes to preserve actual edges
                    
                    
                    %Place in module
                    bwmodule(hi:he, wi:we) = bwcell;
                    if plots subplot(2,4, [4 8]); imshow(bwmodule); end
                    
                    %Pause for debug review
                    if plots pause(.05); end
                    
                end
                
            end
            
            
            
%             % Get Green Channel from cropped image
%             g = I(:,:,2);
%             
%             % Get size of image
%             [r, c] = size(g);
%             
%             % Create mask with a box running through the middle of the image
%             x = [(c/2)-10 (c/2)-10 (c/2)+10 (c/2)+10 (c/2)-10];
%             y = [r 0 0 r r];
%             mask = imcomplement(poly2mask(x,y,r,c));
%             
%             % Remove tape illumination
%             masked = bsxfun(@times, g, cast(mask, class(g)));
%             background = imopen(masked,strel('disk',4));
%             sub = imbinarize(masked + background, 0.6);
%             masked = bsxfun(@times, g, cast(sub, class(g)));
%             
%             % Display the Background Approximation as a Surface
%             if plots, figure; surf(double(background(1:8:end,1:8:end))),zlim([0 255]); title('Background Estimation'); ax = gca; ax.YDir = 'reverse'; end
%                       
%             % Convert to binary
%             T = adaptthresh(masked, 0.6);
%             bw = imbinarize(masked, T);

            %Place binarized cell in output binary image
            bwmodule;
            
            % Perform morphological operations to clean image and improve feature clarity
            bwmodule = bwmorph(bwmodule, 'close', Inf);
            bwmodule = bwmorph(bwmodule, 'clean');
            bwmodule = bwmorph(bwmodule, 'majority');
            if plots, subplot(2,4, [4 8]); imshow(bwmodule); title('Binary Cell Image'); end  
            
            
            
        else
            disp('BinaryConversion: Improper segmentation type (modOrCell)')
        end
        
    case 2 %Site 2 - Bright Blue Fluorescence
        
    case 3 %Site 3 - Bright Blue Ring Fluorescence
        
            
            %Image Scaling
            rawI = I;
            scalarmax = double(sort(I(:)));
            scalarmax = scalarmax(round(.90 .*length(scalarmax))); %90th percentile of image brightness
            scalar = (60000)/scalarmax;
            I = I .* scalar;
            

            
            %Image contrast adjustment
            %I = imadjustn(I, [.1, .85], [.01 .95]);   
            
            %Brightness Adjustment
            comp = imcomplement(I);
            dehazed = imreducehaze(comp);
            I = imcomplement(dehazed);
            
            %Image smoothing
            %I = medfilt3(I, [3 3 3]);
%             for l = 1:3
%                 %I(:,:,1) = wiener2(I(:,:,l), [5 5]);
%                 I(:,:,l) = medfilt2(I(:,:,l), [5 5]);              
%             end
            I = imgaussfilt(I, 2);
            %I(:,:,1) = I(:,:,1).* .8;
            
            %Create and Apply Circular Crop Mask
            [rows, cols, ~] = size(I);
            center = [rows ./ 2, cols ./2];
            [xinds, yinds] = meshgrid(1:rows, 1:cols);
            xdists = xinds - center(1);
            ydists = yinds - center(2);
            dists = sqrt(xdists.^2 + ydists.^2);
            mask = repmat((dists <= (min(center)+700))', 1, 1, 3);
            I(~mask) = 0;
            
            %Pulling Color Channels
            r = I(:,:,1);
            g = I(:,:,2);
            b = I(:,:,3);
            
            %First Filter - Greater greens and blues than reds
            possible = r <g & r < b;
            
            %Convert to HSV
            hsv = rgb2hsv(I);
            h = medfilt2(hsv(:,:,1), [10, 10]); %remove color noise
            s = imadjust(hsv(:,:,2), [.1 .9], [0 1]); %enhance saturation
            v = medfilt2(hsv(:,:,3)); %remove intensity noise
            hsv = cat(3, h, s, v);
            rgb = hsv2rgb(hsv);
            
            
            %Identify areas belonging to certain color groups
            h = imgaussfilt(h .* 360, 3);
            blues = h > 150 & h < 330;
            greens = h > 30 & h < 210;
            reds = h < 90 | h > 270;
            
            %Texture Analysis
            
            
            %Second Filter - 
            %modules = medfilt2(imdilate(blues, strel('disk', 50)) & ~reds, [5 5]); %remove small noise
            
            modules = bwareafilt(blues & ~reds, [1000, 1000000000]);
            %modules = bwareafilt(modules, [1000, 100000000]);
            %modules = imdilate(modules, strel('disk', 10));
            modules = imfill(modules, 'holes');
            modules = imclose(modules, strel('square', 20));
            
            modules = imclose(modules, strel('disk', 20));
            modules = imfill(modules, 'holes');
%             props = regionprops(modules, 'Area', 'Perimeter', 'Solidity');
%             labels = bwlabel(modules);
%             areas = [props(:).Area];
%             perimeters = [props(:).Perimeter];
%             solidities = [props(:).Solidity];
%             ratios = areas./(perimeters.^2);
%             bads = solidities < .6

            %Return only largest module
            modulelabels = bwlabel(modules);
            props = regionprops(modules, 'Area');
            [~, largest] = max([props(:).Area]);
            modules = ismember(modulelabels, largest);
            bwmodule = modules;
            
            if plots; figure; imshow(modules); end
            x = 1;
            
            %POSSIBLE
            %imfill(imdilate(blues & ~reds, strel('disk', 10)), 'holes')
           
        
        
    otherwise
        disp('BinaryConversion: Improper image type (imgType).')
end

end
