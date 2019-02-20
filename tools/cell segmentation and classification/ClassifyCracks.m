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

function warpedPanels = ClassifyCracks(warpedPanels, plots, textType)
%% ClassifyCracks.m 
% Joey Richardson - Southern Company Services 2018
% This function classifies cell damage based on cell properties

% Inputs:
% warpedPanels - (struct) Structure containing panel information from
%                 CrackIdentification program
% textType     - (optional char or string) Determines type of text to
%                 overlay on cells. Options: 'labels', 'all', 'none'
% plots        - (optional boolean default false) Determines whether plots 
%                 will be shown or not
%
% Required Functions:
% Image Processing Toolbox - Mathworks

if nargin < 3
    textType = 'none';
elseif nargin < 2
    plots = false;
end

n = length(warpedPanels);
for k = 1:n    
    % Get all blobs in panel
    blobs = [warpedPanels(k).Cells(:).Blobs];
    
    % Eliminate blobs with lots of holes (high chance they are noisy holes)
    blobs = blobs([blobs.EulerNumber] > -2);

    % Get mean area of the 3-10th largest blobs
    descendingCellArea = sortrows([blobs.Area]', 'descend');
    meanCellArea = mean(descendingCellArea(3:10));
    warpedPanels(k).MeanCellArea = meanCellArea;

    % Get mean and standard deviation of perimeter 
    descendingCellPerimeter = sortrows([blobs.Perimeter]', 'descend');
    meanCellPerimeter = mean(descendingCellPerimeter);
    stdCellPerimeter = std(descendingCellPerimeter);
    
    % Get mean and standard deviation of area to perimeter 
    descendingCellAtP = sortrows([blobs.AreaToPerimeter]', 'descend');
    meanCellAtP = mean(descendingCellAtP);
    stdCellAtP = std(descendingCellAtP);

    % Get mean and standard deviation of area to perimeter 
    descendingCellExtent = sortrows([blobs.Extent]', 'descend');
    meanCellExtent = mean(descendingCellExtent);
    stdCellExtent = std(descendingCellExtent);
    
    if plots, figure; imshow(label2rgb(warpedPanels(k).LabelMatrix, @jet, [98 98 98]./255), 'Border', 'tight'); hold on; end
    crackCounter = 0;
    
    % This section could be replaced with machine learning 
    for i = 1:length(warpedPanels(k).Cells)
        if length(warpedPanels(k).Cells(i).Blobs) > 1
            warpedPanels(k).Cells(i).Cracked = true;
            warpedPanels(k).Cells(i).CrackType = 1;
            for j = 1:length(warpedPanels(k).Cells(i).Blobs) 
                centroidWorld = warpedPanels(k).Cells(i).Rectangle(:,1:2) + warpedPanels(k).Cells(i).Blobs(j).Centroid;
                if j == 1
                    boundingBoxWorld = warpedPanels(k).Cells(i).Rectangle(:,1:2) + warpedPanels(k).Cells(i).Blobs(j).BoundingBox(:,1:2);
                end
            end
            
        else
            centroidWorld = warpedPanels(k).Cells(i).Rectangle(:,1:2) + warpedPanels(k).Cells(i).Blobs.Centroid;
            boundingBoxWorld = warpedPanels(k).Cells(i).Rectangle(:,1:2) + warpedPanels(k).Cells(i).Blobs.BoundingBox(:,1:2);

            if stdCellPerimeter > 0.1*meanCellPerimeter ...
               && warpedPanels(k).Cells(i).Blobs.Perimeter > meanCellPerimeter + 0.35*stdCellPerimeter ...
               && warpedPanels(k).Cells(i).Blobs.Extent < meanCellExtent + 0.5*stdCellExtent ...
               && warpedPanels(k).Cells(i).Blobs.AreaToPerimeter < meanCellAtP + 0.25*stdCellAtP ...
               && warpedPanels(k).Cells(i).Blobs.EulerNumber > -1
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 2;
            
            elseif stdCellPerimeter > 0.025*meanCellPerimeter ... 
                   && (warpedPanels(k).Cells(i).Blobs.Perimeter > meanCellPerimeter + stdCellPerimeter ...
                       || warpedPanels(k).Cells(i).Blobs.Perimeter < meanCellPerimeter - 2*stdCellPerimeter ...
                       || warpedPanels(k).Cells(i).Blobs.Perimeter > meanCellPerimeter + 0.1*meanCellPerimeter) ...
                   && warpedPanels(k).Cells(i).Blobs.Solidity < 0.94 ...
                   && warpedPanels(k).Cells(i).Blobs.EulerNumber > -1
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 3;
            
            elseif stdCellPerimeter > 0.025*meanCellPerimeter ...
                   && warpedPanels(k).Cells(i).Blobs.Perimeter > meanCellPerimeter ...
                   && warpedPanels(k).Cells(i).Blobs.AreaToPerimeter < meanCellAtP - 0.25*stdCellAtP ...
                   && warpedPanels(k).Cells(i).Blobs.Solidity < 0.938 ...
                   && warpedPanels(k).Cells(i).Blobs.EulerNumber > -1
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 4;
    
            elseif warpedPanels(k).Cells(i).Blobs.Perimeter > meanCellPerimeter + 3*stdCellPerimeter...
                   && warpedPanels(k).Cells(i).Blobs.AreaToPerimeter < meanCellAtP - 0.25*stdCellAtP ...
                   && warpedPanels(k).Cells(i).Blobs.EulerNumber > -1
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 5;

            elseif warpedPanels(k).Cells(i).Blobs.FilledRatio < 0.9975 ...
                   && warpedPanels(k).Cells(i).Blobs.EulerNumber > -3 
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 6;

            elseif stdCellExtent > 0.035 ...
                   && ((warpedPanels(k).Cells(i).Blobs.Extent < meanCellExtent - 0.5*stdCellExtent) ...
                        || (warpedPanels(k).Cells(i).Blobs.Extent < meanCellExtent ...
                            && warpedPanels(k).Cells(i).Blobs.AreaToPerimeter < meanCellAtP + 0.1*stdCellAtP)) ...
                   && warpedPanels(k).Cells(i).Blobs.EulerNumber > -1
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 7;
            
            elseif stdCellExtent > 0.015 ...
                   && warpedPanels(k).Cells(i).Blobs.Extent < meanCellExtent - 1.5*stdCellExtent...
                   && warpedPanels(k).Cells(i).Blobs.EulerNumber > -1
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 8;
         
            elseif (warpedPanels(k).Cells(i).Blobs.Extent < 0.875 ...
                    || warpedPanels(k).Cells(i).Blobs.AreaToPerimeter < 31) ...
                   && warpedPanels(k).Cells(i).Blobs.EulerNumber > -1
                warpedPanels(k).Cells(i).Cracked = true;
                warpedPanels(k).Cells(i).CrackType = 9;
       
            else
                 if plots && strcmpi(textType, 'all')
                     text(centroidWorld(1)-60, centroidWorld(2), sprintf('Area: %d\nPerimeter: %.2f\nSolidity: %.4f\nFilledArea: %d\nExtent: %.4f\nEulerNum: %d\nFilledRatio: %.4f\nAtP Ratio: %.4f', ...
                         warpedPanels(k).Cells(i).Blobs.Area, warpedPanels(k).Cells(i).Blobs.Perimeter, warpedPanels(k).Cells(i).Blobs.Solidity, warpedPanels(k).Cells(i).Blobs.FilledArea, warpedPanels(k).Cells(i).Blobs.Extent, warpedPanels(k).Cells(i).Blobs.EulerNumber, warpedPanels(k).Cells(i).Blobs.FilledRatio, warpedPanels(k).Cells(i).Blobs.AreaToPerimeter), ...
                         'Color', 'white', 'FontSize', 7); 
                 end
            end
        end
      
        % Initialize damage types
        warpedPanels(k).Cells(i).CrackType = 0;
        warpedPanels(k).Cells(i).LostAreaPercent = 0;
        warpedPanels(k).Cells(i).Holes = 0;
        warpedPanels(k).Cells(i).Cracks = 0;
        warpedPanels(k).Cells(i).Splits = 0;
        if warpedPanels(k).Cells(i).Cracked

            crackCounter = crackCounter + 1;
            cellBlobs = [warpedPanels(k).Cells(i).Blobs];
            cellArea = sum([cellBlobs(:).Area]);
            
            warpedPanels(k).Cells(i).LostAreaPercent = (1 - cellArea/meanCellArea)*100;
            
           if plots && strcmpi(textType, 'labels'), text(boundingBoxWorld(1), boundingBoxWorld(2)-20, sprintf('%s-%.2f', warpedPanels(k).Cells(i).Label, warpedPanels(k).Cells(i).LostAreaPercent), 'Color', 'white', 'FontWeight', 'bold'); end
            
            % Classify by damage type
            % Hole - only one blob in cell region and EulerNumber < 1
            % Crack - only one blob in cell region and EulerNumber == 1
            % Split - more than one blob in cell region
            if length(cellBlobs) == 1
                if warpedPanels(k).Cells(i).Blobs.EulerNumber < 1
                    warpedPanels(k).Cells(i).Holes = abs(warpedPanels(k).Cells(i).Blobs.EulerNumber) + 1;
                else
                    warpedPanels(k).Cells(i).Cracks = 1;
                end
            else 
                warpedPanels(k).Cells(i).Splits = length(cellBlobs) - 1;
            end
        end
    end
    
    % Set total crack count for panel
    warpedPanels(k).CrackCount = crackCounter;
end
end
