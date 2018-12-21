function [panels] = GetCellProperties(panels, imgType, modOrCell, numCells, numCellsWide, ib, plots)
%% GetCellProperties.m 
% Joey Richardson, Will Hobbs - Southern Company Services 2018
% This function segments cells in each panel and determines various image
% properties about each cell for cell classification.

% Inputs:
% panels - (struct) Structure containing panel information from
%                 CrackIdentification program
% maskedImage  - (image) Perspective corrected image masked to only contain
%                 full panels with a black background
% imgType      - (integer) Determines subset of images being processed.
%                Allows function to be used for multiple different types of
%                input images
% modOrCell    - (char or string) Determines whether processing is being
%                applied to prepare for panel segmentation or cell
%                segmentation as they require different levels of precision
% numCells     - (integer) Total number of cells per module in image
% numCellsWide - (integer) Width of module in cells
% imageBuffer  - (integer) Size of padding to add to each panels(k).Image panel.
%                 Improves accuracy of cell segmentation when adjusted
%                 properly. 
% plots        - (optional boolean default false) Determines whether plots 
%                will be shown or not
%
% Required Functions:
% BinaryConversion.m - Joey Richardson, Southern Company Services
% Image Processing Toolbox - Mathworks

if nargin < 8
    plots = false;
end

n = length(panels);
totcells = numCells * n;
for k = 1:n
    
    %Defining Image Propertiesa
    [imrows, imcols, ~] = size(panels(k).Image);

    %Initial Binarization for image buffering
    bw = BinaryConversion(panels(k).Image, imgType, 'm', plots, numCells, numCellsWide);  
    
    %Defining cell counts
    numCellsHigh = numCells/numCellsWide;
    
    %Contextual Image Buffering - Analyze the spacing between cells to find
    %   ideal buffer size (defined as half the 99th percentile of intercell
    %   spacing
    rowsct = sort(sum(bw));
    rowsct = length(find(sum(bw) < 20));
    colsct = sort(sum(bw, 2));
    colsct = length(find(sum(bw,2) < 20));
    ib = round(mean([rowsct/(numCellsWide-1), colsct/(numCellsHigh-1)])./2) - 2;
    if ib < 0
        ib = 1;
    end
%     rowsct = rowsct(round(.95 * length(rowsct))); %take 95th percentile of 
%        % the number of positive elements in each row
%     colsct = colsct(round(.95 * length(colsct)));
    %ib = round((((imrows - rowsct)/(numCellsHigh-1)) + ((imcols - colsct)/(numCellsWide - 1)))...
        %/4);
    panels(k).Image = padarray(panels(k).Image, [ib, ib], 0);    
    if plots, figure; imshow(panels(k).Image, 'Border', 'tight'); title('panels(k).Image Panel'); end 
    
    % Cell Level Binary Conversion
    panels(k).Binary = BinaryConversion(panels(k).Image, imgType, modOrCell, plots, numCells, numCellsWide);
    
    % Find the boundary of each cell disregarding holes from image 'closing'
    [~, labelBoundary] = bwboundaries(bw, 'noholes');        
    
    % Pad each panel with buffer to improve cell segmentation accuracy
    bufferedImg = panels(k).Binary;%padarray(labelBoundary, [ib, ib], 0);
    panels(k).LabelMatrix = bufferedImg;
    
    % Find bounding region of each cell in image and calculate cell properties
    % Properties are calculated for each cell rather than the whole image
    % to maintain blobs per cell (ie if there are multiple blobs within a
    % single cell region because a cell is completely split).
    cellCounter = 1;   
    widthCell = size(bufferedImg,2)/numCellsWide;
    heightCell = size(bufferedImg,1)/numCellsHigh;
    for i = 1:numCellsHigh
        if i==1
            startRow = 1;
        else
            startRow = round((i-1)*heightCell);
        end
        stopRow = round(i*heightCell);
        for j = 1:numCellsWide
            if j==1
                startCol = 1;
            else
                startCol = round((j-1)*widthCell);
            end
            stopCol = round(j*widthCell);
            
            % Write rectangle of cell region to panels structure
            panels(k).Cells(cellCounter).Rectangle = [startCol startRow stopCol-startCol stopRow-startRow];
        
            % Get cell from panel region
            cell = bufferedImg(startRow:stopRow,startCol:stopCol, :);
            cell_color = panels(k).Image(startRow:stopRow,startCol:stopCol, :);
                        
            % Get region properties of the cell region
            stats = regionprops(cell, 'Centroid', 'Area', 'Perimeter', 'Solidity', 'FilledArea', 'Extent', 'EulerNumber', 'BoundingBox');
            
            % Discard connected regions with an area less than 50 pixels
            fullCellArea = transpose([stats.Area] > 300);
            stats = stats(fullCellArea);      
   
            % Assign values to panels structure
            panels(k).Cells(cellCounter).CellImage = cell;
            panels(k).Cells(cellCounter).CellColorImage = cell_color;
            panels(k).Cells(cellCounter).Cracked = false;
            panels(k).Cells(cellCounter).CrackType = '';
            panels(k).Cells(cellCounter).Label = strcat(num2str(i), char(j + 64)); % Follows IEC standard - assumes j box at top
            panels(k).Cells(cellCounter).LabelNumber = i + j/100;
            
            for m = 1:length(stats)
                panels(k).Cells(cellCounter).Blobs(m).Centroid = stats(m).Centroid;
                panels(k).Cells(cellCounter).Blobs(m).Area = stats(m).Area;
                panels(k).Cells(cellCounter).Blobs(m).Perimeter = stats(m).Perimeter;
                panels(k).Cells(cellCounter).Blobs(m).Solidity = stats(m).Solidity;
                panels(k).Cells(cellCounter).Blobs(m).FilledArea = stats(m).FilledArea;
                panels(k).Cells(cellCounter).Blobs(m).Extent = stats(m).Extent;
                panels(k).Cells(cellCounter).Blobs(m).EulerNumber = stats(m).EulerNumber;
                panels(k).Cells(cellCounter).Blobs(m).FilledRatio = stats(m).Area / stats(m).FilledArea;
                panels(k).Cells(cellCounter).Blobs(m).AreaToPerimeter = stats(m).Area / stats(m).Perimeter;
                panels(k).Cells(cellCounter).Blobs(m).BoundingBox = stats(m).BoundingBox;
            end
            
            % Increment Cell Counter
            cellCounter = cellCounter + 1;
        end
    end

end
end