%   Copyright 2019 Southern Company. 
%
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%       http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.

function [closed, I] = close_image(bw, I, plots)

        %Removing Definite Noise
        bw = bwareafilt(bw, [1000, 10000000]); %Minimum size requirement

        % Applying an initial blur filter
        blurred = imgaussfilt(uint8(bw.*255), 2);
        bw = blurred > 64;

        % Contextual Image Closing - Identify Cells
        %bw = bwareafilt(bw, [1000, 10000000]); %Minimum size requirement
        bwregions = bwconncomp(bw);
        bwprops = regionprops(bwregions, 'Area', 'Perimeter', 'Solidity', 'Centroid', 'BoundingBox');        
        bwareas = [bwprops.Area];
        bwperims = [bwprops.Perimeter];
        bwsolidities = [bwprops.Solidity];
        bwboxes = bwboundaries(bw, 8, 'noholes');
        bwcentroids = [bwprops.Centroid];
        bwcentroids = reshape(bwcentroids, [2, length(bwcentroids)/2])';
        %bwboxes = [bwprops(.BoundingBox];
        bwdists = sqrt(sum((bwcentroids - mean(bwcentroids)).^2, 2))'; %remove outliers
        bwratios = 16 .* (bwareas./(bwperims.^2)); %a perfect square will have a ratio of 1.
        squares = find(bwratios < 1.25 & bwratios > .75 & bwsolidities > .5 & bwdists < 1000); %identify objects that appear to be squares
        bwpoints = bwregions.PixelIdxList(squares);
        bwlabels = bwlabel(bw);
        if plots, figure; imshow(ismember(bwlabels, squares));...
                title('Contextual Image Closing - Identified Squares'); end
        
        %Removing definite non-cells
        posscells = find(bwsolidities >.3 & bwratios < 1.75 & bwratios > .25 & bwdists < max(size(bw))/2);
        %bw = ismember(bwlabels, posscells);
        
        % Contextual Image Closing - Determine Distance between Neighboring
        %   Cell Boundaries
        bwcentroids = bwcentroids(squares, :);
        %bwboxes = bwboxes(squares);
        netcenter = mean(bwcentroids);
        Xcentroids = bwcentroids(:,1);
        Ycentroids = bwcentroids(:,2);
        Xinterdists = repmat(Xcentroids, 1, length(Xcentroids));
        Xinterdists = abs(Xinterdists - Xinterdists');
        Yinterdists = repmat(Ycentroids, 1, length(Ycentroids));
        Yinterdists = abs(Yinterdists - Yinterdists');
        Ninterdists = sqrt(Xinterdists.^2 + Yinterdists.^2);
        [Ninterdists, NearNeighbors] = sort(Ninterdists);
        NearNeighbors = NearNeighbors(2:5, :);
        NearInterdists = zeros(4, length(Xinterdists));
        for i = 1:length(Xcentroids)
            
            
            
            %Isolating Neighbirs of this cell
            neighbors = NearNeighbors(:,i);
            
            for n = 1:4
                %Pulling Box
                cellbox = bwboxes{squares(i)};
                
                %Pulling the boundary of the neighbor
                ind = squares(neighbors(n));
                neighbox = bwboxes{ind};
                
                %Calculating the distance between points on the boxes
                cellbox = reshape(cellbox, length(cellbox),1,2);
                neighbox = reshape(neighbox, length(neighbox), 1, 2);
                cellbox = repmat(cellbox,1, length(neighbox));
                neighbox = repmat(neighbox, 1, (numel(cellbox)/(length(neighbox)*2)));
                neighbox = permute(neighbox, [2 1 3]);
                interdists = (cellbox-neighbox).^2;
                interdists = sqrt(interdists(:,:,1)+interdists(:,:,2));
                NearInterdists(n, i) = min(interdists(:));
                
                
            end
            
            
            
        end
        
        % Contextual Image Closing - Defining closing distance from nearest
        %   neighbors
        NearInterdists = sort(NearInterdists(:));
        cd = ceil(NearInterdists(round(.85.*length(NearInterdists)))); 
        
        % Contextual Image Closing - Apply Gaussian Blur and Threshhold
        bw = imgaussfilt(double(bw), 10);
        bw = bw > .25;
        
        %Contextual Image Closing - Defining the mean orientation of the
        %panels
        center = mean(bwcentroids);
        dist_from_center = sqrt(sum(diff(bwcentroids - repmat(center, length(bwcentroids), 1)).^2, 2));
        incircle = find(dist_from_center < 500);
        ccoeffs = polyfit(bwcentroids(incircle,1), bwcentroids(incircle,2), 1);
        cang = atand(ccoeffs(1)) + 90; %
        
        % Close shapes with rectangular structure element to remove panel bus bars
        % *Structuring element is image size dependent 
        %closed = imclose(bw, strel('rectangle', [cd, cd])); %35 Can contextualize by finding the area of cells before conducting this
        closed = imclose(bw, strel('line', cd, cang));
        closed = imclose(closed, strel('line', cd, cang-90));
        closed = imfill(closed, 'holes');
        if plots, figure; imshow(closed, 'Border', 'tight'); title('Closed Image'); end
        closed = imerode(closed, strel('disk', 5));
        
        %Image rotation - possible solution to corner detection issues
%         closed = imrotate(closed, (cang-90), 'bilinear');
%         I = imrotate(I, (cang-90), 'bilinear');
        if plots, figure; imshow(closed, 'Border', 'tight'); title('Closed Image Eroded + Rotated'); end
end

%Iterative line fit refinement approach
%         repeat = true;
%         it = 1;
%         co = 600;
%         while repeat
%             %Compute linear fit
%             ccoeffs = polyfit(bwcentroids(:,1), bwcentroids(:,2), 1);
%             
%             %Compute line
%             xs = bwcentroids(:,1);
%             ys = polyval(ccoeffs, xs);
%             
%             %Calculate R2 Value
%             R = corrcoef(ys, bwcentroids(:,2));
%             R2 = R(2)^2;
%             
%             %Debug plotting
%             if plots & true
%                 hold on;
%                 plot(bwcentroids(:,1), bwcentroids(:,2), 'ro')
%                 plot(xs, ys, 'r--', 'LineWidth', 5)
%             end
%             
%             %Remove outliers
%             
%             outliers = abs(ys - bwcentroids(:,2)) > co;
%             nonoutliers = find(~outliers);
%             bwcentroids = bwcentroids(nonoutliers, :);
%             
%             
%             co = co - 10;
%             if co < (3*cd); co = (3*cd); end
%             it = it + 1;
%             
%             %Take closest points and 
%             if co == (2*cd)
%                 repeat = false;
%                 xbounds = linspace(min(xs), max(xs)+1, 11);
%                 bwcentroids_new = zeros(10,2);
%                 for p = 1:length(xbounds) - 1
%                     pts = find(xs >= xbounds(p) & xs < xbounds(p+1));
%                     dists = abs(bwcentroids(pts, 2) - ys(pts));
%                     bwcentroids_new(p, :) = bwcentroids(find(bwcentroids(pts,2) == min(dists)), :);
%                 end   
%                 bwcentroids = bwcentroids_new;
%             end
%             
%             if any(outliers)
%                 hold off
%                 imshow(ismember(bwlabels, squares))
%             end
%             
%         end