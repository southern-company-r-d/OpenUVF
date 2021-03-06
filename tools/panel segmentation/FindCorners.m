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

function panels = FindCorners(closed, plots)
%% FindCorners.m 
% Joey Richardson - Southern Company Services 2018
% This program finds accurate estimations of the corners of each full solar
% panel in an image. 

% Inputs:
% closed      - (binary image) Input binary image with each panel 'closed'.
%                A panel is considered closed if the entire module is 1 and
%                the individual cells have been joined.
% plots       - (optional boolean default false) Determines whether plots 
%                will be shown or not
%
% Required Functions:
% Ellipse.m - Stack Overflow
% Image Processing Toolbox - Mathworks
% InterX.m - NS
% LinearFit.m - Joey Richardson, Southern Company Services


% filename = '2 axis binary.JPG';
% imageFile = fullfile(pwd(), 'images', filename);
% image = imread(imageFile);
% bw = image > 245;


if nargin < 2
    plots = true;
end

[rows,cols] = size(closed);

% Find the boundary of each panel disregarding holes from image 'closing'
[binaryBoundary, labelBoundary] = bwboundaries(closed, 'noholes');

% Get the FilledArea and Perimeter region properties from labelBoundary
stats = regionprops(labelBoundary, 'FilledArea', 'Perimeter', 'Extrema');

%Drop identified panels that do not meet edge criteria (intersect edge)
edgePanel = any(reshape(round([stats.Extrema]) == 1 | round([stats.Extrema]) == rows | floor([stats.Extrema]) == cols , 16, length(stats)))';
stats = stats(~edgePanel);
binaryBoundary = binaryBoundary(~edgePanel);

%Drop misshapen panels (merged or only partially included panels)
% areaPerimeterRatios = ([stats.Perimeter].^2)./[stats.FilledArea];
% oddPanelShape = areaPerimeterRatios > 18+6 | areaPerimeterRatios < 18-6;
% stats = stats(~oddPanelShape);
% binaryBoundary = binaryBoundary(~oddPanelShape);

%Drop identified panels that do not meet area criteria (mostly noise and
%   other rows)
fullPanelArea = transpose([stats.FilledArea] > 0.67*max([stats.FilledArea])); %.75
stats = stats(fullPanelArea); %Removing all areas that do not meet area requirement to avoid excessive perimeters
binaryBoundary = binaryBoundary(fullPanelArea);

%Drop identified panels that do not meet perimeter criteria
fullPanelPerimeter = transpose([stats.Perimeter] > max([stats.Perimeter])*0.5); %.9
stats = stats(fullPanelPerimeter);
binaryBoundary = binaryBoundary(fullPanelPerimeter);

%Labeling Panels
panels = stats;
[panels(:).Boundary] = binaryBoundary{:};%{fullPanelArea & fullPanelPerimeter & ~edgePanel};

%Crop image to only include 
extremas = reshape([panels(:).Extrema]', 8*length(panels), 2);




% Create bounding rectangle for total image (for identifying corners in the image bounds)
xvRect = [0, cols, cols, 0,    0];
yvRect = [0, 0,    rows, rows, 0];

if plots, figure; imshow(label2rgb(labelBoundary, @jet, [98 98 98]./255), 'Border', 'tight'); title('Label Matrix'); hold on; end
for k = length(panels):-1:1
    %% Get boundary for panel
    boundary = panels(k).Boundary;
    xboundary = boundary(:,2);
    %xboundary = movmean(xboundary, 20);
    %xboundary = medfilt2(xboundary, [20, 1]);
    yboundary = boundary(:,1);
    %yboundary = medfilt2(yboundary, [20, 1]);
    %yboundary = movmean(yboundary, 20);
    boundaryinds = sub2ind(xboundary, yboundary);
    
    %% Find corner approximations
    
    %Wrap in try catch
    try %angle differential approach
        %Calculate differential angles
        diffangs = atand(diff(yboundary)./diff(xboundary));
        diffangs = [diffangs; diffangs(1:round(.075 .* length(diffangs)))]; %wrapping as the angs start near a corner almost always
    
        %Smooth
        filtwidth = round(.075 .* length(xboundary)); %7.5% of the total element count
        angs = abs(movmean(medfilt2(movmean(diffangs, 30), [filtwidth, 1]), 30));

        %Identify corner candidates
        sides = angs > mean(angs); %binary output says if boundary areas belong
        candidates = find(abs(diff(sides)) == 1);
        if length(candidates) == 3
            candidates = [1; candidates]; %first index is a corner
        else
            candidates = candidates(1:4); %take only first four - will error if less than 3 corners observed
        end
        candidates = mod(candidates, length(xboundary));
        candidates(candidates == 0) = 1;
        
        %Define corners
        corners = [xboundary(candidates), yboundary(candidates)];
        [~, topLeft] = min(sqrt(corners(:,1).^2 + corners(:,2).^2));
        [~, bottomRight] = max(sqrt(corners(:,1).^2 + corners(:,2).^2));
        [~, topRight] = min(sqrt((corners(:,1)-cols).^2 + corners(:,2).^2));
        [~, bottomLeft] = max(sqrt((corners(:,1)-cols).^2 + corners(:,2).^2));
        topLeft = candidates(topLeft);
        bottomRight = candidates(bottomRight);
        topRight = candidates(topRight);
        bottomLeft = candidates(bottomLeft);
        
        
    catch ME %default to simple method    
        [~, topLeft] = min(sqrt(xboundary.^2 + yboundary.^2));
        [~, bottomRight] = max(sqrt(xboundary.^2 + yboundary.^2));
        [~, topRight] = min(sqrt((xboundary-cols).^2 + yboundary.^2));
        [~, bottomLeft] = max(sqrt((xboundary-cols).^2 + yboundary.^2));
    end

    if plots, figure; imshow(label2rgb(labelBoundary, @jet, [98 98 98]./255), 'Border', 'tight'); title('Label Matrix'); hold on; end

    %% Find x and y coordinates of each panel side    
    % Algorithm:
    % 1 - Draw ellipse between corner approximations
    % 2 - Find all points on panel boundary within each ellipse
    % 3 - Run a linear regression on points found to find best fit line
    % 4 - Extrapolate line to edges of the image
%     [xEllipse, yEllipse] = Ellipse(topLeft(2), topLeft(1), bottomLeft(2), bottomLeft(1), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, leftc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
%     leftx = [polyval(leftc, 0), polyval(leftc, rows)]; lefty = [0, rows];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(leftx, lefty, 'black', 'LineWidth', 2); end
%     
%     [xEllipse, yEllipse] = Ellipse(topRight(2), topRight(1), bottomRight(2), bottomRight(1), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, rightc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
%     rightx = [polyval(rightc, 0), polyval(rightc, rows)]; righty = [0, rows];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(rightx, righty, 'black', 'LineWidth', 2); end
%     
%     [xEllipse, yEllipse] = Ellipse(topLeft(2), topLeft(1), topRight(2), topRight(1), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, topc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
%     topx = [0, cols]; topy = [polyval(topc, 0), polyval(topc, cols)];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(topx, topy, 'black', 'LineWidth', 2); end
%     
%     [xEllipse, yEllipse] = Ellipse(bottomLeft(2), bottomLeft(1), bottomRight(2), bottomRight(1), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, bottomc] = LinearFit(yboundary(in | on), xboundary(in | on), 1);
%     bottomx = [0, cols]; bottomy = [polyval(bottomc, 0), polyval(bottomc, cols)];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(bottomx, bottomy, 'black', 'LineWidth', 2); end
%     
    [xEllipse, yEllipse] = Ellipse(xboundary(topLeft), yboundary(topLeft), xboundary(bottomLeft), yboundary(bottomLeft), 0.9);
    [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
    [~, ~, leftc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
    leftx = [polyval(leftc, 0), polyval(leftc, rows)]; lefty = [0, rows];
    if plots, plot(xEllipse,yEllipse,'w-'); plot(leftx, lefty, 'black', 'LineWidth', 2); end
    
    [xEllipse, yEllipse] = Ellipse(xboundary(topRight), yboundary(topRight), xboundary(bottomRight), yboundary(bottomRight), 0.9);
    [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
    [~, ~, rightc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
    rightx = [polyval(rightc, 0), polyval(rightc, rows)]; righty = [0, rows];
    if plots, plot(xEllipse,yEllipse,'w-'); plot(rightx, righty, 'black', 'LineWidth', 2); end
    
    [xEllipse, yEllipse] = Ellipse(xboundary(topLeft), yboundary(topLeft), xboundary(topRight), yboundary(topRight), 0.9);
    [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
    [~, ~, topc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
    topx = [0, cols]; topy = [polyval(topc, 0), polyval(topc, cols)];
    if plots, plot(xEllipse,yEllipse,'w-'); plot(topx, topy, 'black', 'LineWidth', 2); end
    
    [xEllipse, yEllipse] = Ellipse(xboundary(bottomLeft), yboundary(bottomLeft), xboundary(bottomRight), yboundary(bottomRight), 0.9);
    [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
    [~, ~, bottomc] = LinearFit(yboundary(in | on), xboundary(in | on), 1);
    bottomx = [0, cols]; bottomy = [polyval(bottomc, 0), polyval(bottomc, cols)];
    if plots, plot(xEllipse,yEllipse,'w-'); plot(bottomx, bottomy, 'black', 'LineWidth', 2); end
    
    
    %% Find true corners of panel
    % Find the intersection point of each extrapolated linear regression line
    TL = CornerIntersects([leftx;lefty],[topx;topy]);
    BL = CornerIntersects([leftx;lefty],[bottomx;bottomy]);
    TR = CornerIntersects([rightx;righty],[topx;topy]);
    BR = CornerIntersects([rightx;righty],[bottomx;bottomy]);
    
    %% Determine if panels are in the image space
    corners = [TL'; BL'; TR'; BR'];
    corners = corners;
    in = inpolygon(corners(:,1), corners(:,2), xvRect, yvRect);
    
    % Drop panels with less than 4 corners in the image space or if two
    % corners are both on the edge of the image
    if length(in) < 4 || (TL(1,:) < 10 && BL(1,:) < 10) || (TR(1,:) > cols-10 && BR(1,:) > cols-10) ...
                      || (TL(1,:) < 10 && TR(1,:) < 10) || (BL(1,:) > rows-10 && BR(1,:) > rows-10)
        panels(k) = [];
    else
        % Plot of corners
        if plots
            plot(TL(1,:), TL(2,:), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white', 'MarkerSize', 10)
            plot(BL(1,:), BL(2,:), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white', 'MarkerSize', 10)
            plot(TR(1,:), TR(2,:), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white', 'MarkerSize', 10)
            plot(BR(1,:), BR(2,:), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white', 'MarkerSize', 10)
        end
            
        % Save corners to panels structure
        panels(k).Corners.topLeft = [TL(1,:), TL(2,:)];
        panels(k).Corners.bottomLeft = [BL(1,:), BL(2,:)];
        panels(k).Corners.topRight = [TR(1,:), TR(2,:)];
        panels(k).Corners.bottomRight = [BR(1,:), BR(2,:)];

    end 
end
end






%     %Alternative approach - look for significant changes in the direction
%     %   of the points
%        angco = 25;
%     
%        %Defining Boundary Sample Points 
%        sampleinds = floor(linspace(1, length(boundary), 41));
%        sampleinds = sampleinds(1:end-1);
%        samplepts = boundary(sampleinds, :);
%        samplexs = samplepts(:,2);
%        sampleys = samplepts(:,1);
%        
%        %Defining Angles between Sample Points
%        samplediffsx = diff(samplexs);
%        samplediffsx_neg = samplediffsx < 0;
%        samplediffsy = diff(sampleys);
%        sampleangs = atand(samplediffsy./samplediffsx);
%        sampleangs(samplediffsx_neg) = atand(samplediffsy(samplediffsx_neg)./samplediffsx(samplediffsx_neg));
%        sampleangs = abs(medfilt2(sampleangs, [round(.1 * length(sampleangs)), 1]));
%        %sampleangs = abs(movmean(sampleangs, round(.15 * length(sampleangs))));
%        %Defining the boundaries of the corners
%        cornerbounds = find(abs(diff(sampleangs > mean(sampleangs))) == 1);
%        cornerbounds = [cornerbounds , cornerbounds+4];
%        cornerbounds = mod(cornerbounds, length(sampleinds));
%        cornerbounds(cornerbounds == 0) = length(sampleinds);
%        if length(cornerbounds) == 3
%            cornerbounds = [cornerbounds; length(sampleinds), 2];
%        end
%        boundinds = sampleinds(cornerbounds(:));
%        xcornerbounds = samplepts(cornerbounds(:), 2);
%        ycornerbounds = samplepts(cornerbounds(:), 1);
%        
%        %Check that corner bounds has found four corners - should only have
%        %   four edges 
%        if length(cornerbounds) < 4
%            problem = true; 
%        end
%        
%        %Defining edge bounds from cornerbounds
%        edges = cell(4,1);
%        if cornerbounds(end,end) == 2
%            edges{1} = cornerbounds(4,2):cornerbounds(1,1);
%            edges{2} = cornerbounds(1,2):cornerbounds(2,1);
%            edges{3} = cornerbounds(2,2):cornerbounds(3,1);
%            edges{4} = cornerbounds(3,2):cornerbounds(4,1);
%        else %Case of wrapping
%            edges{1} = [cornerbounds(4,2):length(sampleinds), 1:cornerbounds(1,1)];
%            edges{2} = cornerbounds(1,2):cornerbounds(2,1);
%            edges{3} = cornerbounds(2,2):cornerbounds(3,1);
%            edges{4} = cornerbounds(3,2):cornerbounds(4,1);
%        end
%        
%        %Deciding which edge is which
%        edges_mean_vals = zeros(4,2); %x, y used for allocating edges
%        edges_std = zeros(4,2);
%        for e = 1:4
%             edges_mean_vals(e,:) = mean(samplepts(edges{e},:));  
%             edges_std(e, :) = std(samplepts(edges{e},:));     
%        end
%        [~,te] = min(edges_mean_vals(:,1));
%        [~,be] = max(edges_mean_vals(:,1));
%        [~,re] = max(edges_mean_vals(:,2)); 
%        
%        [~,le] = min(edges_mean_vals(:, 2));
%       
%        
%        %Perform line fits on sample pts, dividing them with the angle
%        %method
%        fitscoeffs = zeros(length(cornerbounds), 2);
%        fitsxvals = 1:rows;
%        fitsyvals = zeros(length(cornerbounds), rows);
%        for e = 1:4
%            %pulling edge 
%            edge = edges{e};      
%            
%            %Calculating linear edge fits
%            secvals = samplepts(edge, :);
%            xsecvals = secvals(:,1);
%            ysecvals = secvals(:,2);
%            fitscoeffs(e, :) = polyfit(xsecvals, ysecvals, 1);  
%            fitsyvals(e,:) = polyval(fitscoeffs(e,:), fitsxvals);
%        end
%        topRight = round(InterX([fitsxvals; fitsyvals(re,:)], [fitsxvals; fitsyvals(te,:)]));
%        bottomRight = round(InterX([fitsxvals; fitsyvals(re,:)], [fitsxvals; fitsyvals(be,:)]));
%        bottomLeft = round(InterX([fitsxvals; fitsyvals(le,:)], [fitsxvals; fitsyvals(be,:)]));
%        topLeft = round(InterX([fitsxvals; fitsyvals(le,:)], [fitsxvals; fitsyvals(te,:)]));
%        corners = [topLeft';topRight';bottomLeft';bottomRight'];    
%               
%         if plots & true
%            figure; 
%            subplot(1,2,1); 
%            plot(xboundary, yboundary);
%            hold on; 
%            plot(samplexs, sampleys, 'ro'); 
%            plot(xcornerbounds, ycornerbounds, 'bo');
%            for c = 1:4
%                 plot(corners(c,2), corners(c,1), '*');
%            end
%            subplot(1,2,2); 
%            plot(sampleangs); 
%         end
       
%       
%        
%       for c = 1:length(boundinds)/2;
%           boundarysec = boundary(boundaryinds(2*c-1, 2*c));
%           
%           
%       end
    
%     % Find top left corner and bottom right corner approximations
%     % Top left = min distance to top left of image (X = 0, Y = 0)
%     % Bottom right = max distance to top left of image (X = 0, Y = 0)
%      [~, topLeft] = min(sqrt(xboundary.^2 + yboundary.^2));
%     [~, bottomRight] = max(sqrt(xboundary.^2 + yboundary.^2));
%     
%     % Find top right corner and bottom left corner approximations
%     % Top right = min distance to top right of image (X = columns, Y = 0)
%     % Bottom left = max distance to top right of image (X = columns, Y = 0)
%     [~, topRight] = min(sqrt((xboundary-cols).^2 + yboundary.^2));
%     [~, bottomLeft] = max(sqrt((xboundary-cols).^2 + yboundary.^2));









% [xEllipse, yEllipse] = Ellipse(xboundary(topLeft), yboundary(topLeft), xboundary(bottomLeft), yboundary(bottomLeft), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, leftc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
%     leftx = [polyval(leftc, 0), polyval(leftc, rows)]; lefty = [0, rows];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(leftx, lefty, 'black', 'LineWidth', 2); end
%     
%     [xEllipse, yEllipse] = Ellipse(xboundary(topRight), yboundary(topRight), xboundary(bottomRight), yboundary(bottomRight), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, rightc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
%     rightx = [polyval(rightc, 0), polyval(rightc, rows)]; righty = [0, rows];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(rightx, righty, 'black', 'LineWidth', 2); end
%     
%     [xEllipse, yEllipse] = Ellipse(xboundary(topLeft), yboundary(topLeft), xboundary(topRight), yboundary(topRight), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, topc] = LinearFit(yboundary(in | on), xboundary(in | on), 3);
%     topx = [0, cols]; topy = [polyval(topc, 0), polyval(topc, cols)];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(topx, topy, 'black', 'LineWidth', 2); end
%     
%     [xEllipse, yEllipse] = Ellipse(xboundary(bottomLeft), yboundary(bottomLeft), xboundary(bottomRight), yboundary(bottomRight), 0.9);
%     [in, on] = inpolygon(xboundary, yboundary, xEllipse, yEllipse); 
%     [~, ~, bottomc] = LinearFit(yboundary(in | on), xboundary(in | on), 1);
%     bottomx = [0, cols]; bottomy = [polyval(bottomc, 0), polyval(bottomc, cols)];
%     if plots, plot(xEllipse,yEllipse,'w-'); plot(bottomx, bottomy, 'black', 'LineWidth', 2); end
