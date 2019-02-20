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

function flashCorrected = FlashCorrection(I, flashFolder, plots)
%% FlashCorrection.m 
% Joey Richardson - Southern Company Services 2018
% This program averages images of a camera flash to create a mask for
% correcting the uneven lighting across an image due to the flash.

% Inputs:
% I           - (image) Input image for binary conversion
% flashFolder - (char or string) Folder containing the correction image
%                within the tools/flash correction/correction images/
%                directory
% plots       - (optional boolean default false) Determines whether plots 
%                will be shown or not
%
% Required Functions:
% Image Processing Toolbox - Mathworks

if nargin < 3
    plots = false;
end

[Irows, Icols, ~] = size(I);

flashFile = fullfile(pwd(), 'tools', 'flash correction', 'correction images', flashFolder, 'correctionImage.JPG');
flash = imread(flashFile);
flashGray = rgb2gray(flash);
flashGrayNorm = double(flashGray)/double(max(max(flashGray)));
flashGrayNorm = imresize(flashGrayNorm, [Irows, Icols]);
flashCorrected = uint8(double(I)./flashGrayNorm);
if plots, figure; imshow(flashGrayNorm, 'Border', 'tight'); title('Flash Correction File'); end
if plots, figure; imshow(flashCorrected, 'Border', 'tight'); title('Flash Correction File'); end


end
