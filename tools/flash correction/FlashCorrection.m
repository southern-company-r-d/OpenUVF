function flashCorrected = FlashCorrection(I, flashFolder, plots)
%% FlashCorrection.m 
% Joey Richardson - Southern Company Services 2018
% Copyright © 2019 Southern Company Services, Inc.
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
