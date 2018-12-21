function varargout = colorSpaceViewer(varargin)
%COLORSPACEVIEWER MATLAB code file for colorSpaceViewer.fig
%      COLORSPACEVIEWER, by itself, creates a new COLORSPACEVIEWER or raises the existing
%      singleton*.
%
%      H = COLORSPACEVIEWER returns the handle to a new COLORSPACEVIEWER or the handle to
%      the existing singleton*.
%
%      COLORSPACEVIEWER('Property','Value',...) creates a new COLORSPACEVIEWER using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to colorSpaceViewer_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      COLORSPACEVIEWER('CALLBACK') and COLORSPACEVIEWER('CALLBACK',hObject,...) call the
%      local function named CALLBACK in COLORSPACEVIEWER.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help colorSpaceViewer

% Last Modified by GUIDE v2.5 05-Jul-2018 14:26:37

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @colorSpaceViewer_OpeningFcn, ...
                   'gui_OutputFcn',  @colorSpaceViewer_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before colorSpaceViewer is made visible.
function colorSpaceViewer_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for colorSpaceViewer
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

%% Disable toggleButtons
handles.buttonRGB.Enable = 'off';
handles.buttonHSV.Enable = 'off';
handles.buttonYCbCr.Enable = 'off';
handles.buttonYIQ.Enable = 'off';

% UIWAIT makes colorSpaceViewer wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = colorSpaceViewer_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in buttonRGB.
function buttonRGB_Callback(hObject, eventdata, handles)
% hObject    handle to buttonRGB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Copy I into rgb variable
rgb = evalin('base','I');

%% Copy rgb to workspace
assignin('base','rgb',rgb); 

%% Split image into color channels
[redChannel, greenChannel, blueChannel] = split_channels(rgb);

%% Create histograms
[x, yRedNorm, yGreenNorm, yBlueNorm] = create_histograms(redChannel, greenChannel, blueChannel, 1);

%% Plot images and histograms
update_plots(handles, redChannel, greenChannel, blueChannel, x, yRedNorm, yGreenNorm, yBlueNorm);
title(handles.imgax1, 'Red Channel');
title(handles.imgax2, 'Green Channel');
title(handles.imgax3, 'Blue Channel');

%% Press RGB and depress other buttons
handles.buttonRGB.Value = 1;
handles.buttonHSV.Value = 0;
handles.buttonYCbCr.Value = 0;
handles.buttonYIQ.Value = 0;


% --- Executes on button press in buttonHSV.
function buttonHSV_Callback(hObject, eventdata, handles)
% hObject    handle to buttonHSV (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Load I from workspace
I = evalin('base','I');

%% Convert I to hsv colorspace
hsv = uint8(rgb2hsv(I)*255);

%% Copy hsv to workspace
assignin('base','hsv',hsv); 

%% Split image into color channels
[hueChannel, satChannel, valueChannel] = split_channels(hsv);

%% Create histograms
[x, yHueNorm, ySatNorm, yValueNorm] = create_histograms(hueChannel, satChannel, valueChannel, 1);

%% Plot images and histograms
update_plots(handles, hueChannel, satChannel, valueChannel, x, yHueNorm, ySatNorm, yValueNorm);
title(handles.imgax1, 'Hue Channel');
title(handles.imgax2, 'Saturation Channel');
title(handles.imgax3, 'Value Channel');

%% Press RGB and depress other buttons
handles.buttonRGB.Value = 0;
handles.buttonHSV.Value = 1;
handles.buttonYCbCr.Value = 0;
handles.buttonYIQ.Value = 0;

% --- Executes on button press in buttonYCbCr.
function buttonYCbCr_Callback(hObject, eventdata, handles)
% hObject    handle to buttonYCbCr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Load I from workspace
I = evalin('base','I');

%% Convert I to ycbcr colorspace
ycbcr = rgb2ycbcr(I);

%% Copy hsv to workspace
assignin('base','ycbcr',ycbcr); 

%% Split image into color channels
[yChannel, cbChannel, crChannel] = split_channels(ycbcr);

%% Create histograms
[x, yYNorm, yCbNorm, yCrNorm] = create_histograms(yChannel, cbChannel, crChannel, 1);

%% Plot images and histograms
update_plots(handles, yChannel, cbChannel, crChannel, x, yYNorm, yCbNorm, yCrNorm);
title(handles.imgax1, 'Y Channel');
title(handles.imgax2, 'Cb Channel');
title(handles.imgax3, 'Cr Channel');

%% Press RGB and depress other buttons
handles.buttonRGB.Value = 0;
handles.buttonHSV.Value = 0;
handles.buttonYCbCr.Value = 1;
handles.buttonYIQ.Value = 0;

% --- Executes on button press in buttonYIQ.
function buttonYIQ_Callback(hObject, eventdata, handles)
% hObject    handle to buttonYIQ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of buttonYIQ

%% Load I from workspace
I = evalin('base','I');

%% Convert I to yiq colorspace
yiq = uint8(rgb2ntsc(I)*255);

%% Copy hsv to workspace
assignin('base','yiq',yiq); 

%% Split image into color channels
[yChannel, iChannel, qChannel] = split_channels(yiq);

%% Create histograms
[x, yYNorm, yINorm, yQNorm] = create_histograms(yChannel, iChannel, qChannel, 1);

%% Plot images and histograms
update_plots(handles, yChannel, iChannel, qChannel, x, yYNorm, yINorm, yQNorm);
title(handles.imgax1, 'Y Channel');
title(handles.imgax2, 'I Channel');
title(handles.imgax3, 'Q Channel');

%% Press RGB and depress other buttons
handles.buttonRGB.Value = 0;
handles.buttonHSV.Value = 0;
handles.buttonYCbCr.Value = 0;
handles.buttonYIQ.Value = 1;

% --- Executes on button press in buttonLoad.
function buttonLoad_Callback(hObject, eventdata, handles)
% hObject    handle to buttonLoad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Create a file dialog for images
[filename, ~] = imgetfile;

%% Display image name
handles.imageFile.String = filename;

%% Read the selected image into the variable
[I,map] = imread(filename);

%% Copy I and map to base workspace
assignin('base','I',I); 
assignin('base','map',map); 

%% Split image into color channels
[redChannel, greenChannel, blueChannel] = split_channels(I);

%% Create histograms
[x, yRedNorm, yGreenNorm, yBlueNorm] = create_histograms(redChannel, greenChannel, blueChannel, 1);

%% Plot images and histograms
update_plots(handles, redChannel, greenChannel, blueChannel, x, yRedNorm, yGreenNorm, yBlueNorm);
title(handles.imgax1, 'Red Channel');
title(handles.imgax2, 'Green Channel');
title(handles.imgax3, 'Blue Channel');

%% Enable toggle buttons
handles.buttonRGB.Enable = 'on';
handles.buttonHSV.Enable = 'on';
handles.buttonYCbCr.Enable = 'on';
handles.buttonYIQ.Enable = 'on';

%% Press RGB and depress other buttons
handles.buttonRGB.Value = 1;
handles.buttonHSV.Value = 0;
handles.buttonYCbCr.Value = 0;
handles.buttonYIQ.Value = 0;

function [chan1, chan2, chan3] = split_channels(I)
%% Extract color space channels.
chan1_gray = I(:,:,1); 
chan2_gray = I(:,:,2); 
chan3_gray = I(:,:,3); 

%% Create an all black channel.
allBlack = zeros(size(I, 1), size(I, 2), 'uint8');

%% Create color versions of the individual color channels.
chan1 = cat(3, chan1_gray, allBlack, allBlack);
chan2 = cat(3, allBlack, chan2_gray, allBlack);
chan3 = cat(3, allBlack, allBlack, chan3_gray);

function [histx, hist1, hist2, hist3] = create_histograms(chan1, chan2, chan3, normalize)
%% Get histValues for each channel
[hist1, histx] = imhist(chan1(:,:,1));
[hist2, ~] = imhist(chan2(:,:,2));
[hist3, ~] = imhist(chan3(:,:,3));

%% Normalize histValues if normalize is true
if normalize
    [rows, cols, ~] = size(chan1);
    totalPixels = rows*cols;
    hist1 = (hist1/totalPixels)*100;
    hist2 = (hist2/totalPixels)*100;
    hist3 = (hist3/totalPixels)*100;
end

function update_plots(handles, img1, img2, img3, histx, hist1, hist2, hist3)
%% Plot images and histograms
imshow(img1, 'Parent', handles.imgax1);
imshow(img2, 'Parent', handles.imgax2);
imshow(img3, 'Parent', handles.imgax3);
plot(histx, hist1, 'Red', 'Parent', handles.histax1);
plot(histx, hist2, 'Green', 'Parent', handles.histax2);
plot(histx, hist3, 'Blue', 'Parent', handles.histax3);
