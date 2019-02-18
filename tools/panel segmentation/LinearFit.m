function [fit_rows, fit_columns, coefficients] = LinearFit(rows, cols, iterations)
% LinearFit.m 
% Joey Richardson - Southern Company Services 2018
% Copyright © 2019 Southern Company Services, Inc.
% Linear regression function for images. Used in the CrackIdentification
% script when segmenting panels to find the best fit edges of each panel
% side. 
%
% Inputs:
% rows        - (array) Rows of image boundary to fit line to
% cols        - (array) Columns of image boundary to fit line to
% iterations  - (integer) Number of times to run regression analysis
%
% Required Functions:
%

% Figure out largest range of input data
% A greater range of data on the x axis of the regression analysis will
% allow for a greater number of outputs on the y axis after evaluating the
% polynomial fit. 
rowRange = max(rows) - min(rows);
colRange = max(cols) - min(cols);
if rowRange > colRange
    xy = [rows, cols];
else
    xy = [cols, rows];
end

% Sort rows of matrix xy by the x column to make them monotonically increasing
xy = sortrows(xy);

% Run once for each iteration 
% Break if standard deviation of error is less than 2
for i = 0:iterations 
    % Find polynomial coefficients of linear best fit line
    coeffs = polyfit(xy(:,1), xy(:,2), 1);
    
    % Evaluate polynomial for all x values in xy
    fity = polyval(coeffs, xy(:,1));
    
    % Find the error for each point and the standard deviation of the error
    err = fity - xy(:,2);
    err_std = std(err);

    % Break if standard deviation of error is less than 2
    % Else if less than max iterations, replace x in xy with all x values
    % less than than the standard deviation of the error
    if err_std < 2
        break
    elseif i < iterations 
        xy = xy(abs(err) < err_std, :);
    end
end

% Assign regression outputs
fit_rows = fity;
fit_columns = xy(:,1);
coefficients = coeffs;
end
