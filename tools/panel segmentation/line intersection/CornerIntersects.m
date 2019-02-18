function IX = CornerIntersects(L1, L2)
% Copyright © 2019 Southern Company Services, Inc.
    %Define Values
    L1Xs = L1(1, :);
    L1Ys = L1(2, :);
    L2Xs = L2(1, :);
    L2Ys = L2(2, :);
    
    %Define Line Fits
    L1fit = polyfit(L1Xs, L1Ys, 1);
    L1func = @(X) polyval(L1fit, X);
    L2fit = polyfit(L2Xs, L2Ys, 1);
    L2func = @(X) polyval(L2fit, X);
    
    %Find Intersection
    zerofunc = @(X) L1func(X) - L2func(X);
    interx = fzero(zerofunc, 1000);
    IX = ([interx; L1func(interx)]);
