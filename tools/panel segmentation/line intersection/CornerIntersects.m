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

function IX = CornerIntersects(L1, L2)

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
