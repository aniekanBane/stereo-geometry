function stereoGeometry = EstimateStereoGeometry(IL,IR,stereoParam,detector)
% ESTIMATESTEREOGEOMETRY Function which reads stereo images, rectifies the image pair using the calibrated 
% stereo parameters, detects the ROI using an object dector, uses the bounding 
% box as reference point for feature-based detection and extraction using  blob 
% and corner detectors and descriptors. The candidate matches are obtained and 
% used to compute the disparity where the gradient is == 0. The disparity is employed 
% to estimate the depth (z) and the displacements in the horizontal (x) and vertical 
% (y) axis. The function also outputs a disparity range which can be used to compute 
% a disparity map.

% Author : Aniekan Umanah jr.
% DATE   : April 2021
% Version: 1.000

% success flag
successFlag = 0;
%rectify stereo images
[J1,J2] = rectifyStereoImages(IL,IR,stereoParam,'OutputView',"valid");
dsize = detector.TrainingImageSize; % detector input size
B = abs(stereoParam.TranslationOfCamera2(1,1)); % distance between the cameras (baseline)
f = stereoParam.CameraParameters2.FocalLength(1,1); % focal length
uo = stereoParam.CameraParameters2.PrincipalPoint(1,1); % horizontal axis offset
vo = stereoParam.CameraParameters2.PrincipalPoint(1,2); % vertical axis offset
% size of rectified images
imsizeL = size(J1,[1 2]); 
imsizeR = size(J2,[1 2]); 
G = 0;
% size of rectified image equal to detector input size?
if  isequal(imsizeL,dsize) && isequal(imsizeR,dsize)
    G = 1;
    ImL = J1;
    ImR = J2;
else % (normal behavoir) resize images 
    ImL = imresize(J1,dsize); % resize left image
    ImR = imresize(J2,dsize); % resize right image
end  
% detect the bounding boxes for the cubes in the images
[bboxl, ~] = detect(detector,ImL); 
[bboxr, ~] = detect(detector,ImR);
if G == 1
    BL = bboxl;
    BR = bboxr;
else % (normal behavoiur)scale up bounding box
    sz = size(ImL,[1 2]);
    scale = (imsizeL ./sz); 
    BL = bboxresize(bboxl,scale); %left bbox
    BR = bboxresize(bboxr,scale); %right bbox
end
% convert the images to grayscale for 2D feature-based detection using
% BRISK and KAZE detectors
monoL = rgb2gray(J1);
monoR = rgb2gray(J2);
Points1 = detectBRISKFeatures(monoL,"ROI",BL); %left brisk points
Points2 = detectBRISKFeatures(monoR,"ROI",BR); %right brisk points
points3 = detectKAZEFeatures(monoL,"ROI",BL); %left kaze points
points4 = detectKAZEFeatures(monoR,"ROI",BR); %right kaze points
% extract and find candidate matches
[features1,valid_Points1] = extractFeatures(monoL,Points1,"Method","FREAK");
[features2,valid_Points2] = extractFeatures(monoR,Points2,"Method","FREAK");
[features3,valid_Points3] = extractFeatures(monoL,points3,"Method","KAZE");
[features4,valid_Points4] = extractFeatures(monoR,points4,"Method","KAZE");
% determine candidate matches
indexPairsbrisk = matchFeatures(features1,features2,'MaxRatio',0.7);
indexPairskaze = matchFeatures(features3,features4, "Unique",true);
% obtain candidate matches
matchedPoints1 = valid_Points1(indexPairsbrisk(:,1),:);
matchedPoints2 = valid_Points2(indexPairsbrisk(:,2),:);
matchedPoints3 = valid_Points3(indexPairskaze(:,1),:);
matchedPoints4 = valid_Points4(indexPairskaze(:,2),:);
% combine matched FREAK and KAZE features
matchedLeft = [matchedPoints1.Location; matchedPoints3.Location];
matchedRight = [matchedPoints2.Location; matchedPoints4.Location];
if size(matchedLeft,1) > 0
% find x,y,z
yl = matchedLeft(:,2); 
xl = matchedLeft(:,1);
yr = matchedRight(:,2); 
xr = matchedRight(:,1);
disP = zeros(size(yl));
if isequal(numel(yl),numel(yr)) % same number of element present?
    G = round(yl,4,"significant") - round(yr,4,"significant");
    if G ~= 0 % gradient is not zero?
         disP = 0;
    else % normal behavoiur
         disP = xl - xr;
    end
end
%D = {};
avgdisP = mean(disP);
z = f * B/ avgdisP; % depth
%D{1} = sprintf('%.8f',(z*10e-3));
 
y = vo - yl;
y = mean((z .* y) / f); % vertical displacement
%D{2} = sprintf('%.8f',(y*10e-3));
x = uo - xl;
x = mean((z .* x) / f); % horizontal displacement
%D{3} = sprintf('%.8f',(x*10e-3));
min_disP = min(disP); %minimum disparity
max_disP = max(disP); % maximum disparity
if isinteger(min_disP/16) && isinteger(max_disP/16)
    mini = min_disP;
    maxi = max_disP;
else % values are not interger
    smin = round(min_disP/16);
    smax = round(max_disP/16);
    mini = smin * 16;
    maxi = smax * 16;   
end
disparityRange = [mini maxi];
successFlag =1;
else
    disparityRange = [0 1];
    x = 0; y = 0; z = 0;
end
% Output result
stereoGeometry.EstimatedPositions = table(x,y,z,'VariableNames',{'X (mm)','Y (mm)','Z (mm)'}); 
stereoGeometry.DisparityRange = disparityRange; 
stereoGeometry.RectifiedImages = {J1,J2}; 
stereoGeometry.MatchedPoints = {matchedLeft,matchedRight};
stereoGeometry.LeftBoundingBox = BL;
stereoGeometry.RightBoundingBox = BR;
stereoGeometry.successFlag = successFlag;
end