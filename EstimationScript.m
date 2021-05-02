% load data
load('Detector2.mat');
load('Calibration/stereoParams.mat');
% load images
IL = imread("left/tr_l_0.jpg");
IR = imread("project_data_23rd_April/right/tr_r_0.jpg");
% Estimate position
b = EstimateStereoGeometry(IL,IR,stereoParams,detector);
disp(b.EstimatedPositions)
% Visualize results
rectL = b.RectifiedImages{1};
rectR = b.RectifiedImages{2};
% draw bboxes
annotateIL = insertShape(rectL,"rectangle",b.LeftBoundingBox,"LineWidth",8);
annotateIR = insertShape(rectR,"rectangle",b.RightBoundingBox,"LineWidth",8);
h1=figure;
imshowpair(annotateIL,annotateIR,'montage')
title('Detected Object')
% view corresponding matches
ML = rgb2gray(rectL);
MR = rgb2gray(rectR);
h2 =figure;
showMatchedFeatures(ML,MR,b.MatchedPoints{1},b.MatchedPoints{2},"montage");
title('Matching points')
legend('Leftpts','rightpts')

