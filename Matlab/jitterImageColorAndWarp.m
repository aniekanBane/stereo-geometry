function out = jitterImageColorAndWarp(data)
% Unpack original data.
I = data{1};
C = data{2};
sz = size(I);
labels = data{3};

if numel(sz)==3 && sz(3) ==3
% Apply random color jitter.
I = jitterColorHSV(I,"Brightness",0.2,"Contrast",0.4,"Hue",0,"Saturation",0.2);
end

% Randomly flip and scale and rotate image.
tform = randomAffine2d("XReflection",true,'Scale',[0.8 1],'Rotation',[-15 15]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');

% Apply transformation to image and bounding box labels.
augmentedImage = imwarp(I,tform,"OutputView",rout);
[augmentedBoxes, indices] = bboxwarp(C,tform,rout,'OverlapThreshold',0.25);
augmentedLabels = labels(indices);

% Return augmented data.
out = {augmentedImage,augmentedBoxes,augmentedLabels};
end