
load('pixelgTruth.mat');
% create a pixel label datastore to store the groundTruth data
Dataset = pixelLabelImageDatastore(gTruth); 
imds = imageDatastore(Dataset.Images); % extract images
labelims = Dataset.PixelLabelData; % extract label images
classes = Dataset.ClassNames; % extract categories
labelIDs = gTruth.LabelDefinitions.PixelLabelID; % label id for each class
pxds = pixelLabelDatastore(labelims,classes,labelIDs);

% Analyse and view data
I = readimage(imds,97); % read the 97th image
C = readimage(pxds,97); % read the 97th label data
P = uint8(C); 
B = labeloverlay(I,C); % overlay label on the image
Mask = C == 'cube'; % logical mask to display only the object of interest
figure;
montage({I,P,B,Mask},'BorderSize',4,'Size',[1 4])

% Analyze pixel label data
tbl = countEachLabel(pxds);
% view data distribution
frequency = tbl.PixelCount/sum(tbl.PixelCount);
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

% Prepare datasets
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(Dataset,labelIDs);

% Create DNN for semantic segmentation
inputSize = [256 256 3]; % network image input size
numClasses = numel(classes); % target number of classes
% deeplab model with resnet50 weights
lgraph = deeplabv3plusLayers(inputSize,numClasses,"resnet50");

% Balance the classes using median frequency weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classweights = median(imageFreq) ./ imageFreq;
% update network layer with weights
pxLayer = pixelClassificationLayer("Name","labels","Classes",tbl.Name,"ClassWeights",classweights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);
analyzeNetwork(lgraph)

% Apply data augmentation to the training data to give more variety to the dataset
trainingData = combine(imdsTrain,pxdsTrain);
valData = combine(imdsVal,pxdsVal);
testData = combine(imdsTest,pxdsTest);
augmentedTrainingData = transform(trainingData,@jitterImageColorAndWarp);
rgb = cell(4,1); % cell to store images
% read and display augmentation
for k = 1:4
    data = read(augmentedTrainingData);
    rgb{k} = labeloverlay(data{1,1},data{1,2});
end
montage(rgb,'BorderSize',8)

% Preprocess datasets for training
PPtraining = transform(augmentedTrainingData,@(data)resizeImageandLabel(data,inputSize(1:2)));
PPval = transform(valData,@(data)resizeImageandLabel(data,inputSize(1:2)));

% set training parameters
opts = trainingOptions("sgdm",...
                       'MaxEpochs',30,"MiniBatchSize",2,...
                       'ValidationData',PPval,...
                       'InitialLearnRate',1e-3,...
                       'L2Regularization',0.0005,...
                       'ValidationFrequency',10,...
                       "Shuffle","every-epoch",...
                       "Plots","training-progress",...
                       'ExecutionEnvironment',"auto",...
                       'VerboseFrequency',2);
DCNN = trainNetwork(PPtraining,lgraph,opts);

% perform a quick test with the network
QI = read(imdsTest); % read one of the test images 
QC = semanticseg(QI,DCNN); % apply network
QB = labeloverlay(QI,QC);
QM = QC == 'cube'; % logical mask
montage({QI,QB,QM},'BorderSize',10,"Size",[1 3]); title('Predicted Segmentation')
% view the actual groundTuth data and compare with above
expR = read(pxdsTest);
exp = uint8(expR);
acc = uint8(QC);
QCE = labeloverlay(QP,expR);
QME = expR == 'cube';
montage({QP,QCE,QME},'BorderSize',10,"Size",[1 3]); title('Expected Segmentation')
% view regions of overlap
imshowpair(acc, exp); title('gTruth Comparison')
% segmentation performance
iou = jaccard(QC,expR);
table(classes,iou)

% Evaluate the model using the test set
TestI = transform(imdsTest,@(data)imresize(data,inputSize(1:2)));
pxdsPred = semanticseg(TestI,DCNN,'MiniBatchSize',5,'WriteLocation',temp);
TestL = transform(pxdsTest,@(data)imresize(data{1},inputSize(1:2),'Method',"nearest"));
metrics = evaluateSemanticSegmentation(pxdsPred,TestL);
normConfMat = metrics.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classes,classes,100*normConfMat);
h.XLabel = 'Predicted class';
h.YLabel = 'True class';
h.Title = 'Normalised Confusion Matrix (%)';
imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('image Mean IoU')
metrics.ClassMetrics