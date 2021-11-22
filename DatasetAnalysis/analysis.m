path = "../TrainingData";

load(path + "/Annotations/cubeGTruth.mat");
oldSource = gTruth.DataSource.Source{1}(1:61);
newSource = path + "/Images";

unresolved = changeFilePaths(gTruth, {[oldSource newSource]});

cubeDataset = objectDetectorTrainingData(gTruth); % create training data from groudTruth
summary(cubeDataset) % display the data summary set
head(cubeDataset) % preview the first eight rows

images = imageDatastore(cubeDataset.imageFilename);
labels = boxLabelDatastore(cubeDataset(:, 2:end));
combined = combine(images, labels);

store = cell(12,1);
for k = 1:12
    data = read(combined);
    store{k} = insertShape(data{1,1},'Rectangle',data{1,2},"LineWidth",10);
end

figure;
montage(store, 'BorderSize', 10) % display images with boxes and labels
