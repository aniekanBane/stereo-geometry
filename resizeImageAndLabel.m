function data = resizeImageAndLabel(data,targetSize)
sz = size(data{1}, [1 2]);
scale = targetSize./sz;
data{1} = imresize(data{1},targetSize);
boxEstimate=round(data{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
data{2} = bboxresize(boxEstimate,scale);
end