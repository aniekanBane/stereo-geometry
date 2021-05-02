% Load data
real_z = trainingdata.pos_z;
real_y = trainingdata.pos_y;
real_x = trainingdata.pos_x;
est_z = Positionestimation.Estimated_z;
est_y = Positionestimation.Estimated_y;
est_x = Positionestimation.Estimated_x;
% plot 
figure
plot(real_z)
hold on;
plot(est_z,'.-')
title('Depth (Z)')
ylabel('Distance (m)')
xlabel('Image Pairs')
legend('Real z','Estimated z')
figure
plot(real_y)
hold on;
plot(est_y,'.-')
title('Position Y')
ylabel('Distance (m)')
xlabel('Image Pairs')
legend('Real y','Estimated y')
figure
plot(real_x)
hold on;
plot(est_x,'.-')
title('Position X')
ylabel('Distance (m)')
xlabel('Image Pairs')
legend('Real x','Estimated x')
% group
act = [real_x,real_y,real_z]; % real position
pred = [est_x,est_y,est_z]; % estimated positions

if size(act)~=size(pred)
    error('Error. Matrix sizes must agree.')
end
% remove nan values
I = ~isnan(act) & ~isnan(pred);
act = act(I); 
pred = pred(I);
% rearrange the vector
idx = size(act,1)/3;
act = {act(1:idx),act(idx+1:idx*2),act(idx*2+1:idx*3)};
pred = {pred(1:idx),pred(idx+1:idx*2),pred(idx*2+1:idx*3)};
% calculate rmse
Err =((act{1}-pred{1}).^2) + ((act{2}-pred{2}).^2) + ((act{3}-pred{3}).^2);
RMSE = sqrt(mean(Err));
figure
stem((act{1}-pred{1}) + (act{2}-pred{2}) + (act{3}-pred{3}))
ylabel('Error (m)',"FontSize",15)
xlabel('Image Pairs',"FontSize",15)
title(sprintf('RMSE = %0.7f metres',RMSE),"FontSize",15)
