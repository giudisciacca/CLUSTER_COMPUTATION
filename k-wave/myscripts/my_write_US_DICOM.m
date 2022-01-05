%% Writes DICOM image and save scattering mask
savename = ['kwaved-',saving_name(1:end-4)];
saveas(h_harm, [saving_dir, savename,'.png'],'png');
saveas(h_harm_truth, [saving_dir, savename, 'overTRUTH','.png'],'png');
saveas(h_bmode, [saving_dir,'Bmode_', savename,'.png'],'png');
saveas(h_bmode_truth, [saving_dir,'Bmode_' ,savename, 'overTRUTH','.png'],'png');
image = imread([saving_dir,'Bmode_',savename, '.png']);
figure(6);imshow([saving_dir,'Bmode_',savename, '.png']); drawnow;
[x0, x1, y0, y1] = find_angles(image);
%roipoly(imread([savename]));


info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0 = x0; %xmin
info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1 = x1; %xmax
info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0 = y0;
info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1 = y1;


% find delta
info.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaX = (horz_axis(end) - horz_axis(1))/(x1 -x0 - 1);
info.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaY = (1000*(r(end) - r(1)))/(y1 -y0 -1);
info.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsXDirection = 4; % gives in which units physical delta x is expressed 
info.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsYDirection = 4;

info.SOPClassUID = num2str(rand *100);
savenameDICOM = ['DICOM' , savename, '.dcm'];
dicomwrite(image, [saving_dir,savenameDICOM], info,'CreateMode', 'copy');


delta = dx*1000; %to get mm
scattering_region = permute(scattering_region,[2,3,1]);
scattering_region = imresizen(scattering_region, [3, 3, 3], 'nearest'); delta = delta/3;
save([saving_dir,'ORIGINAL',savename, '.mat'], 'scattering_region', 'delta')
