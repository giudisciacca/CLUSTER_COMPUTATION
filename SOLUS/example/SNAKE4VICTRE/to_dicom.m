% to_dicom
%% create
load(filename);
out_us = uint8(repmat(out_us*255,[1 1 3]));
info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0 = 1;
info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1 = size(out_us,2);
info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1 = size(out_us,1);
info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0 = 1;
info.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaX = 0.01;
info.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsYDirection = 3;
info.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsXDirection = 3;
info.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaY = 0.01;
dicomwrite(out_us,[filename(1:end-4),'.dcm'], info, 'CreateMode','Create');

