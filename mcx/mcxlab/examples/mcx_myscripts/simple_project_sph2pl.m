function [out_projected] = simple_project_sph2pl(idx, TP,sk2br, rad)
% project quantity TP of same size of idx (of 0s and 1s) on the internal
% if TP is a 3D matrix, then out_projected is the projection of the layer
% identified by idx of a 2D matrix.
% if TP is a 2D matrix, then out_projected is the projection of TP on the
% 3D border
TP = squeeze(TP);
nd_tp = ndims(TP);
DIM = size(idx);
z0 = sk2br + rad;
[~,~,ZZ] = meshgrid(1:DIM(1),1:DIM(2), 1:DIM(3));
border  = cat(3, zeros([DIM(1), DIM(2), 1]), diff(idx,1,3)) .* (ZZ <= z0);
if nd_tp == 3 
    out_projected(:,:) = sum(border .* TP,3);
elseif nd_tp == 2
   TP3D = repmat(TP,[1,1,DIM(3)]);
   out_projected(:,:,:)= border .* TP3D;
end
end