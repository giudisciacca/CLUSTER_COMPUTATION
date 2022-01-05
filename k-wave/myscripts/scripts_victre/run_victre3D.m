addpath(genpath('../../'))

%lname = '/scratch0/NOT_BACKED_UP/gdisciac/VICTRE/multifolder/old/VICTRE_';
lname = '/home/gdisciac/VICTRE/multifolder/old/VICTRE_';
saving_predir =  '/home/gdisciac/k-wave/myscripts/scripts_victre'
saving_dir  = [saving_predir,filesep,'US_VICTRE3D'];
mkdir(saving_dir)

i00 = i0;
i0 = floor(i00/5)+1;
sl0 = mod(i00/5)+1;

if ~exist('i0','var')
i0 = 1;
end
iend = i0;

if ~exist('i0','var')
sl0 = 3;
end
slend = sl0;
%if ~exist([saving_dir,filesep,type,num2str(i0),'cyl.mat'],'file')
type = 'TEST_VICRE'
for i = i0:iend
    load([lname,num2str(i),'.mat'])
    %prior3D_in = permute(squeeze(prior(i,:,:,:)),[3,1,2]);
    voidMass0 = permute(voidMass,[3,1,2]);
    structkwave0 = permute(structkwave,[3,1,2]);
    zaxis = logical(squeeze(sum(sum(voidMass(:,:,:),1),2))>=1);
    enumz = (-numel(zaxis)/2):(numel(zaxis)/2)
    shifts = enumz(zaxis);
    shifts = linspace(shifts(1), shifts(2),5);
    for slice = sl0:slend
      
	% simulate
      voidMass = circshift(voidMass0,round(shifts(slice)),3);
      structkwave = circshift(structkwave0,round(shifts(slice)),3)
      pos_slice = round(shifts(slice))*dimVox;
	[out_us, scan_lines_raw] = simulateUS_victre(voidMass, [dimVox,dimVox,dimVox]*10^-3, [type,num2str(i)], structkwave);
      % select prior
      out_us = (out_us - prctile(out_us(:),0))/( prctile(out_us(:),100) - prctile(out_us(:),0));
      out_us(out_us>1) = 1;
      out_us(out_us<0) = 0;
    
      %USim(i,:,:) = out_us;
      close all
      save([saving_dir,filesep,type,num2str(i),'_SL_',num2str(slice) ,'VICTRE.mat'],'out_us','scan_lines_raw','pos_slice')
    end
    % append
end

