addpath(genpath('../../'))
%clear
% load file
% it = i0;
% if it >=1 && it<801
% load /home/gdisciac/OpticalDatasetPrior/PRIOR4DOT_FEMsph.mat
% % load example mat
% load Test_StandardnewBORNscaLargeIncl_1_REC.mat
% 
% saving_dir  = ['/home/gdisciac/',filesep,'sphUS'];
% mkdir(saving_dir)
% type = 'train';
% elseif it >= 801 && it<1001 
% load /home/gdisciac/OpticalDatasetPrior/PRIOR4DOT_FEMsph_t.mat
% % load example mat
% load Test_StandardnewBORNscaLargeIncl_1_REC.mat
% saving_dir  = ['/home/gdisciac/',filesep,'sphUS'];
% mkdir(saving_dir)
% type = 'test';
% i0 = i0 - 800;
% 
% elseif it >=1001
% load /home/gdisciac/OpticalDatasetPrior/PRIOR4DOT_FEMsph_v_t.mat
% load Test_StandardnewBORNscaLargeIncl_1_REC.mat
% saving_dir  = ['/home/gdisciac/',filesep,'sphUS'];
% mkdir(saving_dir)
% type = 'valid';
% i0 = i0 - 1000;
%end

%lname = '/scratch0/NOT_BACKED_UP/gdisciac/VICTRE/multifolder/old/VICTRE_';
lname = '/home/gdisciac/k-wave/myscripts/scripts_victre/VICTRE_';
saving_predir =  '/home/gdisciac/k-wave/myscripts/scripts_victre'
saving_dir  = [saving_predir,filesep,'US_VICTRE'];
mkdir(saving_dir)
if ~exist('i0','var')
i0 = 1;
end
iend = i0;
%if ~exist([saving_dir,filesep,type,num2str(i0),'cyl.mat'],'file')
type = 'TEST_VICRE'
for i = i0:iend
    load([lname,num2str(i),'.mat'])
    %prior3D_in = permute(squeeze(prior(i,:,:,:)),[3,1,2]);
    voidMass = permute(voidMass,[3,1,2]);
    structkwave = permute(structkwave,[3,1,2]);
    % simulate
    [out_us, scan_lines_raw] = simulateUS_victre(voidMass, [dimVox,dimVox,dimVox]*10^-3, [type,num2str(i)], structkwave);
    % select prior
    out_us = (out_us - prctile(out_us(:),0))/( prctile(out_us(:),100) - prctile(out_us(:),0));
    out_us(out_us>1) = 1;
    out_us(out_us<0) = 0;
    
    %USim(i,:,:) = out_us;
    close all
    save([saving_dir,filesep,type,num2str(i),'VICTRE.mat'],'out_us','scan_lines_raw')
    % append
end

