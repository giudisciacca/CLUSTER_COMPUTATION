%% script to generate one US-Bmode image

addpath(genpath('path/to/kwave'))
lname = 'path/to/acoustic/ground/truth/';
saving_dir  = ['path/to/saving/folder'];

type = 'TEST_VICTRE_PARADIGM_'
i = 1;

% load file
load([lname,'.mat'])
    
voidMass = permute(voidMass,[3,1,2]);
structkwave = permute(structkwave,[3,1,2]);

% simulate
[out_us, scan_lines_raw, sound_speed_map] = simulateUS_victre(voidMass, [dimVox,dimVox,dimVox]*10^-3, [type,num2str(i)], structkwave);

% normalise   
out_us = (out_us - prctile(out_us(:),0))/( prctile(out_us(:),100) - prctile(out_us(:),0));
out_us(out_us>1) = 1;
out_us(out_us<0) = 0;
    
  
save([saving_dir,filesep,type,num2str(i),'VICTRE.mat'],'out_us','scan_lines_raw','sound_speed_map');






