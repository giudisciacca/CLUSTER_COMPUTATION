%% MASTER 
disp('Master file');
clearvars;
close all;
cd /home/gdisciac/kwave/k-Wave/myscripts
%cd /scratch0/NOT_BACKED_UP/gdisciac/k-Wave/myscripts
%cd /home/gdisciac/kwave/k-Wave/myscripts/
addpath(genpath('../../'))
%addpath(genpath('./'))
%addpath(genpath('../../../'))
RUN_SIMULATION  = true;   % set to false to reload previous results instead of running simulation
%%
% define a sphere for a highly scattering region
radius = 7e-3;     % [m]
x_pos = 21e-3;%x/2;    % [m]%depth
delta_scatter = 10;
%% 
saving_name = sprintf('PROVA_BALL=xpos_%g__rad_%g__deltaScat_%g.mat', x_pos,radius, delta_scatter );
saving_dir = ['/home/gdisciac/kwave/K-wave_files',filesep,saving_name(1:end-4)];
mkdir(saving_dir); 
saving_dir = [saving_dir , filesep];

%%
my_US_Bmode;
a = 'success';
save('output_s', 'a')
%my_US_Bmode_process;
%my_write_US_DICOM;
