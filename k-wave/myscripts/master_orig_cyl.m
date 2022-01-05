%% MASTER 
disp('Master file');
clearvars;
close all;
%cd /home/gdisciac/kwave/k-Wave/myscripts
cd /cs/academic/phd3/gdisciac/k-Wave/myscripts
%cd /home/gdisciac/kwave/k-Wave/myscripts/
addpath(genpath('../'))
%addpath(genpath('./'))
%addpath(genpath('../../../'))
RUN_SIMULATION  = true;   % set to false to reload previous results instead of running simulation
%%
% define a sphere for a highly scattering region
radius = 5e-3; % [m]
x_pos = 10e-3;%x/2;  
half_height = 5e-3;% [m]%depth
delta_scatter = -83;
%% 
saving_name = sprintf('CYLINDER_5MHz_scatterBKG_focus=xpos_%g__rad_%g__height_%g__deltaScat_%g.mat', x_pos,radius, half_height*2, delta_scatter );
saving_dir = ['/cs/academic/phd3/gdisciac/k-Wave/myscripts/K-wave_files',filesep,saving_name(1:end-4)];
mkdir(saving_dir); 
saving_dir = [saving_dir , filesep];

%%
simulateUS
my_US_Bmode_cyl;
my_US_Bmode_process_cyl;

my_write_US_DICOM;
