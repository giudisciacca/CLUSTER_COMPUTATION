% Simulating B-mode Ultrasound Images Example
%
% This example illustrates how k-Wave can be used for the simulation of
% B-mode ultrasound images (including tissue harmonic imaging) analogous to
% those produced by a modern diagnostic ultrasound scanner. It builds on
% the Defining An Ultrasound Transducer, Simulating Ultrasound Beam
% Patterns, and Using An Ultrasound Transducer As A Sensor examples. 
%
% Note, this example generates a B-mode ultrasound image from a 3D
% scattering phantom using kspaceFirstOrder3D. Compared to ray-tracing or
% Field II, this approach is very general. In particular, it accounts for
% nonlinearity, multiple scattering, power law acoustic absorption, and a
% finite beam width in the elevation direction. However, it is also
% computationally intensive. Using a modern GPU and the Parallel Computing
% Toolbox (with 'DataCast' set to 'gpuArray-single'), each scan line takes
% around 3 minutes to compute. Using a modern desktop CPU (with 'DataCast'
% set to 'single'), this increases to around 30 minutes. In this example,
% the final image is constructed using 96 scan lines. This makes the total
% computational time around 4.5 hours using a single GPU, or 2 days using a
% CPU. 
%
% To allow the simulated scan line data to be processed multiple times with
% different settings, the simulated RF data is saved to disk. This can be
% reloaded by setting RUN_SIMULATION = false within the example m-file. The
% data can also be downloaded from:
%
% http://www.k-wave.org/datasets/example_us_bmode_scan_lines.mat
%
% author: Bradley Treeby
% date: 3rd August 2011
% last update: 9th June 2017
%  
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2011-2017 Bradley Treeby

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>. 

%#ok<*UNRCH>

%clearvars;
%close all;

% simulation settings
DATA_CAST       = 'gpuArray-single';% 'single';%    % set to 'single' or 'gpuArray-single' to speed up computations


% =========================================================================
% DEFINE THE K-WAVE GRID
% =========================================================================

% set the size of the perfectly matched layer (PML)
pml_x_size = 15;%20;                % [grid points]
pml_y_size = 10;%128;                % [grid points]
pml_z_size = 10;%1;                % [grid points]

% set total number of grid points not including the PML
Nx = 2*(3^4)*(2^0) - 2 * pml_x_size; %(3^3)*(2^3) - 2 * pml_x_size;      % [grid points]
Ny = 2*(3^4)*(2^0) - 2 * pml_y_size;      % [grid points]
Nz = 2*(3^4)*(2^0) - 2 * pml_z_size;      % [grid points]

% set desired grid size in the x-direction not including the PML
x = 30e-3;                      % [m]
y = 30e-3;
z = 30e-3;
% calculate the spacing between the grid points
dx = x / Nx;                    % [m]
dy = y / Ny;                        % [m]
dz = z / Nz;                        % [m]

% create the k-space grid
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% =========================================================================
% DEFINE THE MEDIUM PARAMETERS
% =========================================================================

% define the properties of the propagation medium
c0 = 1000;                      % [m/s]
rho0 = 1500;              % [kg/m^3]
% create the time array
t_end = (Nx * dx) * 2.2 / c0;   % [s]
kgrid.makeTime(c0, [], t_end);

% =========================================================================
% DEFINE THE INPUT SIGNAL
% =========================================================================

% define properties of the input signal
source_strength = 1e6;          % [Pa]
tone_burst_freq = 5e6;        % [Hz]
tone_burst_cycles = 1;

% create the input signal using toneBurst 
input_signal = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);

% scale the source magnitude by the source_strength divided by the
% impedance (the source is assigned to the particle velocity)
input_signal = (source_strength ./ (c0 * rho0)) .* input_signal;

% =========================================================================
% DEFINE THE ULTRASOUND TRANSDUCER
% =========================================================================

% physical properties of the transducer
transducer.number_elements = 36;  	% total number of transducer elements
transducer.element_width = ceil(0.0002 / dy);       % width of each element [grid points]
transducer.element_length = ceil(0.004 / dz);  	% length of each element [grid points]
transducer.element_spacing = 0;  	% spacing (kerf  width) between the elements [grid points]
transducer.radius = inf;            % radius of curvature of the transducer [m]

% calculate the width of the transducer in grid points
transducer_width = transducer.number_elements * transducer.element_width ...
    + (transducer.number_elements - 1) * transducer.element_spacing;

% use this to position the transducer in the middle of the computational grid
transducer.position = round([1, Ny/2 - transducer_width/2, Nz/2 - transducer.element_length/2]);

% properties used to derive the beamforming delays
transducer.sound_speed = c0;                    % sound speed [m/s]
transducer.focus_distance = 13e-3;              % focus distance [m]
transducer.elevation_focus_distance = 12e-3;    % focus distance in the elevation plane [m]
transducer.steering_angle = 0;                  % steering angle [degrees]

% apodization
transducer.transmit_apodization = 'Hanning';    
transducer.receive_apodization = 'Rectangular';

% define the transducer elements that are currently active
number_active_elements =52;
transducer.active_elements = ones(transducer.number_elements, 1);

% append input signal used to drive the transducer
transducer.input_signal = input_signal;

% create the transducer using the defined settings
transducer = kWaveTransducer(kgrid, transducer);

% print out transducer properties
transducer.properties;

% =========================================================================
% DEFINE THE MEDIUM PROPERTIES
% =========================================================================

% define a large image size to move across
number_scan_lines = 120;% - 2 * number_active_elements;
Nx_tot = Nx;
Ny_tot = Ny + number_scan_lines * transducer.element_width;
Nz_tot = Nz;

% define a random distribution of scatterers for the medium
background_map_mean = 1;
background_map_std = 0.1;
background_map = background_map_mean + background_map_std * randn([Nx_tot, Ny_tot, Nz_tot]);

% define a random distribution of scatterers for the highly scattering
% region
scattering_map = randn([Nx_tot, Ny_tot, Nz_tot]);
sigma_scatter = 0.0015;
scattering_c0 = (c0 + delta_scatter) * (1 + sigma_scatter * scattering_map);
bound_sigmau = (c0 + delta_scatter) * (1 + 6*sigma_scatter);
bound_sigmad = (c0 + delta_scatter) * (1 - 6*sigma_scatter);
scattering_c0(scattering_c0 > bound_sigmau) = bound_sigmau; %[m/s]
scattering_c0(scattering_c0 < bound_sigmad) = bound_sigmad;
scattering_rho0 = scattering_c0 / (c0 / 1000);  % keep impedance constant
%scattering_rho0 = (c0 - abs(3* delta_scatter) * (1 + sigma_scatter * scattering_map))/ (c0 / 1000);
% define properties
sound_speed_map = c0 * ones(Nx_tot, Ny_tot, Nz_tot) .* background_map;
density_map = rho0 * ones(Nx_tot, Ny_tot, Nz_tot) .* background_map;

z_pos = z/2;    % perpendicular to transducer
y_pos = 0.5 * (Ny + 2*pml_y_size) * dy;%40e-3 ;    % [m]%transducer
scattering_region = makeCyl(Nx_tot, Ny_tot, Nz_tot, round(x_pos/dx), round(Ny_tot/2), round(z_pos/dz), round(radius/dx),round(half_height/dx));

% assign region
sound_speed_map(scattering_region == 1) =   scattering_c0(scattering_region == 1) ;
density_map(scattering_region == 1) = scattering_rho0(scattering_region == 1);

% heterogeneous attenuation
alpha_coeff_map = 3*scattering_region+0.01*(~scattering_region);
%medium.alpha_coeff(scattering_region == 0) = 1.2;% [dB/(MHz^y cm)]
medium.alpha_power = 0.1;
medium.BonA = 2;

% =========================================================================
% RUN THE SIMULATION
% =========================================================================

% preallocate the storage
scan_lines = zeros(number_scan_lines, kgrid.Nt);

% set the input settings
input_args = {...
    'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size, pml_z_size], 'PMLAlpha', [2,2,2]...
    'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false, 'DataPath', '/cs/academic/phd3/gdisciac/k-Wave/binaries/'};

% run the simulation if set to true, otherwise, load previous results from
% disk
if RUN_SIMULATION

    % set medium position
    medium_position = 1;
    
    % loop through the scan lines
    for scan_line_index = 1:number_scan_lines
        
        % update the command line status
        disp('');
        disp(['Computing scan line ' num2str(scan_line_index) ' of ' num2str(number_scan_lines)]);

        % load the current section of the medium
        medium.sound_speed = sound_speed_map(:, medium_position:medium_position + Ny - 1, :);
        medium.density = density_map(:, medium_position:medium_position + Ny - 1, :);
        medium.alpha_coeff = alpha_coeff_map(:, medium_position:medium_position + Ny - 1, :);
        %voxelPlot(medium.density)
        %drawnow;
        % run the simulation
        sensor_data = kspaceFirstOrder3DG(kgrid, medium, transducer, transducer, input_args{:});

        % extract the scan line from the sensor data
        scan_lines(scan_line_index, :) = transducer.scan_line(sensor_data);
        
        % update medium position
        medium_position = medium_position + transducer.element_width;
        close all
    end

    % save the scan lines to disk
    save([saving_dir, saving_name])
    
else
    
    % load the scan lines from disk
    load([saving_dir, saving_name]);
end

%my_US_Bmode_process;
