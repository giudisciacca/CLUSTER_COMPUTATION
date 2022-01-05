% =========================================================================
% PROCESS THE RESULTS
% =========================================================================
FACT_scale = 0.7;
% -----------------------------
% Remove Input Signal
% -----------------------------

% create a window to set the first part of each scan line to zero to remove
% interference from the input signal
scan_line_win = getWin(kgrid.Nt * 2, 'Tukey', 'Param', 0.05).';
scan_line_win = [zeros(1, length(input_signal) * 2), scan_line_win(1:end/2 - length(input_signal) * 2)];

% apply the window to each of the scan lines
scan_lines = bsxfun(@times, scan_line_win, scan_lines);

% store a copy of the middle scan line to illustrate the effects of each
% processing step
scan_line_example(1, :) = scan_lines(end/2, :);

% -----------------------------
% Time Gain Compensation
% -----------------------------

% % % % % % % % % % % create radius variable assuming that t0 corresponds
% to the middle of the input signal % % % % % % % % %
t0 = length(input_signal) * kgrid.dt / 2; % % % % % % % % % 
r = c0 * ((1:length(kgrid.t_array)) * kgrid.dt / 2 - t0);    % [m] 

%%%%%%%%%%%
%%%%%%%%%%
%create time gain compensation function based on attenuation value, transmit frequency, and round trip distance 
tgc_alpha =  0.01;       % [dB/(MHz cm)] % % % %
tgc = exp(2 * tgc_alpha * tone_burst_freq * 1e-6 * r * 100); %

% apply the time gain compensation to each of the scan lines 
scan_lines = bsxfun(@times, tgc,scan_lines);

%%%%%%%%%%
%%%%%%%%%%
% store a copy of the middle scan line to illustrate the effects of each
% processing step
scan_line_example(2, :) = scan_lines(end/2, :);

% -----------------------------
% Frequency Filtering
% -----------------------------

% filter the scan lines using both the transmit frequency and the second
% harmonic
scan_lines_fund = gaussianFilter(scan_lines, 1/kgrid.dt, tone_burst_freq, 100, true);
%set(gca, 'XLim', [0, 6]);
scan_lines_harm = gaussianFilter(scan_lines, 1/kgrid.dt, 2 * tone_burst_freq, 30, true);
%set(gca, 'XLim', [0, 6]);
close all

% store a copy of the middle scan line to illustrate the effects of each
% processing step
scan_line_example(3, :) = scan_lines_fund(end/2, :);

% -----------------------------
% Envelope Detection
% -----------------------------

% envelope detection
scan_lines_fund = envelopeDetection(scan_lines_fund);
scan_lines_harm = envelopeDetection(scan_lines_harm);

% store a copy of the middle scan line to illustrate the effects of each
% processing step
scan_line_example(4, :) = scan_lines_fund(end/2, :);

% -----------------------------
% Log Compression
% -----------------------------

% normalised log compression
compression_ratio = 40;
scan_lines_fund = logCompression(scan_lines_fund, compression_ratio, true);
scan_lines_harm = logCompression(scan_lines_harm, compression_ratio, true);

% store a copy of the middle scan line to illustrate the effects of each
% processing step
scan_line_example(5, :) = scan_lines_fund(end/2, :);

% -----------------------------
% Scan Conversion
% -----------------------------

% upsample the image using linear interpolation
scale_factor = 2;
scan_lines_fund = interp2(1:kgrid.Nt, (1:number_scan_lines).', scan_lines_fund, 1:kgrid.Nt, (1:1/scale_factor:number_scan_lines).');
scan_lines_harm = interp2(1:kgrid.Nt, (1:number_scan_lines).', scan_lines_harm, 1:kgrid.Nt, (1:1/scale_factor:number_scan_lines).');

% =========================================================================
% VISUALISATION
% =========================================================================

% plot the medium, truncated to the field of view
h_phantom = figure;
%imagesc((0:number_scan_lines * transducer.element_width - 1) * dy * 1e3, 0:Nx_tot-1* dx * 1e3, sound_speed_map(:, 1 + Ny/2:end - Ny/2, Nz/2));
imagesc(sound_speed_map(:, :, round(Nz/2)));
axis image;
colormap(gray);
set(gca, 'YLim', [0, Nx_tot]);
title('Scattering Phantom');
xlabel('Horizontal Position [mm]');
ylabel('Depth [mm]');

% plot the processing steps
% figure;
% stackedPlot(kgrid.t_array * 1e6, {'1. Beamformed Signal', '2. Time Gain Compensation', '3. Frequency Filtering', '4. Envelope Detection', '5. Log Compression'}, scan_line_example);
% xlabel('Time [\mus]');
% set(gca, 'XLim', [5, t_end * 1e6]);

% plot the processed b-mode ultrasound image
h_bmode = figure;
horz_axis = (0:length(scan_lines_fund(:, 1)) - 1) * transducer.element_width * dy / scale_factor * 1e3;
imagesc(horz_axis, r * 1e3, (scan_lines_fund.'));
axis image;
scan_lines_fund_T = scan_lines_fund.';scan_lines_fund_T = scan_lines_fund_T(find(r>=2/1000 & r < FACT_scale *x),: );
limits = [min(scan_lines_fund_T(:)) 0.9*max(scan_lines_fund_T(:))];
caxis(limits);
colormap(gray);
set(gca, 'YLim', [0, x * 1000]); 
title('B-mode Image');
xlabel('Horizontal Position [mm]');
ylabel('Depth [mm]');

% plot the processed harmonic ultrasound BMODE image WITH TRUTH
h_bmode_truth = figure;
scan_lines_fund_truth = repmat(double(mat2gray(scan_lines_fund)).', 1,1,3) * 127*0.2;
[xx_truth, yy_truth] = meshgrid(horz_axis, r * 1000);
d_h_ax = (horz_axis(end) - horz_axis(1))/numel(horz_axis);
truth = double((xx_truth - (horz_axis(end) + 0.5 + d_h_ax)/2).^2 + (yy_truth - x_pos * 1000).^2 <= (radius*1000)^2); 
scan_lines_fund_truth(:,:,1) = scan_lines_fund_truth(:,:,1) .* (~truth) + 5 * max(reshape(scan_lines_fund_truth(:,:,1) .* truth, [numel(truth),1])) .* (truth);
imagesc(horz_axis, r * 1e3, scan_lines_fund_truth(1:end,:,:));
axis image;
scan_lines_fund_truth = permute(scan_lines_fund_truth, [2,1,3]);
scan_lines_fund_truth = scan_lines_fund_truth(:,find(r >= 5/1000 & r < FACT_scale * x),1);
limits = [min(scan_lines_fund_truth(:)) ,max(scan_lines_fund_truth(:))];
caxis(limits)
set(gca, 'YLim', [0, x*1000]);
title('B-mode Image with Simulated Truth');
xlabel('Horizontal Position [mm]');
ylabel('Depth [mm]');

% plot the processed harmonic ultrasound image
h_harm = figure;
imagesc(horz_axis, r * 1e3, scan_lines_harm.');
axis image;
scan_lines_harm_T = scan_lines_harm.';scan_lines_harm_T = scan_lines_harm_T(find(r >= 2/1000 & r <  FACT_scale *x), :);
limits = [min(scan_lines_harm_T(:)) max(scan_lines_harm_T(:))];
caxis(limits);
colormap(gray);
set(gca, 'YLim', [0, x*1000]);
title('Harmonic Image');
xlabel('Horizontal Position [mm]');
ylabel('Depth [mm]');




% plot the processed harmonic ultrasound image WITH TRUTH
h_harm_truth = figure;
scan_lines_harm_truth = repmat(double(mat2gray(scan_lines_harm)).', 1,1,3);
[xx_truth, yy_truth] = meshgrid(horz_axis, r * 1000);
d_h_ax = (horz_axis(end) - horz_axis(1))/numel(horz_axis);
truth = double(abs(xx_truth - (horz_axis(end) + 0.5+d_h_ax)/2) <= half_height & abs(yy_truth - x_pos * 1000) <= (radius*1000)); 
scan_lines_harm_truth(:,:,1) = scan_lines_harm_truth(:,:,1) + (0.40 * max(scan_lines_harm_truth(:)))*truth;
imagesc(horz_axis, r * 1e3, scan_lines_harm_truth);
axis image;
scan_lines_harm_T = scan_lines_harm.';scan_lines_harm_T = scan_lines_harm_T(find(r >= 2/1000 & r < FACT_scale * x), :);
limits = [min(scan_lines_harm_T(:)) max(scan_lines_harm_T(:))];
caxis(limits);
set(gca, 'YLim', [0, x*1000]);
title('Harmonic Image with Simulated Truth');
xlabel('Horizontal Position [mm]');
ylabel('Depth [mm]');
% =========================================================================
% VISUALISATION OF SIMULATION LAYOUT
% =========================================================================

% % uncomment to generate a voxel plot of the simulation layout
% 
% % physical properties of the transducer
if exist('transducer_plot', 'file') == 1
    clear transducer_plot
end

 transducer_plot.number_elements = number_active_elements + number_scan_lines - 1;
 transducer_plot.element_width = 2;
 transducer_plot.element_length = 24;

 transducer_plot.element_spacing = 0;
 transducer_plot.radius = inf;
% 
% % transducer position
transducer_plot.position = round([1, Ny/2 - transducer_width/2, Nz/2 - transducer.element_length/2]);
% 
% % create expanded grid
 kgrid_plot = kWaveGrid(Nx_tot, dx, Ny_tot, dy, Nz, dz);
 kgrid_plot.setTime(1, 1);
% 
%% create the transducer using the defined settings
% transducer_plot = kWaveTransducer(kgrid_plot, transducer_plot);
% 
% % create voxel plot of transducer mask and 
 %voxelPlot(single(transducer_plot.active_elements_mask | scattering_region));
 %view(26, 48);