%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATIONS OF KITTIES WITH MCX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
RUN = true;
SAVE = true;
DISPLAY = false;
%% Setting general parameters
cfg.nphoton = 1e9;
cfg.tstart=0;
cfg.tend=7e-9;
cfg.tstep=1e-11;
cfg.detpos=[33 30 1 2];
cfg.isreflect = 0;
cfg.unitinmm = 1;
cfg.srcpos=[1 1 1];
cfg.srctype='planar';
cfg.srcparam1 = [60,0,0,0];
cfg.srcparam2 = [0,60,0,0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;

%% homogeous case. Just layer of scattering
if RUN 
    cfg.vol=uint8( ones(60,60,45));
    cfg.vol = uint8(cfg.vol);
    cfg.vol(15:44, 15:44, 10:44) = 2;
    cfg.prop=[0 0 1 1; % 0 
        1e-7  1e-1  0 1.37;
        1e-4 2.5  0    1.37;]; %3
    % calculate the flux distribution with the given config
    tic;
    hom_cfg = cfg;
    [flux_hom] = mcxlab(hom_cfg);
    flux_hom_data = flux_hom.data;
    time_elapsed_1 = toc;
end


%% Inhomogenous case
cfg.vol = [];
cfg.prop = [];
load kitty_30x30
kitty_layer = kitty_res;
if RUN     
    cfg.vol=uint8(ones(60,60,45)); %liquido
    cfg.vol(15:44, 15:44, 10:44) = 2;
    cfg.vol(15:44,15:44,10) = cfg.vol(15:44,15:44,10) + uint8(kitty_layer) * 2 ;
    cfg.vol = uint8(cfg.vol);
    cfg.prop=[ 0 0 1 1;    
        1e-7  1e-1  0   1.37;  %liquido  
        1e-6    2.5  0    1.37;   %inside 2
        0.1    1  0    1.37; % kitty 3
            ];
    % calculate the flux distribution with the given config
    tic;
    kit_cfg = cfg;
    [flux_kitty] = mcxlab(kit_cfg);
    flux_kitty_data = flux_kitty.data;   
    time_elapsed_2 = toc;
end
disp('saving')
save kitty_workspace 
disp('saved')
%% DISPLAY OUT IMAGE
if DISPLAY
    % take only slice after scattering layer
    hom_slice = flux_hom_data(15:45,15:45,6,20:end);
    kit_slice = flux_kitty_data(15:45,15:45,6,20:end);
    % integrate in time
    hom_sliceCW = double(squeeze(sum(sum(hom_slice,3),4)));
    kit_sliceCW = double(squeeze(sum(sum(kit_slice,3),4)));

    figure;imagesc(log( abs(kit_sliceCW(:, :)) +1))
    figure;imagesc(( 1+  (kit_sliceCW(:, :) - hom_sliceCW(:,:))./(hom_sliceCW(:,:)+1)) );
end
DOUBLE_DISP = 0;
if DOUBLE_DISP
        %imagesc(squeeze(log(flux.data(:,30,:,1)))-squeeze(log(flux.data(:,30,:,1))));
    flux.data = flux_hom_data;
    i_0 = flux.data(:,:,:,1); i_end =  flux.data(:,:,:,end);
    %clims = [0.001*cfg.nphoton*size(src_pos,1), 1*cfg.nphoton*size(src_pos,1)];
    for i=1:size(flux.data,4)
        figure(1); imagesc((squeeze(flux.data(:,30,:, i)))'), axis image;
        %caxis(log(clims)), 
        title(i*cfg.tstep), drawnow, pause(0.2);
        %figure(1); imagesc((squeeze(flux.data(:,30,:, i)))'), axis image,caxis((clims)), title(i*cfg.tstep), drawnow, pause(0.005);
    end

    [X,Y,Z] = meshgrid(1:size(cfg.vol,1),1:size(cfg.vol,2), 1:size(cfg.vol,3));
    ff = flux.data;
    ff = reshape(ff, [size(ff,1)*size(ff,2)*size(ff,3)], size(ff,4));
    detplot = ( (X  - cfg.detpos(1)).^2 + (Y  - cfg.detpos(2)).^2 <= cfg.detpos(4)^2) .* (Z == cfg.detpos(3)); 
    detplot = detplot(:);sumdet = sum(detplot);
    for i=1:size(ff,2)
        out(i) = sum( ff(:,i) .* detplot / sumdet);
    end
    figure(2);plot(squeeze(flux.data(cfg.detpos(1),cfg.detpos(2), cfg.detpos(3),:)))
    figure(3);plot(cfg.tstart:cfg.tstep:cfg.tend-cfg.tstep,squeeze(out))
    
    
    vol = cfg.vol;
    for i=1:size(vol,3)
        figure(1);imagesc(vol(:,:,i)), caxis([1,6]), title(i), pause()
    end
end
    
display('END OF CODE')