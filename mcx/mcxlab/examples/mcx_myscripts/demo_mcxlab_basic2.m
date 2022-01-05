%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show the most basic usage of MCXLAB.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RUN = true;
SAVE = true;
DISPLAY = false;
if RUN 
    clear cfg cfgs
    cfg.nphoton=1e10;
    cfg.vol=uint8(ones(60,60,60));
    cfg.vol(10:20,25:35,5:20) = 2;
    cfg.vol = uint8(cfg.vol);
    cfg.srcpos=[30 30 1];
    cfg.srcdir=[0 0 1];
    cfg.gpuid=1;
    % cfg.gpuid='11'; % use two GPUs together
    cfg.autopilot=1;
    cfg.prop=[0 0 1 1;0.005 1.50 0 1.37; 0.0001 4.5 0 1.37];
    cfg.tstart=0;
    cfg.tend=3e-9;
    cfg.tstep=1e-11;
    cfg.detpos=[10 30 1 2];
    cfg.isreflect = 0;
    cfg.unitinmm = 1;
    % calculate the flux distribution with the given config
    tic;
    [flux, det] = mcxlab(cfg);
    time_elapsed = toc
    if SAVE == 1 
        save example_demo2
    end

else
    load example_demo2
end



if DISPLAY == 1
    %imagesc(squeeze(log(flux.data(:,30,:,1)))-squeeze(log(flux.data(:,30,:,1))));
    i_0 = flux.data(:,:,:,1); i_end =  flux.data(:,:,:,end);
    clims = [0.001*cfg.nphoton, 1*cfg.nphoton];
    for i=1:size(flux.data,4)
        figure(1); imagesc(log(squeeze(flux.data(:,30,:, i)))'), axis image;
        caxis(log(clims)), 
        title(i*cfg.tstep), drawnow, pause(0.01);
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
end
