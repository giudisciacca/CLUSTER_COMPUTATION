%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATIONS OF MNIST SET WITH MCX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
RUN = true;
SAVE = true;
DISPLAY = false;
%% Setting general parameters
cfg.nphoton = 1e7;
cfg.tstart=0;
cfg.tend=6e-9;
cfg.tstep=1e-11;
cfg.detpos=[33 30 1 2];
cfg.isreflect = 1;
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
% just once
if RUN 
    cfg.vol=uint8( ones(60,60,50));
    cfg.vol = uint8(cfg.vol);
    cfg.vol(5:55, 5:55, 10:45) = 2;
    cfg.prop=[0 0 1 1; % 0 
        0.005 0.2 0 1.37;
        0.0025 1.2  0    1.37;]; %3
    % calculate the flux distribution with the given config
    tic;
    hom_cfg = cfg;
    [flux_hom] = mcxlab(hom_cfg);
    flux_hom_data = flux_hom.data;
    whole_homCW = sum(flux_hom_data,4);
    %select slice
    hom_slice = flux_hom_data(:,:,1,1:end);
    hom_slice_5mm = flux_hom_data(:,:,5,1:end);
    hom_slice_15ts = flux_hom_data(:,:,1,15:end);
    % integrate in time
    hom_sliceCW = double(squeeze(sum(sum(hom_slice,3),4)));
    hom_sliceCW_5mm = double(squeeze(sum(sum(hom_slice_5mm,3),4)));
    hom_sliceCW_15ts = double(squeeze(sum(sum(hom_slice_15ts,3),4)));
   
    time_elapsed_1 = toc;
end


%% Inhomogenous case
cfg.vol = [];
cfg.prop = [];
quantum = 0.1;
MNIST = loadMNISTImages('t10k-images.idx3-ubyte');
MNIST = round(reshape(MNIST,[28,28,10000]), abs(log10(quantum)) );

mnist_layer=MNIST(:,:,1);
mnist_size = 1;

if RUN     
    cfg.vol=uint8(ones(60,60,45)); %liquido
    cfg.vol(5:55, 5:55, 10:43) = 2;
    cfg.vol = uint8(cfg.vol);
    cfg.prop = zeros([3 + 1/quantum,4]);
    cfg.prop=[   0    0      1    1;    
                1e-7  1e-1   0    1.37;  % liquido  
                1e-4  1    0    1.37;]; % inside 2
    q_index = 1;
    quantized = zeros(numel(quantum:quantum:1) ,4);
    for q_count = quantum:quantum:1 
            quantized(q_index,:) =  [q_count*1     1      0    1.37;];  % MNIST layer quantized
            q_index = q_index  + 1;
    end
    cfg.prop = [cfg.prop; quantized];
    % calculate the flux distribution with the given config
    tic;
    mnist_sliceCW = zeros([60,60, mnist_size]);
    mnist_sliceCW_5mm = mnist_sliceCW;
    mnist_sliceCW_15ts = mnist_sliceCW;
    whole_mnistCW = zeros(60,60,45, mnist_size); 
    mnist_layer_abs = zeros(60,60, mnist_size);
    for i_mnist = 1:mnist_size
        fprintf('ITERATION:%g \n', i_mnist)
        mnist_layer = MNIST(:,:, i_mnist);
        cfg.vol(16:43,16:43,10) = 2; %reset vol
        cfg.vol(16:43,16:43,10) = cfg.vol(16:43,16:43,10) + uint8( 1/quantum * mnist_layer );
        
        mnist_layer_abs(:,:,i_mnist) = 0;
        for i_abs = 1:max(cfg.vol(:))
            mnist_layer_abs(:,:,i_mnist) = mnist_layer_abs(:,:,i_mnist) + (cfg.vol(:,:,10) == (i_abs) ) * cfg.prop(i_abs+1,1);
        end
        kit_cfg = cfg;    
        evalc('[flux_mnist] = mcxlab(kit_cfg);');
        time_elapsed_1iter = toc;
        disp(round(time_elapsed_1iter));
        %[flux_mnist] = mcxlab(kit_cfg);
        flux_mnist_data = flux_mnist.data;
        
        whole_mnistCW(:,:,:, i_mnist) = sum(flux_mnist_data,4); 
        
        %select slice
        mnist_slice = flux_mnist_data(:,:,1,1:end);
        mnist_slice_5mm = flux_mnist_data(:,:,5,1:end);
        mnist_slice_15ts = flux_mnist_data(:,:,1,15:end);

        % integrate in time
        mnist_sliceCW(:,:, i_mnist) = double(squeeze(sum(sum(mnist_slice,3),4)));
        mnist_sliceCW_5mm(:,:, i_mnist) = double(squeeze(sum(sum(mnist_slice_5mm,3),4)));
        mnist_sliceCW_15ts(:,:, i_mnist) = double(squeeze(sum(sum(mnist_slice_15ts,3),4)));
        
%         figure(1); imagesc(log(mnist_sliceCW(:,:,i_mnist))), axis image;
%         figure(12); imagesc(log(mnist_sliceCW_5mm(:,:,i_mnist))), axis image;
%         figure(13); imagesc(log(mnist_sliceCW_15ts(:,:,i_mnist))), axis image;
%         figure(2); imagesc(mnist_sliceCW(15:45,15:45,i_mnist) - hom_sliceCW(15:45,15:45) ./ hom_sliceCW(15:45,15:45) ), axis image;
%         figure(22); imagesc(mnist_sliceCW_5mm(15:45,15:45,i_mnist) - hom_sliceCW_5mm(15:45,15:45) ./ hom_sliceCW_5mm(15:45,15:45) ), axis image;
%         figure(23); imagesc(mnist_sliceCW_15ts(15:45,15:45,i_mnist) - hom_sliceCW_15ts(15:45,15:45) ./ hom_sliceCW_15ts(15:45,15:45) ), axis image;
    end
    time_elapsed_2 = toc;
end
MNIST_ex = MNIST(:,:, 1:mnist_size);

disp('saving')
dirname = sprintf('DOCM_ph%g/',cfg.nphoton);
mkdir(dirname);
save([dirname,'DOCM_Specifications'], 'cfg')
save([dirname,'DOCM_OriginalABS_layer'], 'mnist_layer_abs')
save([dirname,'DOCM_DiffusedMNIST_3D'], 'whole_mnistCW', 'whole_homCW')
save([dirname,'DOCM_DiffusedMNIST'], 'mnist_sliceCW');
save([dirname,'DOCM_DiffusedMNIST_5mm'], 'mnist_sliceCW_5mm');
save([dirname,'DOCM_DiffusedMNIST_15ts'], 'mnist_sliceCW_15ts');
save([dirname,'DOCM_OriginalMNIST'], 'MNIST');
save([dirname,'DOCM_DiffusedHomogeneous'], 'hom_sliceCW');
save([dirname,'DOCM_DiffusedHomogeneous_5mm'], 'hom_sliceCW_5mm');
save([dirname,'DOCM_DiffusedHomogeneous_15ts'], 'hom_sliceCW_15ts');
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
    figure;imagesc(( 1+  (kit_sliceCW(10:50, 10:50) - hom_sliceCW(10:50,10:50))./(hom_sliceCW(10:50,10:50)+1)) );
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