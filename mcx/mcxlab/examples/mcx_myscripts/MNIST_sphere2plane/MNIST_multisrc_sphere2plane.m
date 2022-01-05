%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATIONS OF MNIST SET WITH MCX SPHERE TO PLANE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
midname = 'MultiGaussSrc_sphere2flat_transl';
RUN = true;
RUN_HOM = true;
SAVE = true;
DISPLAY = false;
pause(3600 * 0);
%% geometry of internal sphere
DIM = [60, 60, 60];
cubX_in =[16,43];
cubY_in = [16,43];
cubZ_in = [20,50];
SKULL2BRAIN = 10;
RAD_IN = 25;
x0 = DIM(1)/2;
y0 = DIM(2)/2;
z0 = RAD_IN + SKULL2BRAIN;
x_vec = 1:DIM(1);
y_vec = 1:DIM(2);
z_vec = 1:DIM(3);
[XX,YY,ZZ] = meshgrid(x_vec, y_vec, z_vec);
idx_sphere_in = ((XX - x0).^2 + (YY - y0).^2 + (ZZ - z0).^2 <= RAD_IN^2) .* (ZZ <= z0);

%% geometry of external sphere

%% optical parameters
outside_0 = [0 0 1 1];
liquido_1 = [1e-7  6e-1   0    1.37];
brain_2 = [1e-2  1    0    1.37];
quantum = 0.1;
surface_3n = [1 1 0 1.37;];
%% Setting general parameters
mnist_size = 1500;
cfg.nphoton = 1e7;
cfg.tstart=0;
cfg.tend=6e-9;
cfg.tstep=1e-11;
cfg.isreflect = 1;
cfg.unitinmm = 1;
[srcx,srcy,srcz] = meshgrid([23,30,37], [23,30,37], 1);
srcpos = [srcx(:), srcy(:), srcz(:)];
N_src = size(srcpos,1);
cfg.srctype='gaussian';
cfg.srcparam1 = [5,0,0,0];
%cfg.srcparam2 = [0,DIM(2),0,0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;

%% homogeous case. Just layer of scattering
% just once
if RUN_HOM 
    cfg.vol= ones(DIM);
    cfg.vol= cfg.vol+ idx_sphere_in;
    cfg.vol = uint8(cfg.vol);
    cfg.prop=[outside_0;    
              liquido_1;  % liquido  1 
              brain_2;]; % inside 2  2
    % calculate the flux distribution with the given config
    tic;
    hom_cfg = cfg;
    whole_homCW = zeros([DIM, N_src]);
    hom_layer_photons = zeros(DIM(1),DIM(2),N_src);
    for i_src = 1:N_src
        hom_cfg.srcpos = srcpos(i_src, :);
        disp(srcpos(i_src,:));
        [flux_hom] = mcxlab(hom_cfg);
        flux_hom_data = flux_hom.data;   
        %integrate in time
        whole_homCW(:,:,:,i_src) = sum(flux_hom_data,4);
        hom_layer_photons(:,:, i_src) = simple_project_sph2pl(idx_sphere_in, whole_homCW(:,:,:,i_src),SKULL2BRAIN,RAD_IN);
    end
    whole_homCW = whole_homCW(:,:,1:cubZ_in(1), :);
    time_elapsed_1 = toc;
    
    
end

%% Inhomogenous case
cfg.vol = [];
cfg.prop = [];
MNIST = loadMNISTImages('t10k-images.idx3-ubyte');
MNIST = round(reshape(MNIST,[28,28,10000]), abs(log10(quantum)) );
MNIST = MNIST(:,:,1:mnist_size);
for i_tr = 1:mnist_size
    MNIST(:,:,i_tr) = circshift(MNIST(:,:,i_tr) ,[randi(16)-8 , randi(16) - 8]);
end
mnist_layer=MNIST(:,:,1);


if RUN     
    cfg.vol=ones(DIM); %liquido
    cfg.vol= cfg.vol + idx_sphere_in;
    cfg.prop = zeros([3 + 1/quantum,4]);
    cfg.prop=[outside_0;    
              liquido_1;  % liquido  1 
              brain_2;]; % inside 2  2
    
    % get quantized values to index
    q_index = 1;
    quantized = zeros(numel(quantum:quantum:1) ,4);
    for q_count = quantum:quantum:1
            prov_surface = [ q_count * surface_3n(1), surface_3n(2:4)];
            quantized(q_index,:) =  prov_surface;  % MNIST layer quantized
            q_index = q_index  + 1;
    end
    cfg.prop = [cfg.prop; quantized];
    
    % calculate the flux distribution with the given config
    tic;
    mnist_layer_abs = zeros(DIM(1),DIM(2), mnist_size);
    mnist_layer_photons = zeros(DIM(1), DIM(2), N_src, mnist_size);
    whole_mnistCW = zeros(DIM(1),DIM(2), cubZ_in(1), N_src, mnist_size); 
    for i_mnist = 1:mnist_size
        fprintf('ITERATION:%g \n', i_mnist)
        mnist_layer = MNIST(:,:, i_mnist);
        % reset volume
        cfg.vol=ones(DIM); %liquido
        cfg.vol= cfg.vol + idx_sphere_in; % brain
        % from layer to sphere
        prov_layer = zeros(DIM(1), DIM(2));
        prov_layer(DIM(1)/2 - 13: DIM(1)/2 + 14,DIM(2)/2 - 13: DIM(2)/2 + 14) = mnist_layer;
        prov_mnist_layer = simple_project_sph2pl(idx_sphere_in, prov_layer, SKULL2BRAIN, RAD_IN);
        cfg.vol = cfg.vol + 1/quantum * prov_mnist_layer;
        cfg.vol = uint8(cfg.vol);
        
        mnist_layer_abs(:,:,i_mnist) = 0;
        vol_idx_flat = simple_project_sph2pl(idx_sphere_in, double(cfg.vol), SKULL2BRAIN, RAD_IN);
        for i_abs = 1:max(cfg.vol(:))
            mnist_layer_abs(:,:,i_mnist) = mnist_layer_abs(:,:,i_mnist) + ( vol_idx_flat  == (i_abs) ) * cfg.prop(i_abs+1,1);
        end

        % simulation
        for i_src = 1:N_src
            cfg.srcpos = srcpos(i_src, :);
            disp(srcpos(i_src,:));
            kit_cfg = cfg;
            evalc('[flux_mnist] = mcxlab(kit_cfg);');
            flux_mnist_data = flux_mnist.data;
            whole_mnistCW0(:,:,:) = sum(flux_mnist_data(:,:,:,:),4); 
            mnist_layer_photons(:,:,i_src, i_mnist) = simple_project_sph2pl(idx_sphere_in, whole_mnistCW0, SKULL2BRAIN, RAD_IN);
            whole_mnistCW(:,:,1:cubZ_in(1),i_src, i_mnist) = sum(flux_mnist_data(:,:,1:cubZ_in(1),:),4);    
        end
        
        
        whole_mnistCW = whole_mnistCW(:,:,1:cubZ_in(1), :,:);
        time_elapsed_1iter = toc;
        disp(round(time_elapsed_1iter));  
    end
    
    time_elapsed_2 = toc;
end
MNIST_ex = MNIST(:,:, 1:mnist_size);

disp('saving')
dirname = sprintf('/scratch0/NOT_BACKED_UP/gdisciac/mcx_files/DOCM_%s_ph%g_radius_%g_dist2scalp%g_liqMus%g/', midname, cfg.nphoton, RAD_IN, SKULL2BRAIN, liquido_1(2));
mkdir(dirname);
save([dirname,'DOCM_',midname ,'_Specifications'], 'cfg', 'srcpos', '-v7.3')
save([dirname,'DOCM_',midname ,'_DiffusedMNIST_3D'], 'whole_mnistCW', 'whole_homCW', 'mnist_layer_abs','mnist_layer_photons','hom_layer_photons','-v7.3')
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
        figure(1); imagesc((squeeze(flux.data(:,40,:, i)))'), axis image;
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