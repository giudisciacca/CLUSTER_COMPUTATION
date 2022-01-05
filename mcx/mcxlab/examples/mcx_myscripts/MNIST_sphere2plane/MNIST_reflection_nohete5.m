%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATIONS OF MNIST SET WITH MCX SPHERE TO PLANE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
scattering_rota = [0.5,1,2,3,4];%, 2, 3, 4, 0.5];
for count_liquido_scatter=1:numel(scattering_rota)
    clearvars -except count_liquido_scatter scattering_rota
    midname = 'reflection_flat_hete';
    RUN = true;
    RUN_HOM = true;
    SAVE = true;
    DISPLAY = false;
    pause(3600 * 0);
    %% geometry of internal sphere
    DIM = [60, 60, 30];
    cubX_in =[2,43];
    cubY_in = [16,43];
    cubZ_in = [2,50];
    SKULL2BRAIN = 10;
    RAD_IN = 1e10;
    x0 = DIM(1)/2;
    y0 = DIM(2)/2;
    z0 = RAD_IN + SKULL2BRAIN;
    x_vec = 1:DIM(1);
    y_vec = 1:DIM(2);
    z_vec = 1:DIM(3);
    [XX,YY,ZZ] = meshgrid(x_vec, y_vec, z_vec);
    idx_sphere_in = ((XX - x0).^2 + (YY - y0).^2 + (ZZ - z0).^2 <= RAD_IN^2) .* (ZZ <= z0);
    
    %% pseudoheterogeneities
    scat_hete.perc_noise = 0.4;
    scat_hete.l_corr = 15;
    
    %% optical parameters
    outside_0 = [0 0 1 1];
    liquido_1 = [1e-7  scattering_rota(count_liquido_scatter)    0    1.37];
    start_liquido = 1;
    end_liquido = 100;
    
    brain_2 = [1e-1  5    0    1.37];
    start_brain = end_liquido + 1;
    index_of_startMNIST = start_brain + 1;
    quantum = 0.1;
    surface_3n = [1 2 0 1.37;];
    %% Setting general parameters
    mnist_size = 1000;
    mnist_init = 1;
    cfg.nphoton = 5e6;%8;
    cfg.tstart=0;
    cfg.tend=6e-9;
    cfg.tstep=1e-10;
    cfg.isreflect = 0;
    cfg.isrefint = 0;
    cfg.isspecular = 0;
    cfg.unitinmm = 1;

% % %     [srcx,srcy,srcz] = meshgrid([15,20,25,30,25,40,45], [15,20,25,30,25,40,45], 0);
% % %     srcpos = [srcx(:), srcy(:), srcz(:)];
% % %     N_src = size(srcpos,1);
% % %     cfg.srctype='gaussian';
% % %     cfg.srcparam1 = [5,0,0,0];
% % %     cfg.srcdir=[0 0 1];
    srcpos = [1, 1, 0];
    N_src = size(srcpos,1);
    cfg.srctype='planar';% 'pattern %'gaussian';%
    cfg.srcparam1 = [DIM(1),0,0,0];%;[3 0 0 0]; % 
    cfg.srcparam2 = [0,DIM(2),0,0];
    cfg.srcdir=[0 0 1];

    cfg.gpuid=1;
    cfg.autopilot=1;
    %cfg.seed=99999;


    %% Inhomogenous case
    cfg.vol = [];
    cfg.prop = [];
    MNIST = loadMNISTImages('t10k-images.idx3-ubyte');
    MNIST = round(reshape(MNIST,[28,28,10000]), abs(log10(quantum)) );
    MNIST = padarray( MNIST(:,:,mnist_init:mnist_init + mnist_size - 1), [2,2],0, 'both');

    prov_MNIST = zeros(DIM(1),DIM(2), mnist_size);
    prov_MNIST(DIM(1)/2 - 15: DIM(1)/2 + 16,DIM(2)/2 - 15: DIM(2)/2 + 16, :) = MNIST;
    MNIST = prov_MNIST;
    
    rng('default');fixed_rand = randi(10, mnist_size, 2) - 5;
    for i_tr = 1:mnist_size
        fl1 = randi(2)-1;
        fl2 = randi(2)-1;
        perm12 = randi(2)-1;
        MNIST(:,:,i_tr) = circshift(MNIST(:,:,i_tr) ,fixed_rand(i_tr,:));
        if fl1 == 1
            MNIST(:,:,i_tr) = flip( MNIST(:,:,i_tr), 1);
        end
        if fl2 == 1
            MNIST(:,:,i_tr) = flip( MNIST(:,:,i_tr), 2);
        end
        if perm12 == 1
            MNIST(:,:,i_tr) = MNIST(:,:,i_tr)';
        end        
    end
    
    %MNIST(:,:,1) = 0;
    %MNIST(13:15,13:15, 1) = 1;
    mnist_layer=MNIST(:,:,1);


    if RUN     
        % calculate the flux distribution with the given config

        mnist_layer_abs = zeros(DIM(1),DIM(2), mnist_size);
        mnist_layer_photons = zeros(DIM(1), DIM(2), N_src, mnist_size);
        hom_layer_photons = zeros(DIM(1), DIM(2), N_src, mnist_size);
        whole_homCW = zeros(DIM(1),DIM(2), cubZ_in(1), N_src, mnist_size);
        whole_mnistCW = zeros(DIM(1),DIM(2), cubZ_in(1), N_src, mnist_size); 
        tic;
        for i_mnist = 1:mnist_size

            cfg.vol=ones(DIM); %liquido
        
            cfg.vol(idx_sphere_in == 1) = 0;
            [cfg.vol, scattering_quants] = mcx_add_correlated_noise(cfg.vol, scattering_rota(count_liquido_scatter),...
                                                start_liquido, end_liquido, scat_hete.l_corr,scat_hete.perc_noise);
            
            liquido_het = zeros(numel(start_liquido:end_liquido),4);
            liquido_het(:,1) = liquido_1(1); 
            liquido_het(:,2) = scattering_quants;
            liquido_het(:,3) = liquido_1(3);
            liquido_het(:,4) = liquido_1(4);
            
            fprintf('Pseudo heterogenities set:%g \n', i_mnist)            
            cfg.vol(idx_sphere_in == 1) = start_brain;
            
            
            cfg_hom = cfg;            
            cfg_hom.prop = zeros([index_of_startMNIST,4]);                
            cfg.prop = zeros([index_of_startMNIST + 1/quantum,4]);                       
            cfg_hom.prop=[outside_0;
                         liquido_het; %pseudo het
                          brain_2;]; % inside 2  2

            cfg_hom.vol = uint8(cfg.vol);
            % get quantized values to index
            q_index = 1;
            quantized = zeros(numel(quantum:quantum:1) ,4);
            for q_count = quantum:quantum:1
                    prov_surface = [ q_count * surface_3n(1), surface_3n(2:4)];
                    quantized(q_index,:) =  prov_surface;  % MNIST layer quantized
                    q_index = q_index  + 1;
            end
            
            cfg.prop = [cfg_hom.prop; 
                        quantized];


            fprintf('ITERATION:%g \n', i_mnist)
            mnist_layer = MNIST(:,:, i_mnist);
            % reset volume

            prov_layer = zeros(DIM(1), DIM(2));
            prov_layer = mnist_layer;
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
                %cfg.srcpattern = zeros(60,60,1);
                %cfg.srcpattern = mnist_layer;
                cfg_hom.srcpos = srcpos(i_src, :);
                %cfg_hom.srcpattern  = cfg.srcpattern;
                disp(srcpos(i_src,:));

                 fprintf('ITERATION_HOMOGENEOUS:%g \n', i_mnist)
                 evalc('[flux_hom] = mcxlab(cfg_hom);');
                 flux_hom_data = flux_hom.data;
                 whole_homCW0(:,:,:) = sum(flux_hom_data(:,:,:,:),4); 
                 hom_layer_photons(:,:,i_src, i_mnist) = simple_project_sph2pl(idx_sphere_in, whole_homCW0, SKULL2BRAIN, RAD_IN);
                 whole_homCW(:,:,1:cubZ_in(1),i_src, i_mnist) = sum(flux_hom_data(:,:,1:cubZ_in(1),:),4);    

 
                 fprintf('ITERATION_HETEROGENEOUS:%g \n', i_mnist)
                 kit_cfg = cfg;
                 evalc('[flux_mnist] = mcxlab(kit_cfg);');
                 flux_mnist_data = flux_mnist.data;
                 %flux_mnist_data = flux_hom.data;%%%%%%%%%%
                 whole_mnistCW0(:,:,:) = sum(flux_mnist_data(:,:,:,:),4); 
                 mnist_layer_photons(:,:,i_src, i_mnist) = simple_project_sph2pl(idx_sphere_in, whole_mnistCW0, SKULL2BRAIN, RAD_IN);
                 whole_mnistCW(:,:,1:cubZ_in(1),i_src, i_mnist) = sum(flux_mnist_data(:,:,1:cubZ_in(1),:),4);    
             end

            whole_homCW = whole_homCW(:,:,1:cubZ_in(1), :,:);
            whole_mnistCW = whole_mnistCW(:,:,1:cubZ_in(1), :,:);
            time_elapsed_1iter = toc;
            disp(round(time_elapsed_1iter));  
        end

        time_elapsed_2 = toc;
    end
    MNIST_ex = MNIST(:,:, 1:mnist_size);

    disp('saving')
    dirname = sprintf('/home/gdisciac/DOCM_%s_ph%g_radius_%g_dist2scalp%g_liqMus%g/', midname, cfg.nphoton, RAD_IN, SKULL2BRAIN, liquido_1(2));
    mnist_spec = sprintf('mnistFrom%gto%g', mnist_init, mnist_init+mnist_size);
    mkdir(dirname);
    save([dirname,'DOCM_', midname, mnist_spec,'_Specifications'], 'cfg','cfg_hom', 'srcpos', '-v7.3')
    save([dirname,'DOCM_', midname, mnist_spec,'_DiffusedMNIST_3D'], 'whole_mnistCW', 'whole_homCW', 'mnist_layer_abs','mnist_layer_photons','hom_layer_photons','-v7.3')
    disp('saved')

end
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
