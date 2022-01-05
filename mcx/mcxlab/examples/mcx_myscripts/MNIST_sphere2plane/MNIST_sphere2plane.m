%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATIONS OF MNIST SET WITH MCX SPHERE TO PLANE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
RUN = true;
RUN_HOM = true;
SAVE = true;
DISPLAY = false;
pause(3600 * 12);
%% geometry of internal sphere
DIM = [60, 60, 45];
SKULL2BRAIN = 7;
RAD_IN = 25;
x0 = DIM(1)/2;
y0 = DIM(2)/2;
z0 = RAD_IN + SKULL2BRAIN;
x_vec = 1:DIM(1);
y_vec = 1:DIM(2);
z_vec = 1:DIM(3);
[XX,YY,ZZ] = meshgrid(x_vec, y_vec, z_vec);
idx_sphere_in = ((XX - x0).^2 + (YY - y0).^2 + (ZZ - z0).^2 <= RAD_IN^2) .* (ZZ <= z0);

%fine_grid
fine_step = 0.1;
[Xx, Yy, Zz] = meshgrid(1:fine_step:DIM(1), 1:fine_step:DIM(2), 1:fine_step:DIM(3));
idx_sphere_in_fine = ((Xx - x0).^2 + (Yy - y0).^2 + (Zz - z0).^2 <= RAD_IN^2 ) .* (Zz <= z0);
%% optical parameters
outside_0 = [0 0 1 1];
liquido_1 = [1e-7  1e-1   0    1.37];
brain_2 = [1e-2  1    0    1.37];
quantum = 0.1;
surface_3n = [1 1 0 1.37;];
%% Setting general parameters
mnist_size = 1000;
cfg.nphoton = 2e8;
cfg.tstart=0;
cfg.tend=6e-9;
cfg.tstep=1e-11;
cfg.isreflect = 1;
cfg.unitinmm = 1;
cfg.srcpos=[1 1 1];
cfg.srctype='planar';
cfg.srcparam1 = [DIM(1),0,0,0];
cfg.srcparam2 = [0,DIM(2),0,0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;



%% homogeous case. Just layer of scattering
% just once
if RUN_HOM 
    cfg.vol= ones(DIM);
    cfg.vol = cfg.vol + idx_sphere_in;
    prov_diff_vol = cat(3, zeros([DIM(1), DIM(2), 1]), diff(cfg.vol - 1,1,3)) .* (ZZ <= z0) ; 
    cfg.vol = uint8(cfg.vol);
    cfg.prop=[outside_0;    
              liquido_1;  % liquido  1 
              brain_2;]; % inside 2  2
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
    
    hom_layer_photons(:,:) = project_sphere2plane(XX, x0, YY, y0, ZZ, SKULL2BRAIN, RAD_IN, prov_diff_vol, whole_homCW(:,:,:) );
end


%% Inhomogenous case
cfg.vol = [];
cfg.prop = [];
MNIST = loadMNISTImages('t10k-images.idx3-ubyte');
MNIST = round(reshape(MNIST,[28,28,10000]), abs(log10(quantum)) );

mnist_layer=MNIST(:,:,1);


if RUN     
    cfg.vol=ones(60,60,45); %liquido
    cfg.vol = cfg.vol + idx_sphere_in;
    cfg.prop = zeros([3 + 1/quantum,4]);
    cfg.prop=[outside_0;    
              liquido_1;  % liquido  1 
              brain_2;]; % inside 2  2
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
    mnist_sliceCW = zeros([DIM(1),DIM(2), mnist_size]);
    mnist_sliceCW_5mm = mnist_sliceCW;
    mnist_sliceCW_15ts = mnist_sliceCW;
    whole_mnistCW = zeros(DIM(1),DIM(2),DIM(3), mnist_size); 
    mnist_layer_abs = zeros(DIM(1),DIM(2), DIM(3), mnist_size);
    mnist_planar_rep = zeros(60,60,mnist_size);
    flatted_mnist = zeros(DIM(1), DIM(2), mnist_size);
    flatted_mnist_abs = flatted_mnist;
    mnist_layer_photons = flatted_mnist;
    for i_mnist = 1:mnist_size
        fprintf('ITERATION:%g \n', i_mnist)
        mnist_layer = MNIST(:,:, i_mnist);
        % reset volume
        cfg.vol=ones(60,60,45); %liquido
        cfg.vol = cfg.vol + idx_sphere_in; % brain
        % from layer to sphere
        prov_layer = zeros(DIM(1), DIM(2), 1);
        prov_layer(16:43,16:43) = 1/quantum * mnist_layer;
        prov_layer = repmat(prov_layer, [1,1,DIM(3)]);
        prov_diff_vol = cat(3, zeros([DIM(1), DIM(2), 1]), diff(cfg.vol - 1,1,3)) .* (ZZ <= z0) ; % diff of binary: 1s and 0s 
        prov_layer = prov_diff_vol .* prov_layer;
        cfg.vol = cfg.vol + prov_layer;
        cfg.vol = uint8(cfg.vol);
        mnist_layer_abs(:,:,i_mnist) = 0;
        for i_abs = 1:max(cfg.vol(:))
            mnist_layer_abs(:,:,:,i_mnist) = mnist_layer_abs(:,:,:, i_mnist) + (cfg.vol(:,:,:) == (i_abs) ) * cfg.prop(i_abs+1,1);
        end
        
        % from sphere to "deformed" layer
        prov_flatted = project_sphere2plane(XX,x0,YY,y0,ZZ, SKULL2BRAIN, RAD_IN, prov_diff_vol, cfg.vol - 1 );
%         X_coor = XX;
%         Y_coor = YY;
%         Z_coor = ZZ;
%         xy_dist2centre = sqrt( (X_coor - x0).^2 + (Y_coor - y0).^2 );
%         xy_dist2centre_flat = xy_dist2centre(:,:,1);
%         z_dist2centre = abs(Z_coor - SKULL2BRAIN).* prov_diff_vol;
%         xyz_dist2centre =  sum( sqrt(xy_dist2centre.^2 +z_dist2centre.^2),3);
%         prov_layer_flat = sum(double(cfg.vol - 1).* prov_diff_vol ,3);
%         flatted_mnist(:,:, i_mnist) = zeros(DIM(1), DIM(2));
%         prov_flatted = zeros(DIM(1), DIM(2));
%         dist_x0 = 1;
%         dist_y0 = 1;
%         
%         for i_x = 1:DIM(1)
%             for i_y = 1:DIM(2)
%                 if (xy_dist2centre_flat(i_x, i_y) <= RAD_IN)
%                     dist = round( RAD_IN * xy_dist2centre_flat(i_x, i_y) ./ sqrt(RAD_IN.^2 - xy_dist2centre_flat(i_x, i_y).^2) );
%                     theta = atan( (i_y - y0 ) / (i_x - x0 ));
%                     if (i_x - x0) < 0
%                         theta = theta + pi;
%                     end
%                     dist_x = x0 + ceil(dist * cos(theta));                    
%                     dist_y = y0 + ceil(dist * sin(theta));
%                     if dist_x < DIM(1) && dist_x>0 && dist_y < DIM(2) && dist_y >0
%                         if dist_x0 < DIM(1) && dist_x0>0 && dist_y0 <DIM(2) && dist_y0 >0 && abs(dist_x0 - dist_x) < 0.5 * DIM(1) && abs(dist_y0 - dist_y) < 0.5 * DIM(2)
%                            prov_flatted(min(dist_x0, dist_x) :(max(dist_x0,dist_x)+1) , min(dist_y0, dist_y):(max(dist_y0,dist_y)+1) ) ...
%                                = ...
%                                prov_layer_flat(i_x, i_y); 
%                         else    
%                            prov_flatted(dist_x, dist_y) = prov_layer_flat(i_x, i_y);
%                         end
%                     end
%                     dist_x0 = dist_x;
%                     dist_y0 = dist_y;
%                 end
%             end
%         end
 
        % get absorption from indexes
        for i_abs = 1:max(cfg.vol(:))
            flatted_mnist_abs(:,:,i_mnist) = flatted_mnist_abs(:,:, i_mnist) + (prov_flatted(:,:) == (i_abs) ) * cfg.prop(i_abs+1,1);
        end
        flatted_mnist(:,:, i_mnist) = prov_flatted;
        %figure; imagesc(prov_flatted);

        % simulation
        kit_cfg = cfg;    
        evalc('[flux_mnist] = mcxlab(kit_cfg);');
        time_elapsed_1iter = toc;
        disp(round(time_elapsed_1iter));
        %[flux_mnist] = mcxlab(kit_cfg);
        flux_mnist_data = flux_mnist.data;
        
        whole_mnistCW(:,:,:, i_mnist) = sum(flux_mnist_data,4); 
        % project photons from sphere to 2d layer
        mnist_layer_photons(:,:, i_mnist) = project_sphere2plane(XX, x0, YY, y0, ZZ, SKULL2BRAIN, RAD_IN, prov_diff_vol, whole_mnistCW(:,:,:,i_mnist) );
        
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
dirname = sprintf('/scratch0/NOT_BACKED_UP/gdisciac/mcx_files/DOCM_sphere2plane_ph%g_radius_%g_dist2scalp%g/',cfg.nphoton, RAD_IN, SKULL2BRAIN);
mkdir(dirname);
save([dirname,'DOCM_sphere2plane_Specifications'], 'cfg')
save([dirname,'DOCM_sphere2plane_OriginalABS_layer'], 'mnist_layer_abs','flatted_mnist','flatted_mnist_abs', 'mnist_layer_photons')
save([dirname,'DOCM_sphere2plane_DiffusedMNIST_3D'], 'whole_mnistCW', 'whole_homCW')
save([dirname,'DOCM_sphere2plane_DiffusedMNIST'], 'mnist_sliceCW');
save([dirname,'DOCM_sphere2plane_DiffusedMNIST_5mm'], 'mnist_sliceCW_5mm');
save([dirname,'DOCM_sphere2plane_DiffusedMNIST_15ts'], 'mnist_sliceCW_15ts');
save([dirname,'DOCM_sphere2plane_OriginalMNIST'], 'MNIST');
save([dirname,'DOCM_sphere2plane_DiffusedHomogeneous'], 'hom_sliceCW', 'hom_layer_photons');
save([dirname,'DOCM_sphere2plane_DiffusedHomogeneous_5mm'], 'hom_sliceCW_5mm');
save([dirname,'DOCM_sphere2plane_DiffusedHomogeneous_15ts'], 'hom_sliceCW_15ts');
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