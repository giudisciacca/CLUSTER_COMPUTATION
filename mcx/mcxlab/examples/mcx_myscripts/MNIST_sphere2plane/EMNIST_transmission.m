%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATIONS OF MNIST SET WITH MCX SPHERE TO PLANE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except Ilaunch
scattering_rota = [0.25,0.5,1,2];%[0.1,0.3,0.5,0.7,1,3,4,5,7,10,15,20];%, 2, 0.3];
perc_rota = [0,0.2,0.5,0.8]; 
ph_rota = [1e6,1e7,1e8];
%Ilaunch = 16;
Icompare = 1;
for count_ph = 1:numel(ph_rota)
	for count_perc = 1:numel(perc_rota)
		for count_liquido_scatter=1:numel(scattering_rota)
			if Icompare == Ilaunch		
				found_ph = count_ph
				found_perc = count_perc
				found_scatter = count_liquido_scatter
			end
			Icompare = Icompare+1;
		end
	end
end
g_aniso = 0.89;
scattering_rota = [0.25,0.5,1,2]./0.89;
for count_ph = found_ph
for count_perc = found_perc
for count_liquido_scatter=found_scatter

    %clearvars -except count_liquido_scatter scattering_rota count 
    checkboard = zeros(60,60);
    checkboard(3:6:end,3:6:end)=1;
    midname = 'transmission_2021remake_equalised_';
    RUN = true;
    RUN_HOM = true;
    SAVE = true;
    %% geometry of internal sphere
    NX=60; NY = 60;NZ = 30+2;
    SKULL2BRAIN=30;
    DIM = [NX, NY, NZ];
    x_vec = 1:DIM(1);
    y_vec = 1:DIM(2);
    z_vec = 1:DIM(3);
    [XX,YY,ZZ] = meshgrid(x_vec, y_vec, z_vec);
    %idx_sphere_in = ((XX - x0).^2 + (YY - y0).^2 + (ZZ - z0).^2 <= RAD_IN^2) .* (ZZ <= z0);
    idx_sphere_in = XX*0; % no sphere
    
    %% pseudoheterogeneities
    scat_hete.perc_noise = perc_rota(count_perc); % chosen by rota
    scat_hete.l_corr = 15; % keep fixed
    
    %% optical parameters
    outside_0 = [0 0 1 1];
    liquido_1 = [1e-7  scattering_rota(count_liquido_scatter)    0    1.37];
    start_liquido = 1;
    end_liquido = 21;   
    brain_2 = [1e-2  1    0    1.37];
    start_brain = end_liquido + 1;
    index_of_startMNIST = start_brain + 1;
    quantum = 1;
    surface_3n = [1 1 0.89 1.37;];
    %% Setting general parameters
    cfg.autopilot=1;
    cfg.gpuid=1;
    cfg.nphoton = 5.5e5; 
    nph_TOT = ph_rota(count_ph); % chosen by rota
    cfg.nphoton = 0.55 * nph_TOT;    
    cfg.tstart=0;
    cfg.tend=6e-9;
    cfg.tstep=1e-11;
    
    cfg.isreflect = 0;
    cfg.issaveref = 1;
    cfg.unitinmm = 1;
    cfg.isnormalized = 0;
    cfg.detpos = [];
    for i = 1:NX-1
        for j = 1:NY-1
            cfg.detpos=cat(1,cfg.detpos,[i,j,2, 0.5]);
        end
    end
    cfg.maxdetphoton = 1e7;
    srcpos = [1, 1, NZ];
    N_src = 1;
    cfg.srctype= 'pattern';
    cfg.srcparam1 = [DIM(1),0,0,DIM(1)];
    cfg.srcparam2 = [0,DIM(2),0,DIM(2)];
    cfg.srcdir=[0 0 -1];
    %% MNIST handling
    mnist_size = 1200;
    NUMBER_MNIST = mnist_size;
    mnist_init = 1;
    
    MNIST_0 = load('emnist-letters.mat');
    MNIST_0 = double(MNIST_0.dataset.train.images'/255);
    MNIST_0 = round(reshape(MNIST_0,[28,28,124800]), abs(log10(quantum)) );
    MNIST_0 = round(imresize(reshape(MNIST_0,[28,28,124800]), [40,40])./ ...
        max(max(imresize(reshape(MNIST_0,[28,28,124800]), [40,40]),[],1),[],2), abs(log10(quantum)));
    
    MNIST = zeros(60,60, size(MNIST_0,3));
    MNIST(11:50,11:50,:) = MNIST_0;

    rng('default');
 
    perm_rand = randperm(124800, NUMBER_MNIST);
    MNIST = MNIST(:,:,perm_rand);

    fixed_rand = randi(10, mnist_size, 2) - 5;
    for i_tr = 1:mnist_size
        MNIST(:,:,i_tr) = circshift(MNIST(:,:,i_tr) ,fixed_rand(i_tr,:));
    end
    
    mnist_layer_abs = zeros(DIM(1),DIM(2), mnist_size);
    %% Inhomogenous case
    cfg.vol = [];
    cfg.prop = [];
  
    mnist_layer=MNIST(:,:,1);


    if RUN     
        % calculate the flux distribution with the given config

        mnist_layer_photons = zeros(DIM(1), DIM(2), N_src, mnist_size);
        hom_layer_photons = zeros(DIM(1), DIM(2), N_src, mnist_size);
        whole_homCW = zeros(DIM(1),DIM(2), NZ, N_src, mnist_size);
        whole_controlCW = zeros(DIM(1),DIM(2), NZ, N_src, mnist_size); 
        tic;
        for i_mnist = 1:mnist_size
            % handle volume
            cfg.vol=ones(DIM); %liquido 
            
            [cfg.vol, scattering_quants] = mcx_add_correlated_noise(cfg.vol, scattering_rota(count_liquido_scatter),...
                                                start_liquido, end_liquido, scat_hete.l_corr,scat_hete.perc_noise);
            

            fprintf('Pseudo heterogenities set:%g \n', i_mnist)
            cfg.vol(idx_sphere_in == 1) = start_brain;
            cfg.vol(:,:,end) = 0;
            cfg.vol(:,:,1) = 0;
               
            %cfg.prop = zeros([index_of_startMNIST + 1/quantum,4]);                       
            % get quantized values to index
            q_index = 1;
            for q_count = 1:numel(scattering_quants)
                    prov_surface = [ surface_3n(1), scattering_quants(q_count) *1, surface_3n(3:4)];
                    quantized(q_index,:) =  prov_surface; 
                    q_index = q_index  + 1;
            end
            
            cfg.prop =[outside_0,  
                        quantized];
                    
            fprintf('ITERATION:%g \n', i_mnist)
            mnist_layer = MNIST(:,:, i_mnist);


           % simulation
            for i_src = 1:N_src % number of sources
                cfg.srcpos = srcpos(i_src, :);
                cfg.srcpattern = zeros(60,60,1);
                cfg.srcpattern = mnist_layer;
                
                cfg.srcpos = srcpos(i_src, :);
                cfg.srcpattern  = cfg.srcpattern;
                disp(srcpos(i_src,:));

                fprintf('ITERATION_MAIN:%g \n', i_mnist)
                
                flux_hom_data = 0;
                nphdet = 0;
                while nphdet <  nph_TOT
                    cfg.seed = randi(10000);
                    evalc('[flux_hom, det] = mcxlab(cfg);');
                    flux_hom_data = flux_hom_data + flux_hom.data;
                    nphdet = nphdet + numel(det.detid)*((2/sqrt(pi))^2);
                    
                end
                
                whole_homCW0(:,:,:) = sum(flux_hom_data(:,:,:,:),4); 
                hom_layer_photons(:,:,i_src, i_mnist) = simple_project_sph2pl(idx_sphere_in, whole_homCW0, NZ, 1e20);
                whole_homCW(:,:,1:NZ,i_src, i_mnist) = sum(flux_hom_data(:,:,1:NZ,:),4);    


                 fprintf('ITERATION_CONTROL:%g \n', i_mnist)
                 cfg.srcpattern = zeros(60,60);
                 cfg.srcpattern = checkboard;
                 kit_cfg = cfg;
                 
                flux_control_data = 0;
                nphdet = 0;
                while nphdet <  nph_TOT
                    kit_cfg.seed = randi(10000);
                    evalc('[flux_control,det] = mcxlab(kit_cfg);');
                    flux_control_data = flux_control_data + flux_control.data;
                    nphdet = nphdet + numel(det.detid)*((2/sqrt(pi))^2);                   
                end
                 whole_controlCW0(:,:,:) = sum(flux_control_data(:,:,:,:),4);
                 control_layer_photons(:,:,i_src, i_mnist) = simple_project_sph2pl(idx_sphere_in, whole_controlCW0, NZ, 1e20);
                 whole_controlCW(:,:,1:NZ,i_src, i_mnist) = sum(flux_control_data(:,:,1:NZ,:),4);

             end

            whole_homCW = whole_homCW(:,:,1:NZ, :,:);
            whole_controlCW = whole_controlCW(:,:,1:NZ, :,:);

            time_elapsed_1iter = toc;
            disp(round(time_elapsed_1iter));  
        end

        time_elapsed_2 = toc;
    end
    MNIST_ex = MNIST(:,:, 1:mnist_size);

    disp('saving')
    dirname = sprintf('/home/gdisciac/mcx_sim/DOCM_%s_ph%g_radius_inf_dist2scalp%g_liqMus%g_var%g/',...
		midname, cfg.nphoton, SKULL2BRAIN,scattering_rota(count_liquido_scatter),perc_rota(count_perc) );
    mnist_spec = sprintf('mnistFrom%gto%g', mnist_init, mnist_init+mnist_size);
    mkdir(dirname);
    save([dirname,'DOCM_', midname, mnist_spec,'_Specifications'], 'cfg', 'srcpos', '-v7.3')
    save([dirname,'DOCM_', midname, mnist_spec,'_DiffusedMNIST_3D'],'checkboard','MNIST','whole_controlCW', 'whole_homCW', 'mnist_layer_abs','mnist_layer_photons','hom_layer_photons','-v7.3')
    disp('saved')

end
end
end


