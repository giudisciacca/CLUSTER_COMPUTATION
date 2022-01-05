% save([dirname,'DOCM_sphere2plane_Specifications'], 'cfg')
% save([dirname,'DOCM_sphere2plane_OriginalABS_layer'], 'mnist_layer_abs','flatted_mnist','flatted_mnist_abs', 'mnist_layer_photons')
% save([dirname,'DOCM_sphere2plane_DiffusedMNIST_3D'], 'whole_mnistCW', 'whole_homCW')
% save([dirname,'DOCM_sphere2plane_DiffusedMNIST'], 'mnist_sliceCW');
% save([dirname,'DOCM_sphere2plane_DiffusedMNIST_5mm'], 'mnist_sliceCW_5mm');
% save([dirname,'DOCM_sphere2plane_DiffusedMNIST_15ts'], 'mnist_sliceCW_15ts');
% save([dirname,'DOCM_sphere2plane_OriginalMNIST'], 'MNIST');
% save([dirname,'DOCM_sphere2plane_DiffusedHomogeneous'], 'hom_sliceCW', 'hom_layer_photons');
% save([dirname,'DOCM_sphere2plane_DiffusedHomogeneous_5mm'], 'hom_sliceCW_5mm');
% save([dirname,'DOCM_sphere2plane_DiffusedHomogeneous_15ts'], 'hom_sliceCW_15ts');
%save([dirname,'DOCM_',midname ,'_Specifications'], '-v7.3','cfg', 'srcpos')
%save([dirname,'DOCM_',midname ,'_DiffusedMNIST_3D'],'-v7.3' ,'whole_mnistCW', 'whole_homCW', 'mnist_layer_abs')

disp('saving')
dirname = sprintf('/scratch0/NOT_BACKED_UP/gdisciac/mcx_files/DOCM_%s_ph%g_radius_%g_dist2scalp%g/', midname, cfg.nphoton, RAD_IN, SKULL2BRAIN);
mkdir(dirname);
mnist_size = 1500;
whole_mnistCW = whole_mnistCW(:,:,:,:,1:mnist_size);
whole_homCW = whole_mnistCW(:,:,:,1:mnist_size);
mnist_layer_abs = mnist_layer_abs(:,:,1:mnist_size);
mnist_layer_photons = mnist_layer_photons(:,:,1:mnist_size);
hom_layer_photons = hom_layer_photons(:,:,1:mnist_size);
save([dirname,'DOCM_',midname ,'_Specifications'], 'cfg', 'srcpos', '-v7.3')
save([dirname,'DOCM_',midname ,'_DiffusedMNIST_3D'], 'whole_mnistCW', 'whole_homCW', 'mnist_layer_abs','mnist_layer_photons','hom_layer_photons','-v7.3')
disp('saved')