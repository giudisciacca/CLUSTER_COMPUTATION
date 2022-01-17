close all

NUMBER_MNIST = 1200;
MNIST = zeros([60,60,NUMBER_MNIST]); 

%MNIST_0 = loadMNISTImages('t10k-images.idx3-ubyte');
%MNIST_0 = reshape(MNIST_0,[28,28,10000]);
quantum = 1;%1/1000; 
MNIST_0 = load(['/home/gdisciac/mcx/mcxlab/examples/mcx_myscripts/MNIST_set/emnist-letters.mat']);
MNIST_0 = double(MNIST_0.dataset.train.images'/255);
MNIST_0 = round(imresize(reshape(MNIST_0,[28,28,124800]), [40,40])./ ...
        max(max(imresize(reshape(MNIST_0,[28,28,124800]), [40,40]),[],1),[],2), abs(log10(quantum)));

MNIST = zeros(60,60, size(MNIST_0,3));
MNIST(11:50,11:50,:) = MNIST_0(:,:,:);

rng('default');
perm_rand = randperm(124800, NUMBER_MNIST);
MNIST = MNIST(:,:,perm_rand);

fixed_rand = randi(10, NUMBER_MNIST, 2) - 5;
for i_tr = 1:NUMBER_MNIST
    MNIST(:,:,i_tr) = circshift(MNIST(:,:,i_tr) ,fixed_rand(i_tr,:));
end

    
N1 = 60;
N2 = 60;
N3 = 30;
numberK = 1;
train_num = 800;
t_sam = 1000;
v_sam = 1200;
totsam = 1200;

meshfact = 1;
lavg_sca = [0.5,1,2];%,4];
lavg_var = [0,0.20,0.50,0.80];
%% decide sim param

Icompare = 1;
for count_perc = 1:numel(lavg_var)
    for count_liquido_scatter=1:numel(lavg_sca)
        if Icompare == Ilaunch		
            found_perc = lavg_var(count_perc);
            found_scatter = lavg_sca(count_liquido_scatter);
        end
        Icompare = Icompare+1;
    end
end


%%

for avg_sca = found_scatter
for avg_var = found_perc
datasets_name = ['FEM_2021_remake_avgsca',num2str(avg_sca),'_avgvar_',num2str(avg_var),'_'];
%% define mesh and basis
[vtx, idx, etp]  = mkslab([0,0,0; N1-1,N2-1,N3-1],[meshfact*N1,meshfact*N2,meshfact*N3]);
global mesh
mesh = toastMesh(vtx,idx,etp);
global basis
basis = toastBasis(mesh,[N1,N2,N3]);

mua = 0;
ref_bkg = 1.4;
nnd = mesh.NodeCount;
mua = ones(nnd,1) * mua;
ref = ones(nnd,1) * ref_bkg;


kappa = 1./(3*generateRandomKappa(numberK, N3,avg_sca, avg_var));

%% ACTUAL_DISTR
iii = 1;

input = zeros(N1,N2, NUMBER_MNIST*numberK);
diffused = zeros(N1,N2, NUMBER_MNIST*numberK);
diffused_nohete = zeros(N1, N2, NUMBER_MNIST);
checkboard = zeros(N1,N2,N3);
checkboard(3:6:end,3:6:end,1) = 1;

for i=1:NUMBER_MNIST
    disp(i)
    %kappa(:,:,i) = generateRandomKappa(numberK);
    for k = 1:numberK
        input(:,:,iii) = MNIST(:,:,i);
        input3D = zeros(N1,N2,N3);
        [xx, yy] = meshgrid(1:60, 1:60);
        %input0 = exp(-((xx - 30).^2 + (yy - 30).^2)/ 2 );
        %input3D(29:31,29:31,1) = 1;
        input3D(:,:,1) = input(:,:,iii);
        prov_diffused(:,:) = diffuseMNIST3(kappa(:,:,:,k), input3D); 
        
        
        diffused(:,:,iii) = prov_diffused(:,:);
        if i == 1
            prov_diffused_control(:,:) = diffuseMNIST3(kappa(:,:,:,k), checkboard);
            diffused_control = prov_diffused_control(:,:);
            diffused_control_nohete  = prov_diffused_control(:,:);%diffuseMNIST3(1/(3*(avg_sca)*ones(N1,N2,N3)), checkboard);
        end
        iii = iii+1;
    end 
    prov_diffused_nohete(:,:) = zeros(60,60);% diffuseMNIST3(1/(3*(avg_sca)*ones(N1,N2,N3)), input3D);
    diffused_nohete(:,:,i) =  prov_diffused_nohete(:,:);
    
end
diffused_control = repmat(diffused_control,[1,1,NUMBER_MNIST]);
diffused_control_nohete = repmat(diffused_control_nohete,[1,1,NUMBER_MNIST]);

imagesInput = input;
imagesDiff = reshape(squeeze(diffused(:,:,:)), [1,60,60,numberK*NUMBER_MNIST]);
imagesControl = reshape(squeeze(diffused_control(:,:,:)), [1,60,60,numberK*NUMBER_MNIST]);
%save(['OriginalDiffTOTRANDOM0.2perc', '.mat'],'-v7.3','imagesInput','imagesDiff')
%%
if avg_var ~=0

for i=1:numberK
    idx = i:numberK:numberK*NUMBER_MNIST;
    imagesInput_ = input(:,:,idx);
    imagesDiff_ = reshape(squeeze(diffused(:,:, idx)), [1,60,60,NUMBER_MNIST]);
    imagesControl_ = reshape(squeeze(diffused_control(:,:, idx)), [1,60,60,NUMBER_MNIST]);
    
    i_str = num2str(i);
    kappa_set = kappa(:,:,:,i);
    % train
    imagesInput = imagesInput_(:,:,1:train_num);
    imagesDiff = imagesDiff_(:,:,:,1:train_num);
    imageControl = imagesControl_(:,:,:,train_num +1 : t_sam);
    save([datasets_name,i_str, '.mat'],'-v7.3','imagesInput','imagesDiff','imagesControl', 'kappa_set', 'diffused_nohete','diffused_control_nohete')
    % test
    imagesInput = imagesInput_(:,:,train_num +1 : t_sam);
    imagesDiff = imagesDiff_(:,:,:,train_num +1 : t_sam);
    imageControl = imagesControl_(:,:,:,train_num +1 : t_sam);
    save([datasets_name,i_str, '_t.mat'],'-v7.3','imagesInput','imagesDiff','imagesControl' ,'kappa_set', 'diffused_nohete','diffused_control_nohete')
    %valid
    imagesInput = imagesInput_(:,:, t_sam+1:v_sam);
    imagesDiff = imagesDiff_(:,:, t_sam+1:v_sam);
    imageControl = imagesControl_(:,:, t_sam+1:v_sam);
    save([datasets_name,i_str, '_v_t.mat'],'-v7.3','imagesInput','imagesDiff','imagesControl' ,'kappa_set', 'diffused_nohete','diffused_control_nohete')
    
       
    if i == 1
        imagesInput_ = input(:,:,idx);
        imagesDiff_ = reshape(squeeze(diffused_nohete(:,:,:)), [1,60,60,NUMBER_MNIST]);
        imagesControl_ = reshape(squeeze(diffused_control_nohete(:,:,:)), [1,60,60,NUMBER_MNIST]);
        i_str = num2str(i);
        
        imagesInput = imagesInput_(:,:,1:train_num);
        imagesDiff = imagesDiff_(:,:,:,1:train_num);
        imagesControl = imagesControl_(:,:,:,1:train_num);
        save([datasets_name,'_nohete', '.mat'],'-v7.3','imagesInput','imagesDiff','imagesControl', 'kappa_set', 'diffused_nohete')
    
        imagesInput = imagesInput_(:,:,train_num +1 : t_sam);
        imagesDiff = imagesDiff_(:,:,:,train_num +1 : t_sam);
        imagesControl = imagesControl_(:,:,:,train_num +1:t_sam);
        save([datasets_name,'_nohete', '_t.mat'],'-v7.3','imagesInput','imagesDiff','imagesControl' ,'kappa_set', 'diffused_nohete')    
   
        imagesInput = imagesInput_(:,:,t_sam:v_sam);
        imagesDiff = imagesDiff_(:,:,:,t_sam:v_sam);
        imagesControl = imagesControl_(:,:,:,t_sam:v_sam);
        save([datasets_name,'_nohete', '_v_t.mat'],'-v7.3','imagesInput','imagesControl','imagesDiff', 'kappa_set', 'diffused_nohete') 
    end
end

end
%% sets together
% n_tog =  [4];
% 
% for i=1:numel(n_tog)
%     idx = 1:numberK:numberK*NUMBER_MNIST;
%     for ii = 2:n_tog(i)
%         idx0 = ii:numberK:numberK*NUMBER_MNIST;
%         idx = cat(1,idx,idx0);
%     end
%     idx = sort(idx);
%     imagesInput_ = input(:,:,idx);
%     imagesDiff_ = reshape(squeeze(diffused(:,:, idx)), [1,60,60,NUMBER_MNIST*n_tog(i)]);i_str = num2str(n_tog(i));
%     kappa_set = kappa(:,:,:,i);
%     
%     imagesInput = imagesInput_(:,:,1:train_num);
%     imagesDiff = imagesDiff_(:,:,:,1:train_num);
%     save(['OriginalDiff_from1to',i_str, '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')
% 
%     imagesInput = imagesInput_(:,:,train_num +1 : tot_sam);
%     imagesDiff = imagesDiff_(:,:,:,train_num +1 : tot_sam);
%     save(['OriginalDiff_from1to',i_str, '_t.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')
% end



 %% combinations
% n_comb =  [4];
% for comb=1:numel(n_comb)
%     for ex = 1:2
%         linear_coeff = rand(1,1,1,n_comb(comb));
%         linear_coeff = linear_coeff/sum(linear_coeff); 
%         kappa_comb = sum(linear_coeff.* kappa(:,:,:,1:n_comb(comb)),4);
%         input = zeros(60,60, NUMBER_MNIST);
%         diffused = zeros(60,60, NUMBER_MNIST);
%         for i=1:NUMBER_MNIST
%             disp(i)            
%             input(:,:,i) = MNIST(:,:,i);
%             input3D = zeros(N1,N2,N3);
%             input3D(:,:,1) = input(:,:,i);
%             prov_diffused(:,:) = diffuseMNIST3(kappa_comb(:,:,:), input3D); 
% 
%             %prov_diffused(:,:,:) = diffuseMNIST3(kappa_comb(:,:), MNIST(:,:,i), dt, iter); 
%             diffused(:,:,i) = prov_diffused(:,:);                
%         end
%         imagesInput_ = input;
%         imagesDiff_ = reshape(squeeze(diffused(:,:,:)), [1,60,60,NUMBER_MNIST]);
%         comb_str = num2str(n_comb(comb));
%         ex_str = num2str(ex);
%         
%         imagesInput = imagesInput_(:,:,1:train_num);
%         imagesDiff = imagesDiff_(:,:,:,1:train_num);
%         save(['OriginalDiff_ex_',ex_str,'_combOf_',comb_str, '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_comb')
%  
%         imagesInput = imagesInput_(:,:,train_num+1:tot_sam);
%         imagesDiff = imagesDiff_(:,:,:,train_num+1:tot_sam);
%         save(['OriginalDiff_ex_',ex_str,'_combOf_',comb_str, '_t.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_comb' )
%     end
% end

%% all random
iii = 1;
%input = zeros(N1,N2, NUMBER_MNIST);
diffused = zeros(N1,N2, NUMBER_MNIST);
diffused_nohete = zeros(N1, N2, NUMBER_MNIST);
kappa = 1./(3*generateRandomKappa(NUMBER_MNIST, N3, (avg_sca), avg_var));
for i=1:NUMBER_MNIST
    disp(i)
    %kappa(:,:,i) = generateRandomKappa(numberK);
        input(:,:,i) = MNIST(:,:,i);
        input3D = zeros(N1,N2,N3);
        input3D(:,:,1) = input(:,:,i);
        prov_diffused(:,:) = diffuseMNIST3(kappa(:,:,:,i), input3D); 
        prov_diffused_control(:,:) = diffuseMNIST3(kappa(:,:,:,i), checkboard);
        diffused(:,:,i) = prov_diffused(:,:);
        diffused_control(:,:,i) = prov_diffused_control(:,:);
        
end

imagesInput_ = input;
imagesDiff_ = reshape(squeeze(diffused(:,:,:)), [1,60,60,NUMBER_MNIST]);
imagesControl_ = reshape(squeeze(diffused_control(:,:,:)), [1,60,60,NUMBER_MNIST]);


imagesInput = imagesInput_(:,:,1:train_num);
imagesDiff = imagesDiff_(:,:,:,1:train_num);
imagesControl = imagesControl_(:,:,:,1:train_num);
save([datasets_name,'_allRandom_', '.mat'],'-v7.3','imagesInput','imagesControl','imagesDiff', 'kappa' )

imagesInput = imagesInput_(:,:,train_num+1:t_sam);
imagesDiff = imagesDiff_(:,:,train_num+1:t_sam);
imagesControl = imagesControl_(:,:,train_num+1:t_sam);
save([datasets_name,'_allRandom', '_t.mat'],'-v7.3','imagesInput','imagesControl','imagesDiff', 'kappa' )

imagesInput = imagesInput_(:,:,t_sam:v_sam);
imagesDiff = imagesDiff_(:,:,t_sam:v_sam);
imagesControl = imagesControl_(:,:,t_sam:v_sam);
save([datasets_name,'_allRandom', '_v_t.mat'],'-v7.3','imagesInput','imagesControl','imagesDiff', 'kappa' )
end
end
%% load and permute
% direc = dir('*.mat');
% names = {direc.name};
% for i = 1:numel(names)
%     clearvars -except names i
%     load(names{i});
%     Nr = randperm(size(imagesDiff,4));
%     imagesInput = imagesInput(:,:,Nr);
%     imagesDiff = imagesDiff(:,:,:,Nr);
%     i
%     if contains(names{i},'combOf')
%          save(names{i},'-v7.3','imagesInput','imagesDiff', 'kappa_comb' )
%     elseif contains(names{i}, 'from1to') 
%         save(names{i},'-v7.3','imagesInput','imagesDiff', 'kappa_set')
%     %else
%     %    diffused_nohete = diffused_nohete(:,:,Nr);
%     %    save(names{i},'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete' )
% 
%     end
% end
% 

% basis.delete();
% %clear basis
% mesh.delete(); 
% %clear mesh

%% FEM Define
function out2d = diffuseMNIST3_old(kappa,input)
% takes as input kappa and mnist, returns phi over last slice
    global mesh
    global basis
    %kappa = kappa *0+0.001;
    Kappa = basis.Map('B->M', kappa);
    Input = basis.Map('B->M', input);
    L = dotSysmat(mesh,0*ones(mesh.NodeCount,1), Kappa,1.4*ones(mesh.NodeCount,1),0);
    M = mesh.Massmat;
    Out = pcg(gpuArray(L),gpuArray(Input));
    %Out = L\Input;
    out = reshape( basis.Map('M->B', Out), size(kappa));
    out2d = out(:,:,end);
    out2d = out2d/max(out2d);
    figure(1);imagesc(out2d);
end

function out2d = diffuseMNIST3(kappa,input)
% takes as input kappa and mnist, returns phi over last slice
    global mesh
    global basis
    Mus = basis.Map('B->M',1/(3*kappa));
    Input = basis.Map('B->M', input);
    L = dotSysmat2_noC(mesh,1e-7*ones(mesh.NodeCount,1),Mus,1.4*ones(mesh.NodeCount,1));
    Out = L\Input;
    out = reshape( basis.Map('M->B', Out), size(kappa));
    out2d = out(:,:,end);
    out2d = out2d/max(out2d(:));
    %figure(1);imagesc(out2d);
end


function Phi = TimeSolver(M,K,qvec)
    dt = 10;      % step size [ps]  
    nstep = 10; % number of time steps
    theta = 0.5;  % Crank-Nicholson
    K0 = gpuArray(-(K * (1-theta) - M * 1/dt)); % matrix for step n
    K1 = gpuArray(K * theta + M * 1/dt);        % matrix for step n+1
    
    %Out = gather(pcg(gpuArray(L),gpuArray(Input),1e-7,200))
    q = gpuArray(qvec/dt);  % source at n=0
    phi(:,1) = pcg(K1,q,1e-7,200);   % Phi_1
    
    for i=2:nstep % loop over remaining steps
        q = K0 * phi(:,i-1);
        phi(:,i) = pcg(K1,q,1e-7,200);
        
    end
    Phi = sum(gather(phi),2);
    return
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function out = diffuseMNIST(kappa, mnist, iter)
    h = mnist(:);
    N = 60;
    for i = 1:iter
        %out = dt * L\h;
        L = assembleLaplacian(kappa(:,:,i));
        AA = speye(N*N) + L;
        h = AA*h;
        out(:,:,i) = reshape(h,N,N);
        %figure(1), imagesc(out(:,:,i)),pause(0.05);
    end
end

%%
function out2d = diffuseMNIST3_neu(kappa, input)

    h = input(:);
    
    [N1,N2,N3] = size(kappa);    
    L = -assembleLaplacian3(ones(size(kappa(2:end-1,2:end-1,2:end-1))));
    
    %AA = L;
    h = L\h;
    %h = AA*h;
    
    out_res(:,:,:) = reshape(h,[N1,N2,N3]);
    out2d = out_res(:,:,end);
        %figure(1), imagesc(out(:,:,i)),pause(0.05);
end


function out2d = diffuseMNIST3_wrong(kappa, input)  
    %input = input.* kappa;%generateRandomKappa(1, 1, 1);
    h = input(:);
    [N1,N2,N3] = size(kappa);
    
    k1 = ones(size(kappa));
    k2 = 0.1*k1;
    k3 = 30*k1;
    L1 = assembleLaplacian3(k1);
    L2 = assembleLaplacian3(k2);
    L3 = assembleLaplacian3(k3);
    
    L = assembleLaplacian3(kappa);
    
    %AA = L;
    out = (-L)\h;
    %h = AA*h;
    
    out_res(:,:,:) = reshape(out,[N1,N2,N3]);
    out2d = out_res(:,:,end);
        %figure(1), imagesc(out(:,:,i)),pause(0.05);
end


function out2d = diffuseMNIST3_refl(kappa, input)
    input(:,:,1) = 1 - input(:,:,1);
    input = flip(input,3);
    input0 = zeros(size(kappa));
    input0(:,:,1) = 1;
    [N1,N2,N3] = size(kappa);    
    
    L = assembleLaplacian3(kappa);    
    
    h = input(:);
    
    %AA = L;
    illumination = reshape(L\ (-input0(:)), [N1,N2,N3]);
    
    q0 = flip(input0,3).*illumination;
    q1 = input.*illumination;
    %h = AA*h;
    h0 =reshape( L\ (- q0(:)),[N1,N2,N3]);
    %h0 = h0/ max(h0(:));
    h1 =reshape( L\ (-q1(:)),[N1,N2,N3]);
   % h1 = h1 /max(h1(:));

    out_res(:,:,:) =(h0 - h1)./(1+h0);
    out2d = out_res(:,:,1);
        %figure(1), imagesc(out(:,:,i)),pause(0.05);
end


function L = assembleLaplacian(kappa)

    N = 60;
    n1 = N;
    n2 = N;
    oneN = ones(N+1,1);
    D1x = spdiags([oneN -oneN],-1:0,N+1,N);
    D2x = -D1x'*D1x; 
    % The following allows interpolation to pixel mid-points;
    intx1d = spdiags([0.5*oneN 0.5*oneN],-1:0,N+1,N);

    D1x3d = kron(speye(N),D1x);
    D1y3d = kron(D1x,speye(N));
    intx2d = kron(speye(N),intx1d);
    inty2d = kron(intx1d,speye(N));

    % define kappa
    kap2d = kappa;

    kap1d = reshape(kap2d,[],1);
    kap1x = spdiags(intx2d*kap1d,0:0,(n1+1)*n2,(n1+1)*n2);
    kap1y = spdiags(inty2d*kap1d,0:0,n1*(n2+1),n1*(n2+1));
    L = -(D1x2d'*kap1x*D1x2d + D1y2d'*kap1y*D1y2d + D1z2d'*kap1z*D1z2d);
end


% % %%
% h = figure;set(h,  'Position', [20,20,1520, 320]);
% mnist_sample = 1;
% limits = [0 1]
% subplot(1,5,1);imagesc(imagesInput(:,:,mnist_sample)), axis image,caxis(limits),colorbar, title('Input','fontsize', 8)
% subplot(1,5,2);imagesc(kappa_set(:,:)), axis image, colorbar,title('Slice of Heterogenous Kappa','fontsize', 8)
% subplot(1,5,3);imagesc(diffused_nohete(:,:,mnist_sample)), axis image, colorbar,title('Homogenoeus Diffusion', 'fontsize', 8)
% subplot(1,5,4);imagesc(squeeze(imagesDiff(1,:,:,mnist_sample))), axis image, colorbar, title('Heterogenous Diffusion','fontsize', 8)
% subplot(1,5,5);imagesc((squeeze(imagesDiff(1,:,:,mnist_sample))- diffused_nohete(:,:,1))./diffused_nohete(:,:,1)), axis image, colorbar,title({'Normalised Difference',  'of diffused images'}, 'fontsize', 8)
% 

% %% load and permute all avg sets
% direc = dir('avg_0.45/*te.mat');
% names = {direc.name};
% for i = 1:numel(names)
%     i
%     clearvars -except names i
%     a = load(['avg_0.3/',names{i}]);
%     b = load(['avg_0.45/',names{i}]);
%     c = load(['avg_0.9/',names{i}]);
%     imagesDiff = cat(4, a.imagesDiff, b.imagesDiff, c.imagesDiff);
%     imagesInput = cat(3, a.imagesInput, b.imagesInput, c.imagesInput);
%     Nr = randperm(size(imagesDiff,4));
%     imagesInput = imagesInput(:,:,Nr);
%     imagesDiff = imagesDiff(:,:,:,Nr);
%     
%     if contains(names{i},'combOf')
%          save(['avg_mix_0.3_0.45_0.9/',names{i}],'-v7.3','imagesInput','imagesDiff')
%     elseif contains(names{i}, 'from1to') 
%         save(['avg_mix_0.3_0.45_0.9/',names{i}],'-v7.3','imagesInput','imagesDiff')
%     else
%         %diffused_nohete = cat(3, a.diffused_nohete, b.diffused_nohete, c.diffused_nohete);
%         %diffused_nohete = diffused_nohete(:,:,Nr);
%         save(['avg_mix_0.3_0.45_0.9/',names{i}],'-v7.3','imagesInput','imagesDiff')
%     end
% end
% 
