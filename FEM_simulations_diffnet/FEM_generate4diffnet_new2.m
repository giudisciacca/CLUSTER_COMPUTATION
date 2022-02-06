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
N3 = 15;
numberK = 1;
train_num = 800;
t_sam = 1000;
v_sam = 1200;
totsam = 1200;

meshfact = 1;
lavg_sca = [0.25,0.5,1,2];%,4];
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
datasets_name = ['FEM_2021_15mmremake_avgsca',num2str(avg_sca),'_avgvar_',num2str(avg_var),'_'];
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
input = zeros(N1,N2, NUMBER_MNIST*numberK);
diffused = zeros(N1,N2, NUMBER_MNIST*numberK);
diffused_nohete = zeros(N1, N2, NUMBER_MNIST);
checkboard = zeros(N1,N2,N3);
checkboard(3:6:end,3:6:end,1) = 1;

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
save([datasets_name,'_allRandom', '.mat'],'-v7.3','imagesInput','imagesControl','imagesDiff', 'kappa' )

imagesInput = imagesInput_(:,:,train_num+1:t_sam);
imagesDiff = imagesDiff_(:,:,:,train_num+1:t_sam);
imagesControl = imagesControl_(:,:,:,train_num+1:t_sam);
save([datasets_name,'_allRandom', '_t.mat'],'-v7.3','imagesInput','imagesControl','imagesDiff', 'kappa' )

imagesInput = imagesInput_(:,:,t_sam:v_sam);
imagesDiff = imagesDiff_(:,:,:,t_sam:v_sam);
imagesControl = imagesControl_(:,:,:,t_sam:v_sam);
save([datasets_name,'_allRandom', '_v_t.mat'],'-v7.3','imagesInput','imagesControl','imagesDiff', 'kappa' )
end
end
%% FEM Define
function out2d = diffuseMNIST3(kappa,input)
% takes as input kappa and mnist, returns phi over last slice
    global mesh
    global basis
    Mus = basis.Map('B->M',1/(3*kappa));
    Input = basis.Map('B->M', input);
    L = dotSysmat2_noC(mesh,1e-7*ones(mesh.NodeCount,1),Mus,1.4*ones(mesh.NodeCount,1));
    Out = L\Input;
    %Out = abs(gather(pcg(gpuArray(L),gpuArray(Input),1e-8,200)));
    out = reshape( basis.Map('M->B', Out), size(kappa));
    out2d = out(:,:,end);
    out2d = out2d/max(out2d(:));
    %figure(1);imagesc(out2d);
end


