clear
close all

NUMBER_MNIST = 1000;
MNIST = zeros([60,60,NUMBER_MNIST]); 

MNIST_0 = loadMNISTImages('t10k-images.idx3-ubyte');
MNIST_0 = reshape(MNIST_0,[28,28,10000]);

MNIST(17:44,17:44,:) = MNIST_0(:,:,1:NUMBER_MNIST);
% for i_tr = 1:NUMBER_MNIST
%      MNIST(:,:,i_tr) = circshift(MNIST(:,:,i_tr) ,[randi(28)-14 , randi(28) - 14]);
% end

% MNIST_0 = load('../stl10_matlab/train.mat');
% MNIST_0 = reshape(MNIST_0.X,[5000, 96,96,3]);
% MNIST_0 = permute( MNIST_0(1:NUMBER_MNIST, 19:78, 19:78,:), [2,3,4,1]);
% % for i_tr= 1:NUMBER_MNIST
% %     MNIST(:,:, i_tr) = squeeze(double(mean( rgb2gray(MNIST_0(:,:,:,i_tr)), 3) ))/255;
% % end
    
N1 = 60;
N2 = 60;
N3 = 10;
numberK = 5;
train_num = 850;
tot_sam = 1000;

avg_sca = 1;
kappa = generateRandomKappa(numberK, N3, 1/(3*(avg_sca)));

%% ACTUAL_DISTR
iii = 1;

input = zeros(N1,N2, NUMBER_MNIST*numberK);
diffused = zeros(N1,N2, NUMBER_MNIST*numberK);
diffused_nohete = zeros(N1, N2, NUMBER_MNIST);
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
        
        iii = iii+1;
    end 
    prov_diffused_nohete(:,:) = diffuseMNIST3(1/(3*(avg_sca)*ones(N1,N2,N3)), input3D);
    diffused_nohete(:,:,i) = prov_diffused_nohete(:,:);
end


imagesInput = input;
imagesDiff = reshape(squeeze(diffused(:,:, :)), [1,60,60,numberK*NUMBER_MNIST]);
%save(['OriginalDiffTOTRANDOM0.2perc', '.mat'],'-v7.3','imagesInput','imagesDiff')

for i=1:numberK
    idx = i:numberK:numberK*NUMBER_MNIST;
    imagesInput_ = input(:,:,idx);
    imagesDiff_ = reshape(squeeze(diffused(:,:, idx)), [1,60,60,NUMBER_MNIST]);i_str = num2str(i);
    kappa_set = kappa(:,:,:,i);
    %save(['OriginalDiff_allRandom_',i_str, '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set' )
    imagesInput = imagesInput_(:,:,1:train_num);
    imagesDiff = imagesDiff_(:,:,:,1:train_num);
    save(['OriginalDiff_',i_str, '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')
    imagesInput = imagesInput_(:,:,train_num +1 : tot_sam);
    imagesDiff = imagesDiff_(:,:,:,train_num +1 : tot_sam);
    save(['OriginalDiff_',i_str, '_t.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')
    if i == 1
        imagesInput_ = input(:,:,idx);
        imagesDiff_ = reshape(squeeze(diffused_nohete(:,:,:)), [1,60,60,NUMBER_MNIST]);i_str = num2str(i);
        
        imagesInput = imagesInput_(:,:,1:train_num);
        imagesDiff = imagesDiff_(:,:,:,1:train_num);
        save(['OriginalDiff_nohete', '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')
    
        imagesInput = imagesInput_(:,:,train_num +1 : tot_sam);
        imagesDiff = imagesDiff_(:,:,:,train_num +1 : tot_sam);
        save(['OriginalDiff_nohete', '_t.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')    
    end
end


%% sets together
n_tog =  [4];

for i=1:numel(n_tog)
    idx = 1:numberK:numberK*NUMBER_MNIST;
    for ii = 2:n_tog(i)
        idx0 = ii:numberK:numberK*NUMBER_MNIST;
        idx = cat(1,idx,idx0);
    end
    idx = sort(idx);
    imagesInput_ = input(:,:,idx);
    imagesDiff_ = reshape(squeeze(diffused(:,:, idx)), [1,60,60,NUMBER_MNIST*n_tog(i)]);i_str = num2str(n_tog(i));
    kappa_set = kappa(:,:,:,i);
    
    imagesInput = imagesInput_(:,:,1:train_num);
    imagesDiff = imagesDiff_(:,:,:,1:train_num);
    save(['OriginalDiff_from1to',i_str, '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')

    imagesInput = imagesInput_(:,:,train_num +1 : tot_sam);
    imagesDiff = imagesDiff_(:,:,:,train_num +1 : tot_sam);
    save(['OriginalDiff_from1to',i_str, '_t.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete')
end



 %% combinations
n_comb =  [4];
for comb=1:numel(n_comb)
    for ex = 1:2
        linear_coeff = rand(1,1,1,n_comb(comb));
        linear_coeff = linear_coeff/sum(linear_coeff); 
        kappa_comb = sum(linear_coeff.* kappa(:,:,:,1:n_comb(comb)),4);
        input = zeros(60,60, NUMBER_MNIST);
        diffused = zeros(60,60, NUMBER_MNIST);
        for i=1:NUMBER_MNIST
            disp(i)            
            input(:,:,i) = MNIST(:,:,i);
            input3D = zeros(N1,N2,N3);
            input3D(:,:,1) = input(:,:,i);
            prov_diffused(:,:) = diffuseMNIST3(kappa_comb(:,:,:), input3D); 

            %prov_diffused(:,:,:) = diffuseMNIST3(kappa_comb(:,:), MNIST(:,:,i), dt, iter); 
            diffused(:,:,i) = prov_diffused(:,:);                
        end
        imagesInput_ = input;
        imagesDiff_ = reshape(squeeze(diffused(:,:,:)), [1,60,60,NUMBER_MNIST]);
        comb_str = num2str(n_comb(comb));
        ex_str = num2str(ex);
        
        imagesInput = imagesInput_(:,:,1:train_num);
        imagesDiff = imagesDiff_(:,:,:,1:train_num);
        save(['OriginalDiff_ex_',ex_str,'_combOf_',comb_str, '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_comb')
 
        imagesInput = imagesInput_(:,:,train_num+1:tot_sam);
        imagesDiff = imagesDiff_(:,:,:,train_num+1:tot_sam);
        save(['OriginalDiff_ex_',ex_str,'_combOf_',comb_str, '_t.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa_comb' )
    end
end

%% all random
iii = 1;
%input = zeros(N1,N2, NUMBER_MNIST);
diffused = zeros(N1,N2, NUMBER_MNIST);
diffused_nohete = zeros(N1, N2, NUMBER_MNIST);
kappa = generateRandomKappa(NUMBER_MNIST, N3, 1/3*(avg_sca));
for i=1:NUMBER_MNIST
    disp(i)
    %kappa(:,:,i) = generateRandomKappa(numberK);
        input(:,:,i) = MNIST(:,:,i);
        input3D = zeros(N1,N2,N3);
        input3D(:,:,1) = input(:,:,i);
        prov_diffused(:,:) = diffuseMNIST3(kappa(:,:,:,i), input3D); 
        diffused(:,:,i) = prov_diffused(:,:);
end

imagesInput_ = input;
imagesDiff_ = reshape(squeeze(diffused(:,:,:)), [1,60,60,NUMBER_MNIST]);

imagesInput = imagesInput_(:,:,1:train_num);
imagesDiff = imagesDiff_(:,:,:,1:train_num);
save(['OriginalDiff_allRandom', '.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa' )

imagesInput = imagesInput_(:,:,train_num+1:tot_sam);
imagesDiff = imagesDiff_(:,:,:,train_num+1:tot_sam);
save(['OriginalDiff_allRandom', '_t.mat'],'-v7.3','imagesInput','imagesDiff', 'kappa' )



%% load and permute
direc = dir('*.mat');
names = {direc.name};
for i = 1:numel(names)
    clearvars -except names i
    load(names{i});
    Nr = randperm(size(imagesDiff,4));
    imagesInput = imagesInput(:,:,Nr);
    imagesDiff = imagesDiff(:,:,:,Nr);
    i
    if contains(names{i},'combOf')
         save(names{i},'-v7.3','imagesInput','imagesDiff', 'kappa_comb' )
    elseif contains(names{i}, 'from1to') 
        save(names{i},'-v7.3','imagesInput','imagesDiff', 'kappa_set')
    %else
    %    diffused_nohete = diffused_nohete(:,:,Nr);
    %    save(names{i},'-v7.3','imagesInput','imagesDiff', 'kappa_set', 'diffused_nohete' )

    end
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


function out2d = diffuseMNIST3(kappa, input)  
    input = input .* generateRandomKappa(1, 1, 1);
    h = input(:);
    [N1,N2,N3] = size(kappa);
    
    L = -assembleLaplacian3(kappa);
    
    %AA = L;
    h = L\h;
    %h = AA*h;
    
    out_res(:,:,:) = reshape(h,[N1,N2,N3]);
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
