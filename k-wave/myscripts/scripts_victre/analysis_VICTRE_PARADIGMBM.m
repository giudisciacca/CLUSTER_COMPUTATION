% analysis victre
%Con = {'125e-3','25e-2','1e-1','1','2','3','4','5','6','7'};
clear
close all
NN =1;
ConN = 1:NN%00;%,'1e-1','1','2','3','4','5','6','7'};
for i = ConN
    Con{i}  = num2str(ConN(i));
end
FIT_STR = 'MeanFitted';
type='fit4param'
sumFailed = [];
for isim_back = 1:numel(Con)
    isim = isim_back;%NN +1 -isim_back;
    disp(isim)
    BM='M';
    if isim > 50
        isim = isim-50;
        BM='B';
    end
    loadname = ['Test_Standard_PARADIGM',BM,'__type',type,'_tau5e-03_mu0MeanFittedSample',Con{isim},'_REC.mat'];
    if exist(loadname,'file')
    tmp = load(loadname);
    tmp.REC.loadname = loadname;

    %tmp.REC.method = method{1};
    RECset(isim) = tmp.REC;
    Qtmp= QuantifyDOT(tmp.REC,1);
    Qtmp.SOLUS_FigMerit.mua.ValInSca = Qtmp.SOLUS_FigMerit.musp.ValIn;
    Qtmp.SOLUS_FigMerit.mua.Cs = Qtmp.SOLUS_FigMerit.musp.C;
    Qset(isim) = Qtmp.SOLUS_FigMerit.mua;
    %Qset(isim).ValInSca = Qtmp.SOLUS_FigMerit.mua.ValIn;
    load(['VICTRE_PARADIGM_',Con{isim} ,'.mat'])
    Ground(isim).mua = contrastMua;
    Ground(isim).mass = voidMass;
    vM = repmat(voidMass,[1 1 1 8]);
    for l =1:8
        id = zeros(size(vM));
        id(:,:,:,l)=vM(:,:,:,l); 
        Ground(isim).muain(l) = mean(structopt.mua(id(:)==1));
        Ground(isim).muspin(l) = mean(structopt.musp(id(:)==1));
        Ground(isim).mua0(l) = mean(structopt.mua0(id(:)==1));
        Ground(isim).label = label;
        %Ground(isim).Vol = sum(voidMass(:));
    end
    Ground(isim).Vol = sum(voidMass(:));
    Ground(isim).z = sum( (1:90).*squeeze(sum(sum(voidMass,2),1))')/sum(voidMass(:));
    else
        
        disp(loadname)
    end
    
end
%%
Ct =[];
Cr = [];
At = [];
Ar = [];
Ct0 = [];
CNR =[];
bkA = [];
bkAfit = [];
inAt = [];
inAr = [];
inArV = [];
inSrV = [];
inArZ = [];
inSrZ = [];
inSr =[];
inSt = [];
IDX = [];
IDXL=[];
Crs =[];
for i = 1:NN%00
    idx = (1+(i-1)*8:i*8)'
    idxl = i;
    if strcmpi(Ground(i).label,'benign')
        IDX = cat(1,IDX,idx);
        IDXL = cat(1,IDXL,i);
    end
    Ct = cat(1,Ct,(+Ground(i).muain(:) - RECset(i).opt.mua0(:))./RECset(i).opt.mua0(:));
    Ct0 = cat(1,Ct0,(+Ground(i).muain(:) -Ground(i).mua0(:))./Ground(i).mua0(:));
    Cr = cat(1,Cr,Qset(i).C(:));
    Crs = cat(1,Crs,Qset(i).Cs(:));
    CNR = cat(1,CNR,Qset(i).CNR(:));
    At = cat(1,At,Ground(i).muain(:));
    Ar = cat(1,Ar,Qset(i).acc(:));
    bkA = cat(1,bkA,Ground(i).mua0(:));
    bkAfit = cat(1,bkAfit,RECset(i).opt.mua0(:));
    inAt = cat(1,inAt,Ground(i).muain(:));
    inSt = cat(1,inSt,Ground(i).muspin(:)); 
    inAr = cat(1,inAr,Qset(i).ValIn(:) );
    inSr = cat(1,inSr,Qset(i).ValInSca(:) );
    inArV = cat(1,inArV,Qset(i).ValIn(:)/Ground(i).Vol );
    inSrV = cat(1,inSrV,Qset(i).ValInSca(:)/Ground(i).Vol );
    inArZ = cat(1,inArZ,Qset(i).ValIn(:)*Ground(i).z );
    inSrZ = cat(1,inSrZ,Qset(i).ValInSca(:)*Ground(i).z );
end
%%
 
if 1 == 1
 res_fact = 1;
for i = 2:2%numel(RECset)
    REC = RECset(i);
     Nr = ceil(sqrt(REC.radiometry.nL));Nc = round(sqrt(REC.radiometry.nL));
     
     
    nfig = 500+i;
    fh = figure(nfig);
    for inl = 1:REC.radiometry.nL
    SubPlotMap(reshape(REC.opt.bmua(:,inl),REC.grid.dim),...
        [num2str(REC.radiometry.lambda(inl)) ' nm'],nfig,Nr,Nc,inl,res_fact);
    end
    fh.NumberTitle = 'off';fh.Name = ['Reconstructed Absorption Con', Con{i}];
    %    ------------------------ Reference musp ----------------------------------
%     nfig = 600+i;
%     fh = figure(nfig);
%     for inl = 1:REC.radiometry.nL
%     SubPlotMap(reshape(REC.opt.bmusp(:,inl),REC.grid.dim),...
%         [num2str(REC.radiometry.lambda(inl)) ' nm'],nfig,Nr,Nc,inl,res_fact);
%     end
%     fh.NumberTitle = 'off';fh.Name = ['Reconstructed Scattering tau', num2str(tauset(i))];
%     
%     % -------------------------------------------------------------------------------
    
end
end
 %%
% if SAVE ==1 && numel(RECset) ==1
%     savename = ['reconAbs','_',exfile,'_type_', method{1},'_tau_',num2str(tau,'%10.0e'),'.pdf']
%     fh;
%     print(savename,'-dpng')
% end
close all
figure; scatter(Ct0, Cr, 4,'r'), title('C vs Ct (ground) '), axis([-2,2,-2,2]), grid on
hold on
scatter(Ct0(IDX), Cr(IDX), 4,'g'), title('C vs Ct (ground) '), axis([-2,2,-2,2]), grid on

figure; scatter(Ct, Cr,4,'r'), title('C vs Ct '), axis([-2,2,-2,2]), grid on
hold on
scatter(Ct(IDX), Cr(IDX),4,'g'), title('C vs Ct '), axis([-2,2,-2,2]), grid on

figure; scatter(Ct, CNR,'r'),title('CNR vs C '), axis([-2,2,-2,2]), grid on
hold on
scatter(Ct(IDX), CNR(IDX),'g'),title('CNR vs C '), axis([-2,2,-2,2]), grid on

figure; scatter(bkA, bkAfit,4,'r'),title('Homogeneous fit: fitted vs true   '),grid on
hold on
scatter(bkA(IDX), bkAfit(IDX),4,'g'),title('Homogeneous fit: fitted vs true   '),grid on

figure; scatter(inAt, inAr,4,'r'),title('Abs vs Recon Abs  '), axis([-0.005,0.1,-0.1,0.1]), grid on
hold on
scatter(inAt(IDX), inAr(IDX),4,'g'),title('Abs vs Recon Abs  '), axis([-0.005,0.1,-0.1,0.1]), grid on
[a,b] = hist(inAt,15);
plot(b, a/sum(a(:)))

lsym = {'*','o','*','v','s','d','x','^'};
figure;
for l = 1:8

scatter(inSr(l:8:end), inAr(l:8:end),10,[lsym{l},'r']),title('Recon Sca vs Recon Abs by lambda  '),grid on %axis([-0.005,0.03,-0.005,0.03]), grid on
hold on
inSr_r = inSr(IDX);
inAr_r = inAr(IDX);
scatter(inSr_r(l:8:end), inAr_r((l:8:end)),10,[lsym{l},'g']), %axis([-0.005,0.03,-0.005,0.03]), grid on

end


lsym = {'*','o','*','v','s','d','x','^'};
ld =[6,7,8];
figure;
scatter3(inAr(ld(1):8:end), inAr(ld(2):8:end),inAr(ld(3):8:end),10,[lsym{l},'r']),title('Recon Sca vs Recon Abs by lambda 6 7 8 '),grid on %axis([-0.005,0.03,-0.005,0.03]), grid on
hold on
inSr_r = inSr(IDX);
inAr_r = inAr(IDX);
scatter3(inAr_r(ld(1):8:end), inAr_r(ld(2):8:end),inAr_r(ld(3):8:end),10,[lsym{l},'g']), %axis([-0.005,0.03,-0.005,0.03]), grid on

%

tmp = reshape(inAt',[size(inAr,1)/8,8]);
[coeff,score,latent] = pca(tmp);
Xcentered = score*coeff';
scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')

% tmp = reshape(cat(2,randn(1,75*8),8+randn(1,75*8)),[8,150])';
% [coeff,score,latent] = pca(tmp);
% Xcentered = score*coeff';
% scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
% scatter(Xcentered(75:150,1),Xcentered(75:150,2),'g')


close all
tmp = reshape([inAr';inSr'],[16,2*size(inAr,1)/16])';
tmp1 = tmp;
tmp =tmp(:,end-12:end);
[coeff,score,latent] = pca(tmp);%,'VariableWeights','variance');
Xcentered = score*coeff';
figure
  scatter3(Xcentered(:,1),Xcentered(:,2),Xcentered(:,3),'r'),hold on
  scatter3(Xcentered(IDXL,1),Xcentered(IDXL,2),Xcentered(IDXL,3),'g')
figure
 scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
 scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')

close all
tmp = reshape([inArZ';inSrZ'],[16,2*size(inArV,1)/16])';
tmp1 = tmp;
tmp =tmp(:,end-12:end);
[coeff,score,latent] = pca(tmp);%,'VariableWeights','variance');
Xcentered = score*coeff';
figure
 scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
 scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')
 
 
% truth
tmp = reshape([inAt';inSt'],[16,2*size(inAr,1)/16])';
tmp = tmp;
[coeff,score,latent] = pca(tmp);
Xcentered = score*coeff';
% scatter3(Xcentered(:,1),Xcentered(:,2),Xcentered(:,3),'r'),hold on
% scatter3(Xcentered(IDXL,1),Xcentered(IDXL,2),Xcentered(IDXL,3),'g')
figure
scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')
%% moving from truth to recon
tmpC = [tmp;tmp1];
[coeff,score,latent] = pca(tmpC);
Xcentered = score*coeff';
% scatter3(Xcentered(:,1),Xcentered(:,2),Xcentered(:,3),'r'),hold on
% scatter3(Xcentered(IDXL,1),Xcentered(IDXL,2),Xcentered(IDXL,3),'g')
figure
scatter(Xcentered(1:NN,1),Xcentered(1:NN,2),'r'),hold on
scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')
scatter(Xcentered(NN:end,1),Xcentered(NN:end,2),'r^'),hold on
scatter(Xcentered(NN+IDXL,1),Xcentered(NN+IDXL,2),'g^')


% close all
% tmp = reshape([Cr';Crs'],[16,2*size(inAr,1)/16])';
% [coeff,score,latent] = pca(tmp);%,'VariableWeights','variance');
% Xcentered = score*coeff';
% %  scatter3(Xcentered(:,1),Xcentered(:,2),Xcentered(:,3),'r'),hold on
% %  scatter3(Xcentered(IDXL,1),Xcentered(IDXL,2),Xcentered(IDXL,3),'g')
%  scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
%  scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')

