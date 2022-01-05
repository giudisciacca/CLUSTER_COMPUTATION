% analysis victre
%Con = {'125e-3','25e-2','1e-1','1','2','3','4','5','6','7'};
clear
close all
nconc =5;
nsca = 2;
ntot = nconc+nsca;
NN =120
ConN = 1:NN%00;%,'1e-1','1','2','3','4','5','6','7'};
for i = ConN
    Con{i}  = num2str(ConN(i));
end
FIT_STR = 'MeanFitted';
type='spectral_usprior'
stringConc = {'Hb','HbO2','Lipid','H2O','Collagen','a','b'}
sumFailed = [];
for isim_back = numel(Con):-1:1
    isim = isim_back;%NN +1 -isim_back;
    disp(isim)
    
    loadname = ['Test_Standard_PARADIGM__type',type,'_tau5e-03_mu0MeanFittedSample',Con{isim},'_REC.mat'];
    if exist(loadname,'file')
    tmp = load(loadname);
    tmp.REC.loadname = loadname;

    %tmp.REC.method = method{1};
    RECset(isim) = tmp.REC;
    Qtmp= QuantifyDOT(tmp.REC,1);
    Qtmp.SOLUS_FigMerit.mua.ValIn = [];
    for ic =1:nconc
        Qtmp.SOLUS_FigMerit.mua.ValIn(ic) = Qtmp.SOLUS_FigMerit.(stringConc{ic}).ValIn;
        Ground(isim).ValOut(ic) = tmp.REC.spe.opt.conc0(ic);
    end
    for ic =nconc+1:nconc+nsca
        Qtmp.SOLUS_FigMerit.mua.ValIn(ic) = Qtmp.SOLUS_FigMerit.(stringConc{ic}).ValIn;
        if ic == 6
            Ground(isim).ValOut(ic) = tmp.REC.spe.opt.a0;
        elseif ic==7
            Ground(isim).ValOut(ic) = tmp.REC.spe.opt.b0;
        end
    end
    Qtmp.SOLUS_FigMerit.mua.Cs = Qtmp.SOLUS_FigMerit.musp.C;
    Qset(isim) = Qtmp.SOLUS_FigMerit.mua;
    %Qset(isim).ValInSca = Qtmp.SOLUS_FigMerit.mua.ValIn;
    load(['VICTRE_PARADIGM_',Con{isim} ,'.mat'])
%     Ground(isim).mua = contrastMua;
%     Ground(isim).mass = voidMass;
     vM = repmat(voidMass,[1 1 1 8]);
    for l =1:8
        id = zeros(size(vM));
        id(:,:,:,l)=vM(:,:,:,l); 
        Ground(isim).muain(l) = mean(structopt.mua(id(:)==1));
        Ground(isim).muspin(l) = mean(structopt.musp(id(:)==1));
        Ground(isim).mua0(l) = mean(structopt.mua0(id(:)==1));
         Ground(isim).label = label;
    end
    [~,~,Ground(isim).ValIn(1:5),Ground(isim).ValIn(6),Ground(isim).ValIn(7)] = FitVoxel(Ground(isim).muain,Ground(isim).muspin,tmp.REC.spe);
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
inSr =[];
inSt = [];
IDX = [];
IDXL=[];
Crs =[];
outAt = [];
for i = 1:NN%00
    idx = (1+(i-1)*ntot:i*ntot)';
    idxl = i;
    if strcmpi(Ground(i).label,'benign')
        IDX = cat(1,IDX,idx);
        IDXL = cat(1,IDXL,i);
    end
    inAr = cat(1,inAr,Qset(i).ValIn(:) );
    inAt = cat(1,inAt,Ground(i).ValIn(:) );
    outAt = cat(1,outAt,Ground(i).ValOut(:) );
    
end
%%
 
if 0 == 1
 res_fact = 1;
for i = 1:20%numel(RECset)
    REC = RECset(i);
     Nr = ceil(sqrt(REC.radiometry.nL));Nc = round(sqrt(REC.radiometry.nL));
     
     
%     nfig = 500+i;
%     fh = figure(nfig);
%     for inl = 1:REC.radiometry.nL
%     SubPlotMap(reshape(REC.opt.bmua(:,inl),REC.grid.dim),...
%         [num2str(REC.radiometry.lambda(inl)) ' nm'],nfig,Nr,Nc,inl,res_fact);
%     end
%     fh.NumberTitle = 'off';fh.Name = ['Reconstructed Absorption Con', Con{i}];
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
    
%     nfig = 700+i;
%     fh = figure(nfig);
%     for inl = 1:REC.radiometry.nL
%         SubPlotMap(reshape(REC.opt.Mua(:,:,:,inl),[REC.grid.dim]),...
%         [num2str(REC.radiometry.lambda(inl)) ' nm'],nfig,Nr,Nc,inl,res_fact);
%     end
%     fh.NumberTitle = 'off';fh.Name = ['Truth', Con{i}];
     nfig = 800+i;
     Nr = 3;Nc =3; 
    fh = figure(nfig);
    for ic = 1:REC.spe.nCromo
        SubPlotMap(reshape(REC.opt.bConc(:,ic),REC.grid.dim),...
            [REC.spe.cromo_label{ic} ' Map'],nfig,Nr,Nc,ic,res_fact);
    end
    % Hbtot and SO2
    REC.opt.HbTot = REC.opt.bConc(:,strcmpi(REC.spe.cromo_label,'hb'))+...
        REC.opt.bConc(:,strcmpi(REC.spe.cromo_label,'hbo2'));
    REC.opt.So2 = REC.opt.bConc(:,strcmpi(REC.spe.cromo_label,'hbo2'))./REC.opt.HbTot;
    SubPlotMap(reshape(REC.opt.HbTot,REC.grid.dim),'HbTot Map',nfig,Nr,Nc,ic+1,res_fact);
    SubPlotMap(reshape(REC.opt.So2,REC.grid.dim),'So2 Map',nfig,Nr,Nc,ic+2,res_fact);

    % a b scattering
    SubPlotMap(reshape(REC.opt.bA,REC.grid.dim),'a Map',nfig,Nr,Nc,ic+3,res_fact);
    SubPlotMap(reshape(REC.opt.bbB,REC.grid.dim),'b Map',nfig,Nr,Nc,ic+4,res_fact);

    fh.NumberTitle = 'off';fh.Name = ['Reconstructed Conc Sam', Con{i},' / ',Ground(i).label];

end
end

%%

tmp = reshape(inAt',[size(inAr,1)/ntot,ntot]);
tmp = (tmp - mean(tmp))./std(tmp);
labels = zeros(size(tmp,1),1);
labels(IDXL,1) = 1;
%genDataset('CLASSIFICATION_SPECTRAL',tmp',labels')
[coeff,score,latent] = pca(tmp);
Xcentered = score*coeff';
figure
scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')

% tmp = reshape(cat(2,randn(1,75*ntot),ntot+randn(1,75*ntot)),[ntot,150])';
% [coeff,score,latent] = pca(tmp);
% Xcentered = score*coeff';
% scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
% scatter(Xcentered(75:150,1),Xcentered(75:150,2),'g')
%%

close all
tmp = reshape([(inAr')],[ntot,size(inAr,1)/ntot])';
tmp1 = (tmp- mean(tmp))./std(tmp);
%tmp =tmp(:,end-12:end);
[coeff,score,latent] = pca(tmp);%,'VariableWeights','variance');
Xcentered = score*coeff';
figure
  scatter3(Xcentered(:,1),Xcentered(:,2),Xcentered(:,3),'r'),hold on
  scatter3(Xcentered(IDXL,1),Xcentered(IDXL,2),Xcentered(IDXL,3),'g')
figure
 scatter(Xcentered(:,1),Xcentered(:,2),'r'),hold on
 scatter(Xcentered(IDXL,1),Xcentered(IDXL,2),'g')

%
figure
 scatter(tmp(:,3),tmp(:,2),'r'),hold on
 scatter(tmp(IDXL,3),tmp(IDXL,2),'g')
