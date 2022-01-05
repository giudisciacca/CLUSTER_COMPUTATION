function launchMC(N)
N=1
% load
 addpath(genpath('/cs/academic/phd3/gdisciac/mcx/src'));
 addpath(genpath('/cs/research/medim/gdisciac/mcx/mcxlab/'));
addpath(genpath('/cs/academic/phd3/gdisciac/mcx/mcxlab/'));

addpath ../
EXP_DATA = 0;
SetQM_DOT
folder = '/cs/research/medim/gdisciac/SOLUS/example/VICTRE_PARADIGM/Blood_corrected/';
load([folder,'VICTRE_PARADIGM_',num2str(N)]);
load('EXP_Tomo.mat')
irf = EXP.irf.data;

% simulate
DataTD = zeros([256,8,8,8]);
hete=zeros([256,8,8,8]);
homo=zeros([256,8,8,8]);
RefTD = zeros([256,8,8,8]);
for ilambda=1:8

[hete(:,:,:,ilambda),~ ]= MC_VICTRE2(structure,structopt.mua(:,:,:,ilambda),structopt.musp(:,:,:,ilambda), DOT.Source.Pos ,DOT.Detector.Pos);
% simulate homo
[homo(:,:,:,ilambda),taxis ]= MC_VICTRE2(structure,structopt.mua0(:,:,:,ilambda),structopt.musp0(:,:,:,ilambda), DOT.Source.Pos ,DOT.Detector.Pos);
end

% resample IRF and repmat
t = EXP.time.axis;
t = -t(1)+t;
taxis = 0:95:255*95;
tover = 0:min(t(2)-t(1),taxis(2)-taxis(1))/2:max(taxis(end),t(end));
% if taxis(end)>t(end)
%    t = t(1):(t(2)-t(1)):taxis(end);
%    irf = cat(1, irf_old,zeros(numel(t)-size(irf,1),8));   
% end
for i = 1:size(irf,2) 
    y = interp1(t,irf(:,i)',tover);
    y = interp1(tover,y,taxis);
    IRF(:,i)=y;
end

if size(IRF,2)==8
    IRF = permute(IRF,[1,3,4,2]);
    IRF = repmat(IRF,[1,8,8,1]);
    IRF = reshape(IRF, [size(IRF,1),numel(IRF)/size(IRF,1)]);   
end

% save theodata
save(['Test_Standard_MC_theo_',num2str(N)],'hete','homo')

% convolve by IRF
DataTD = ConvIRF(reshape(hete,[size(hete,1), numel(hete)/size(hete,1)]),IRF);
RefTD = ConvIRF(reshape(homo,[size(homo,1), numel(homo)/size(homo,1)]),IRF);

% save forwdata
save(['Test_Standard_MC_',num2str(N)],'RefTD','DataTD')
end