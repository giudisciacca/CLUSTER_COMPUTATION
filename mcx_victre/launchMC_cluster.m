function launchMC_cluster(N)

% load
mcx_install
addpath ../
EXP_DATA = 0;
current = cd;
cd /home/gdisciac/SOLUS
DOT_install
cd /home/gdisciac/mcx_victre
SetQM_DOT
folder = '/home/gdisciac/SOLUS/example/VICTRE_PARADIGM/';
load([folder,'VICTRE_PARADIGM_',num2str(N)]);
load('EXP_Tomo.mat')
irf = EXP.irf.data;

nstep = 325;
mcdt = 25;

% simulate
DataTD = zeros([nstep,8,8,8]);
hete=zeros([nstep,8,8,8]);
homo=zeros([nstep,8,8,8]);
RefTD = zeros([nstep,8,8,8]);
for ilambda=1:8
disp('hete calc')
ilambda

[hete(:,:,:,ilambda),~ ]= MC_VICTRE2(structure,structopt.mua(:,:,:,ilambda),structopt.musp(:,:,:,ilambda), DOT.Source.Pos ,DOT.Detector.Pos);
% simulate homo
disp('homo calc')
[homo(:,:,:,ilambda),taxis] = MC_VICTRE2(structure,structopt.mua0(:,:,:,ilambda),structopt.musp0(:,:,:,ilambda), DOT.Source.Pos ,DOT.Detector.Pos);
end

% save theodata
save(['Test_Standard_MCtheo_',num2str(N)],'hete','homo')

% resample IRF and repmat
t = EXP.time.axis;
t = -t(1)+t;
taxis = 0:mcdt:((nstep-1)*mcdt);
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
IRF(isnan(IRF)) = 0;
if size(IRF,2)==8
    IRF = permute(IRF,[1,3,4,2]);
    IRF = repmat(IRF,[1,8,8,1]);
    IRF = reshape(IRF, [size(IRF,1),numel(IRF)/size(IRF,1)]);   
end
% convolve by IRF
DataTD = ConvIRF(reshape(hete,[size(hete,1), numel(hete)/size(hete,1)]),IRF);
RefTD = ConvIRF(reshape(homo,[size(homo,1), numel(homo)/size(homo,1)]),IRF);

% save forwdata
save(['Test_Standard_MC_',num2str(N)],'RefTD','DataTD')
end
