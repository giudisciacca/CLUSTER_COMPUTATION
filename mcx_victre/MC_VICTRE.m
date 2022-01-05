function [outdata,taxis] = MC_VICTRE(strlab, mua, musp,srcpos,detpos,backup)

%
% define
u_mm = 0.5;
cfg.isreflect = 1;
cfg.unitinmm = u_mm;%0.5;
tol_ph = 10^(5);

N_src = size(srcpos,1);
cfg.srctype='isotropic';
cfg.srcparam1 = [1,0,0,0];
%cfg.srcparam2 = [0,DIM(2),0,0];
cfg.srcdir=[0 0 1];
%cfg.gpuid=1

cfg.autopilot=1;

mcdt = 25*10^(-12);
nstep = 400;




ulab = unique(strlab(:));
cfg.prop = zeros(numel(ulab),4);
for i = 1:numel(ulab)
    ain = mean(mua(strlab(:)==ulab(i)));
    sin = mean(musp(strlab(:)==ulab(i)));
    cfg.prop(i,:) = [ain,sin,0,1.37] ;
    strlab(strlab==ulab(i))=i;
end
cfg.prop=cat(1, [0,0,1,1],cfg.prop);
cfg.vol= cat(3,0*zeros(size(strlab,1),size(strlab,2)),strlab);
offset =round(0.5*size(cfg.vol).*[1,1,0]);%*cfg.unitinmm.*[1,1,0];
cfg.vol = uint8(cfg.vol);
cfg.maxdetphoton = 3*10^6;
% calculate the flux distribution with the given config
tic;


detposs = ([(offset+[0,0,2])+((1/(u_mm))*detpos),5*ones(size(detpos,1),1)]);
srcposs =round((1/u_mm)*srcpos+offset+[0,0,2]);


tstep = mcdt;
%out = zeros([nstep,N_det,N_src]);
dat = zeros(nstep,8,8);
nph_det_tot = zeros(8,8);
if exist(backup,'file')
load(backup)
end
fluence_surf=zeros(size(cfg.vol,1),size(cfg.vol,2),nstep);
ph_reg = 0;

cfg.issrcfrom0 = 0;   

    
cfg.nphoton = 10^8;
cfg.maxdetphoton =2*10^7;

cfg.tstart =0;% (i_t-1)* tstep ;
cfg.tstep = tstep;
cfg.tend = tstep*(nstep);
dets = 1:8;
srcs = 1:8;

cfg.detpos = detposs;
flag_coll = 0;
while flag_coll == 0
    clear cfgs
    for i_src = srcs
        cfg.srcpos = srcposs(i_src,:) ;
        cfg.seed=randi(999999);
        cfgs(i_src) = cfg;
    end      

    %evalc('[~,outdet] = mcxlab(cfgs(srcs));');%,'preview');
    [~,outdet] = mcxlab(cfgs(srcs));

    for i_src = srcs
        for i_det =dets
            if ~isempty(outdet(srcs==i_src).data)
            	tmp = outdet(srcs==i_src);
                tmp.unitinmm = u_mm;
                [counts,nph_det]=mymcxdettpsf(tmp,i_det,cfg.prop,[0, cfg.tend,cfg.tstep]);            	
		%tmp.ppath =  tmp.ppath(tmp.detid==i_det,:)*u_mm;
            	%tof=mmcdettime(tmp,cfg.prop);
            	%[counts, ~]=histc(tof,0:cfg.tstep:cfg.tend);
	    else
		counts = 0;	
	    end
	    if isrow(counts)
                counts = counts';
            end
            if sum(squeeze(dat(:,i_det,i_src))) < tol_ph
                dat(:,i_det, i_src)= squeeze(dat(:,i_det, i_src))+counts; 
		nph_det_tot(i_det, i_src)= (nph_det_tot(i_det, i_src))+nph_det;
            end
        end
        n_coll =  nph_det_tot;%sum(squeeze(dat(:,:,i_src)),1);
        if min(n_coll(:)) >= tol_ph
            srcs(srcs == i_src)=[];        
        end
    end

    disp(['lowest photon count:',num2str( min(nph_det_tot(:)) )])
    if isempty(srcs)
        flag_coll = 1;
    end

	save(backup,'nph_det_tot','dat');
end
toc
outdata = dat;

taxis = 0:cfg.tstep:cfg.tend;


end



function ex()
      cfg.nphoton=1e7;
       cfg.vol=uint8(ones(60,60,60));
       cfg.vol(20:40,20:40,10:30)=2;    % add an inclusion
       cfg.prop=[0 0 1 1;0.005 1 0 1.37; 0.2 10 0.9 1.37]; % [mua,mus,g,n]
       cfg.issrcfrom0=1;
       cfg.srcpos=[30 30 1];
       cfg.srcdir=[0 0 1];
       cfg.detpos=[30 20 1 1;30 40 1 1;20 30 1 1;40 30 1 1];
       cfg.vol(:,:,1)=0;   % pad a layer of 0s to get diffuse reflectance
       cfg.issaveref=1;
       cfg.gpuid=1;
       cfg.autopilot=1;
       cfg.tstart=0;
       cfg.tend=5e-9;
       cfg.tstep=5e-10;
       cfg.unitinmm = 1;
       % calculate the fluence distribution with the given config
       [fluence,detpt,vol,seeds,traj]=mcxlab(cfg);
 
       % integrate time-axis (4th dimension) to get CW solutions
       cwfluence=sum(fluence.data,4);  % fluence rate
       cwdref=sum(fluence.dref,4);     % diffuse reflectance
       % plot configuration and results
       subplot(231);
       mcxpreview(cfg);title('domain preview');
       subplot(232);
       imagesc(squeeze(log(cwfluence(:,30,:))));title('fluence at y=30');
       subplot(233);
       hist(detpt.ppath(:,1),50); title('partial path tissue#1');
       subplot(234);
       plot(squeeze(fluence.data(30,30,30,:)),'-o');title('TPSF at [30,30,30]');
       subplot(235);
       newtraj=mcxplotphotons(traj);title('photon trajectories')
       subplot(236);
       imagesc(squeeze(log(cwdref(:,:,1))));title('diffuse refle. at z=1');
 
end



