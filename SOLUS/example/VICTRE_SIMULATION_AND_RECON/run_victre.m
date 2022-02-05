imethod = {{'fit4param',''}}
       
str{1} = ['FORWARD=1;\nREC.solver.prejacobian.load = false;\nLOAD_FWD_TEO = 0;']; 
       
str{2} = sprintf(['EXP_DATA=0;SPECTRA=1;NUM_TW = 80;\n',....
                'REC.solver.prior.path =[''PATH/TO/GROUND/TRUTH''];\n',...,
                'REC.solver.type=''%s'';\n',...
                ],...
                imethod{1}{2},Con{isim},imethod{1}{1},imethod{1}{2},FIT_STR, Con{isim}, ...
                FIT_STR, Con{isim});
         
str{3} = sprintf('DOT.opt.hete1.path =[''PATH/TO/PRIOR''];\n SHOWPLOTS=0;\n',Con{isim});
cmd = [str{1},str{2},str{3}];
fileID = fopen('Override_MultiSim.m','w');
fprintf(fileID,cmd);
fclose(fileID);
clearvars -except isim FIT_STR sumFailed Con imethod ConN
tic
DOT_core
toc
