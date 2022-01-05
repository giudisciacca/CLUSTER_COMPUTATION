%% launch sim batch
%isim = 5;


ConN =1:730;%:100;%,'1e-1','1','2','3','4','5','6','7'};
for i = ConN
    Con{i}  = num2str(ConN(i));
end
FIT_STR = 'MeanFitted';
sumFailed = 0;
for isim = i0
    for imethod = {{'fit4param','DT_'}}%{{'fit4param',''},{'fit4param','DT_'}}%{'usprior','DT_'},{'usprior',''} }% {{'fit4param',''},{'components_fit',''}}%,{'usprior','DT'}'}
TWstr = '20';
        % down
        if contains(imethod{1}{1},'tk0')
            str{1} = ['FORWARD=1;\nREC.solver.prejacobian.load = false;\nLOAD_FWD_TEO = 0;']; 
        elseif contains(imethod{1}{1},'fit') %&& isim <=28
            str{1} = ['FORWARD=1;\nREC.solver.prejacobian.load = false;\nLOAD_FWD_TEO = 0;']; 
        TWstr='40';
	else
            str{1} = ['FORWARD=1;\nREC.solver.prejacobian.load = false;\nLOAD_FWD_TEO = 0;'];
        end
	
	
	
        tmpunderscore='';
	if strcmpi(imethod{1}{2},'DT_')
		tmpunderscore='_';
	end

        str{2} = sprintf(['EXP_DATA=0;SPECTRA=1;NUM_TW = ',TWstr,';\n',....
                'REC.solver.prior.path =[''%sPrior_VICTRE_PARADIGM%s%s''];\n',...,
                'REC.solver.type=''%s'';REC.solver.tau=0.01;\n',...
                'RECFILE_APPEND = [''PARADIGM7_%s_type'',REC.solver.type,''_tau'',num2str(REC.solver.tau,''%%%%10.0e''),''_mu0%sSample%s''];\n',...
                'RECFILE_APPEND_DATA = [''PARADIGM7_mu0%sSample%s''];\n',...
                ],...
                imethod{1}{2},tmpunderscore,Con{isim},imethod{1}{1},imethod{1}{2},FIT_STR, Con{isim}, ...
                FIT_STR, Con{isim});
         
        %str{4} = 'system([''cp Test_Standard'',RECFILE_APPEND_,''_Data.mat Test_Standard'',RECFILE_APPEND,''_Data.mat''])';
        str{3} = sprintf('DOT.opt.hete1.path =[''VICTRE_PARADIGM_%s''];\n SHOWPLOTS=0;\n',Con{isim});
        cmd = [str{1},str{2},str{3}];
        fileID = fopen(['Override_MultiSim',num2str(isim),'.m'],'w');
        fprintf(fileID,cmd);
        fclose(fileID);

        clearvars -except isim FIT_STR sumFailed Con imethod ConN
        tic
        DOT_core

        toc
        end

end
