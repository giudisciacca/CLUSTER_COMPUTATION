%==========================================================================
%%                      RECONSTRUCTION DOMAIN: CW or TD
%==========================================================================
REC.domain = 'td';          % CW or TD: data type to be inverted
REC.type_fwd = 'linear';    % 'linear' or 'fem'.
% -------------------------------------------------------------------------
REC.time.roi = 'auto';% [282 946;264 941;241 942;232 946;222 946;218 942;213 941;218 944];
REC.time.roiRange1 = 0.95;
REC.time.roiRange2 = 0.1;

                        % selected dinamically by the user.
NUM_TW = 80;            % Number of Time Windows within ROI
% =========================================================================
%%                        Initial parameter estimates 
% =========================================================================
% In this section all the parameter for the inverse solver are setted.
% --------------------------- Optical properties --------------------------
REC.solver.variables = {'mua','mus'}; % variables mua,mus.
REC.opt.mua0 = 0.1;    % absorption [mm-1]
REC.opt.musp0 = 1.0;      % reduced scattering [mm-1]
REC.opt.nB = 1.4;
if SPECTRA == 0
mua_ = [ 0.0027    0.0021    0.0027    0.0072    0.0129    0.0266    0.0145    0.0087];
musp_=[  1.0512    0.9444    0.7958    0.8955    0.8239    0.8449    0.7186    0.9421];
mua_ = mua_(lamda_id); musp_ = musp_(lamda_id);
Xr = {mua_,musp_};
else
a_ = 0.789550;	b_ =1.05756;
conc_ = [9.50498	0.92704	0.15076	0.63606	1.11337];
Xr = {conc_,[a_ b_]};
end

% ---------------------- Solver and regularization ------------------------
REC.solver.tau = 0.005;            % regularisation parameter
REC.solver.fit_reference = true;% true;
REC.solver.fit_reference_far = false;
REC.solver.fit_reference_tw = false;
REC.solver.fit_reference_fwd = 'linear';
REC.solver.type = 'USprior';         % 'born','GN': gauss-newton, 
                                  % 'USprior': Simon's strutural prior
                                  % 'LM': Levenberg-Marquardt,
                                  % 'l1': L1-based minimization
                                  % 'fit': fitting homogeneous data
                                  % 'spectral_fit': fitting homogeneous
                                  % data with spectral model
                                  % 'spectral_usprior': spectral approach
                                  % to USprior
                                  % 'spectral_born': recon with spectral
                                  % Jacobian
                                  % 'born_spectral_post_proc': multi_wave
                                  % classical born with post proc
                                  % chromophores estimation
if strcmpi(REC.solver.type,'spectral_born')&&SPECTRA == 0
MEx = MException('spectral_born:SpectralDataInput','Set SPECTRA = 1 to use spectra_born');
throwAsCaller(MEx);    
end
% =========================================================================
%%                            US prior 
% =========================================================================
REC.solver.prior.path = ['priorVICTRE.mat'];%['DTsilicon_mask.mat'];%;['ham_maskYX_flip.mat'];
% =========================================================================
%%                     load a precomputed jacobian 
% =========================================================================
% Pay attention! The jacobian depends on source-detectors configuration,
% optical properties of the background and number of time-windows.
REC.solver.prejacobian.load = false;
REC.solver.prejacobian.path = 'precomputed_jacobians/J';
