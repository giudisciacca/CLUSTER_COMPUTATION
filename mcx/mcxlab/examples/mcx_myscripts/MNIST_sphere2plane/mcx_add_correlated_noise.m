function [domain, coefficient] = mcx_add_correlated_noise(domain, avg_coefficient, start_index, end_index, l_corr, perc_noise)

% [coefficient, domain] = add_correlated_noise(domain, avg_coefficient, start_index, end_index, l_corr, perc_noise)
% % Working example
% %     domain = cat(3,3*ones(10,10,5), zeros(10,10,5));
% %     avg_coefficient = 0.3;
% %     start_index = 3;
% %     end_index = 10;
% %     l_corr = 2;
% %     perc_noise = 0.2;
    
    % define domain of noise
    if start_index == end_index
        coefficient = avg_coefficient;
        domain = domain .* start_index;
        return;
    end
    
    q_domain = double(domain == start_index);    
    nan_domain = q_domain;
    nan_domain(q_domain==0) = nan;
        
    [Nx, Ny, Nz] = size(domain);
    [xx,yy, zz] = meshgrid(0:Nx-1,0:Ny-1,0:Nz-1);
    %generate noise over the whole domain    
    noise = randn(size(domain));    
    gaus = exp( - ((xx-0.5*(Nx-1)).^2 + (yy-0.5*(Ny-1)).^2 + (zz-0.5*(Nz-1)).^2 )/l_corr^2);    
    mus = avg_coefficient * ones(size(domain));              
    mus0 = ifftn(((fftn(gaus)) .* (fftn(noise))));
    mus0 = mus0 - mean(mus0(:));
    mus0 = perc_noise  * (mus0/(max(mus0(:)) - min(mus0(:))));
    mus = abs(mus + mus0) +  0.001 * ((mus + mus0)<=0.001);
    mus = mus .* (mus > 0); 
    q_mus = q_domain .* (mus);    
    nan_mus = nan_domain .* (mus);
    
    
    codebook = linspace(1.01* min(nan_mus(:)) , 0.99 * max(nan_mus(:)), numel(start_index:1:end_index)+1)'; 
    partition = linspace(min(nan_mus(:)), max(nan_mus(:)), numel(codebook)-1)';
    [indexes, quants] = quantiz(q_mus(:), partition, codebook);
    
    indexes = q_domain .* reshape(indexes, size(domain)); 
    quants = q_domain .* reshape(quants, size(domain));
    %quants = q_domain .* reshape(codebook(indexes+1), size(domain));
    
    
    domain = domain + indexes;
    domain = domain .* (q_domain  > 0);
%    coefficient = unique(quants);
    coefficient = codebook(1:end-1);
    
end
