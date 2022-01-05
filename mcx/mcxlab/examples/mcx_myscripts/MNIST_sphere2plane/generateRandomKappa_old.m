function kappa = generateRandomKappa(number, depth, avg)
    
    [xx,yy, zz] = meshgrid(-29.5:29.5,-29.5:29.5, 0:depth-1);
    l_corr = 15;
    gaus = exp( - (xx.^2 + yy.^2 + zz.^2 )/l_corr^2);
    
    
    kappa = avg * ones(60,60,depth,number);
    perc = 0.8 * kappa(1);
    for i = 1:number
        noise = randn([60,60, depth]);        
        kappa0 = ifftn(((fftn(gaus)) .* (fftn(noise))));
        kappa0 = kappa0 - mean(kappa0(:));
        kappa0 = perc  * (kappa0/(max(kappa0(:)) - min(kappa0(:))));
        kappa(:,:,:,i) = kappa(:,:,:,i) + kappa0;
    end
end