 function out = reshape_USimage(im, REC, bb)
% bb is the size of the images in mm*10^-1
% x is the vertical axis
 
    bbx = bb(1);%mm
    bby = bb(2);
    
    
    bbx = bbx;
    bby = bby;
    
    im_resized = imresize(im, size(im)*10*REC.grid.dx,'nearest');
    
    endx = size(im_resized,1);
    endy = size(im_resized,2);
    
    %out = zeros(min(endx,bbx), min(endy,bby));
    out = zeros(bbx, bby);
    
    idx = 1:min(endx, bbx);% round(floor(endx/2)-bbx/2:floor(endx/2)+bbx/2);
    idy = round((endy/2)-bby/2+1:(endy/2)+bby/2);
    
    outx = 1:bbx;
    outy=1:bby;
    
    if numel(idx) < bbx
        outx = 1:endx;
    end
    if numel(idy) < bby
        outy = round(floor(endy/2)-bby/2:floor(endy/2)+bby/2);
    end
        
    %out = zeros(max(size(im_resised,1),bbx),max(size(im_resised,2),bby));
    
    %if size(im_resised,1)
    out = im_resized(idx,idy);
    %out(outx,outy) = size(im_resized(idx,idy))

end