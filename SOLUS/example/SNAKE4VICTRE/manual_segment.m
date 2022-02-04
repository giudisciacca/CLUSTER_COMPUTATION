% manual segment
FORMAT = 'DICOM'
CROP =1;
%% create

close all
loadname = filename;
    
    if strcmpi(FORMAT, 'DICOM') == 1 
        im = dicomread(loadname);
        info = dicominfo(loadname);

        if CROP == 1
            xmin = info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0;
            xmax = info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1;
            ymin = info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0;
            ymax = info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1;
            im = im(ymin:ymax, xmin:xmax, :);
        end

        % find delta
        delta_x = info.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaX;
        delta_y = info.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaY;
        unx = info.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsXDirection; % gives in which units physical delta x is expressed 
        uny = info.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsYDirection;


        if unx == 3
            delta_x = delta_x * 10; % transform deltax from cm/px to mm/px
        else
            disp('Error: see dicom dictionary to know what units are assigned to delta_x')
        end

        if uny == 3
            delta_y = delta_y * 10;
        else
            disp('Error: see dicom dictionary to know what units are assigned to delta_x')
        end

        if delta_x ~= delta_y

            disp('DICOM image has problems with calibrations: pixels are not squares')
            delta = 0.5 * (delta_x + delta_y);
        else
            delta = delta_x;

        end

    end

    [segmented, cor] = pointSplineSegs(im);
 
    save(['COR_',loadname(1:end-4)],'segmented','cor','im','delta')


    

    loadname = (['COR_',loadname(1:end-4)])
    load(loadname)
    
    %% Snake
    [sgm, cor_updated ] = snake_fitting(im, cor);
    figure(1001);imshow(cat(3,im(:,:,1),200*sgm,200*segmented))
    save(['SNAKE_',loadname],'im','cor','sgm','segmented','cor_updated')
    dec = input('good? ');
    if dec == 1% in[ut ok
        i = i+1;
    end
    %% Extrusion
    disp('Extrusion...');
    mask3D = retrieve_ellipsoid(sgm);
    %% SAVE
    mask3D = logical(mask3D(:,:,1:min(500, size(mask3D,3))));
   save(out_filename, 'mask3D', 'delta');

    
