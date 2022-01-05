function [out_projected] = project_sphere2plane(XX,x0,YY,y0,ZZ, Skull2Brain, radius,sphere_surface, values_in_volume)
% from sphere to "deformed" layer

    xy_dist2centre = sqrt( (XX - x0).^2 + (YY - y0).^2 );
    xy_dist2centre_flat = xy_dist2centre(:,:,1);
    z_dist2centre = abs(ZZ - Skull2Brain).* sphere_surface;
    xyz_dist2centre =  sum( sqrt(xy_dist2centre.^2 +z_dist2centre.^2),3);
    prov_layer_flat = sum(double(values_in_volume).* sphere_surface ,3);
    flatted_mnist(:,:) = zeros(size(XX,1), size(YY,2));
    prov_flatted = zeros(size(XX,1), size(YY,2));
    dist_x0 = 1;
    dist_y0 = 1;
    DIM = size(XX);
    for i_x = 1:size(XX,1)
        for i_y = 1:size(YY,2)
            if (xy_dist2centre_flat(i_x, i_y) <= radius)
                dist = round( radius * xy_dist2centre_flat(i_x, i_y) ./ sqrt(radius.^2 - xy_dist2centre_flat(i_x, i_y).^2) );
                theta = atan( (i_y - y0 ) / (i_x - x0 ));
                if (i_x - x0) < 0
                    theta = theta + pi;
                end
                dist_x = x0 + ceil(dist * cos(theta));                    
                dist_y = y0 + ceil(dist * sin(theta));
                if dist_x < DIM(1) && dist_x > 0 && dist_y < DIM(2) && dist_y > 0
                    if dist_x0 < DIM(1) && dist_x0 > 0 && dist_y0 < DIM(2) && dist_y0 >0 && abs(dist_x0 - dist_x) < 0.5 * DIM(1) && abs(dist_y0 - dist_y) < 0.5 * DIM(2)
                       prov_flatted(min(dist_x0, dist_x) :(max(dist_x0,dist_x)+1) , min(dist_y0, dist_y):(max(dist_y0,dist_y)+1) ) ...
                           = ...
                           prov_layer_flat(i_x, i_y); 
                    else    
                       prov_flatted(dist_x, dist_y) = prov_layer_flat(i_x, i_y);
                    end
                end
                dist_x0 = dist_x;
                dist_y0 = dist_y;
            end
        end
    end
out_projected = prov_flatted;
end