function surface3D = interpolateSurface(surface3D, dims,[XX, YY, ZZ], values)

FACT = 10;
%%  lauch example
[XX, YY,ZZ] = meshgrid(-30:1:30, -30:1:30, -30:1:30 );
surface3D =  (XX.^2 + YY.^2 + ZZ.^2 <=  30^2) .* ZZ>=0;
surface3D = cat(3,zeros(size(XX,1), size(XX,2)), diff(surface3D,1,3));

% from volume to nodes. It gets z(x,y)
ZZ = squeeze(sum(ZZ .*(surface3D ~= 0),3));
%XX = XX(surface3D ~= 0);
%YY = YY(surface3D ~= 0);
%values = values .* (surface3D ~= 0);

% calculate distance for each pixel
lx = sqrt(1 + ( ( ...
                    squeeze( ZZ(:,2:end) - ZZ(:,1:end-1) )  ./ ... 
                    squeeze( XX(:,2:end,1) - XX(:,1:end-1,1) ) ).^2) );
ly = sqrt(1 + ( ( ...
                    squeeze( ZZ(2:end,:) - ZZ(1:end-1,:) )  ./ ... 
                    squeeze( YY(2:end,:,1) - YY(1:end-1,:,1) ) ).^2) );


sum_lx = max(sum(lx,1));
sum_ly = max(sum(ly,2)); 

% maps to higher dimensions
higher_lx = FACT * lx;
higher_ly = FACT * ly;
higher_map =  zeros(FACT * [sum_lx, sum_ly]);

higher_coorx = cumsum(lx);
higher_coory = cumsum(ly);


to_interp = value(i:i+1, j:j+1);
interpolated( higher_coorx(i):higher_coorx(i) + ext_x, higher_coory(j):higher_coorx(j):ext_y) = intep2(2,2,to_interp, ext_x, ext_y);


% interpolate to lower dimension

%[XXf, YYf, ZZf] = meshgrid(-30:0.1:30,-30:0.1:30,-30:0.1:30);

% map to finer 2D  mesh
Vq = interp3(XX, YY, ZZ, values, XXf(:), YYf(:), ZZf(:));

% interpolate back to dims




    function createMesh(surface, nodes)
    % creates
    
    
    end
end




