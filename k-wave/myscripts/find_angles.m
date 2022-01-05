function [y0,y1,x0, x1] = find_angles(im)
x1 = 1;
y1 = 1;
x0 = size(im,1);
y0 = size(im,2);
side = [25 25 3];
test = 255 * ones(25, 25, 3);
    for i = 1: (size(im,1) - side(1))
        for j = 1:(size(im,2) -side(2))
            extr = im(i:i + side(1) - 1 , j:j + side(2) -1 , :);
            if all(extr ~= test)
                if i <= x0
                    x0 = i;
                end
                if i >= x1
                    x1 = i+side(1);
                end
                if j <= y0
                    y0 = j;
                end
                if j  >= y1
                    y1 = j+side(2);
                end
            end
        end
    end
return;    
end
