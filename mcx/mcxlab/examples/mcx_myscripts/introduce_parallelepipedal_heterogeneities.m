function out_vol = introduce_parallelepipedal_heterogeneities(VOL, DOMAIN, NUMBER, SIZE_LIMITS, VALUE)
% 
% Introduces parallelepipedal heterogeneities as a binary mask in a defined
% domain of the mask VOL

x_max = DOMAIN(1,2);
x_min = DOMAIN(1,1);
y_max = DOMAIN(2,2);
y_min = DOMAIN(2,1);
z_max = DOMAIN(3,2);
z_min = DOMAIN(3,1);

lx_max = SIZE_LIMITS(1,2);
lx_min = SIZE_LIMITS(1,1);
ly_max = SIZE_LIMITS(2,2);
ly_min = SIZE_LIMITS(2,1);
lz_max = SIZE_LIMITS(3,2);
lz_min = SIZE_LIMITS(3,1);


out_vol = VOL;
indices = zeros(NUMBER, 3, 2);
i = 1;
while i <= NUMBER
   mask_cand = zeros(size(VOL)); 
   lx_cand = lx_min + randi(lx_max - lx_min);
   ly_cand = ly_min + randi(ly_max - ly_min);
   lz_cand = lz_min + randi(lz_max - lz_min);
   
   x_pos0 = x_min + randi(x_max - x_min - lx_cand);
   x_pos1 = x_pos0 + lx_cand;
   y_pos0 =  y_min + randi(y_max - y_min - ly_cand);
   y_pos1 = y_pos0 + ly_cand;
   z_pos0 =  z_min + randi(z_max - z_min - lz_cand);
   z_pos1 = z_pos0 + lz_cand;
   
   mask_cand(x_pos0:x_pos1,y_pos0:y_pos1,z_pos0:z_pos1) = VALUE;
   
   not_free = check_if_already_occupied(out_vol, mask_cand ~= 0);
   if not_free == 0
    indices(i,:,:) =  [x_pos0,x_pos1;y_pos0,y_pos1;z_pos0,z_pos1] ; 
    out_vol = out_vol + mask_cand;
    i = i + 1;
   end
end

return;

end

function  out_flag = check_if_already_occupied(VOL, VOL_to_check)
% if portion of space is already occupied it gives 1, 0 otherwise
bin_vol = (VOL >= 1 ); % if any value is bigger than 1 then set to one
bin_vol = bin_vol(:); %vectorise
VOL_to_check = VOL_to_check(:);
to_test = VOL_to_check(:) + bin_vol(:);
out_flag = any(to_test >= 2);

end
