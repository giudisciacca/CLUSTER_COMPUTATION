function L = assembleLaplacian3(kappa)
    n1 = size(kappa,1);
    n2 = size(kappa,2);
    n3 = size(kappa,3);
    
    D1dz = spdiags([-ones(n3,1), ones(n3, 1)], -1:0, n3,n3);
%    D1dz(1, 1) = 1;  
%    D1dz(1, n3) = 1;%D1dz(n3, 1) - 1;  
%    D1dz(n3, 1) = 1;
    D1dy = spdiags([-ones(n2,1), ones(n2, 1)], -1:0, n2,n2);
    D1dy(1, 1) = 1;
    D1dy(1, n2) = 1;%D1dy(n2 ,1) - 1;
    %D1dy(n2, 1) = 1;
    D1dx = spdiags([-ones(n1,1), ones(n1, 1)], -1:0, n1,n1);
    D1dx(1, 1) = 1;
    D1dx(1, n1) = 1;
    %D1dx(n1, 1) = D1dx(n1, 1) - 1;
    
    D1z3d = kron(kron(D1dz, speye(n2)), speye(n1));
    D1y3d = kron(kron(speye(n3),D1dy),speye(n1));
    D1x3d = kron(speye(n3),kron(speye(n2), D1dx));
    
   
    kap1d = spdiags(reshape(kappa,[],1), 0:0, n1*n2*n3, n1*n2*n3);
    %kap1x = spdiags(intx2d*kap1d,0:0,(n1+1)*n2,(n1+1)*n2);
    %kap1y = spdiags(inty2d*kap1d,0:0,n1*(n2+1),n1*(n2+1));
    %kap1z = spdiags(inty2d*kap1d,0:0,n1*(n2+1),n1*(n2+1));
    
    L = -(D1x3d'*kap1d*D1x3d + D1y3d'*kap1d*D1y3d + D1z3d'*kap1d*D1z3d); 
   % L = -(D1x3d'*kap1x*D1x3d + D1y3d'*kap1y*D1y3d + D1z3d'*kap1z*D1z3d);
end