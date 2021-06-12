function [ I, x_hat, yrn ] = OMP( y, A, Kmax )
[~, N] = size(A);

r = y;
I = [];
for l1 = 1:Kmax
    r_old = r;
    I_old = I;
    [~, index] = max(abs(A'*r));
    %I = [I, index];
    I = union(I, index);
    xp = A(:,I)\y;
    r = y - A(:,I)*xp;
    if(norm(r) >= norm(r_old))
        I = I_old;
        r = r_old;
        break;
    end
end

x_hat = zeros(N,1);
xp = A(:,I)\y;
x_hat(I) = xp;
yrn=norm(r);
end