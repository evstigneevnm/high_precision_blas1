function [v] = linear_operator(x, a, re, h, N)
%[v] = linear_operator(x, a, re, h, N)
%   advection-diffusion with a>0 and re>0

    v = 0.*x;
    for j=1:N
        xp = 0;
        if(j<N)
            xp =x(j+1,1);
        end
        xm = 0;
        if(j>1)
            xm = x(j-1,1);
        end
        
        v(j,1) = a./h.*(x(j,1)-xm) - (1/re)*(xp - 2.*x(j,1) + xm)./(h.*h);
        
    end
    
    
end
