function [z] = precond(r, a, re, h, N)
%[z] = precond(r, a, re, h, N)
%   advection-diffusion preconditioner with a>0 and re>0

    
    z = r./(a./h - (1./re).*(-2)./(h.*h));
    
%     for j=1:N
%         xp = 0;
%         if(j<N)
%             xp =x(j+1,1);
%         end
%         xm = 0;
%         if(j>1)
%             xm = x(j-1,1);
%         end
%         
%         z(j,1) = a./h.*(x(j,1)-xm) - (1/re)*(xp - 2.*x(j,1) + xm)./(h*h);
%         
%     end
    
end