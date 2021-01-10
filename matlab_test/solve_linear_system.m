function [x, flag, relres, iter, resvec] = solve_linear_system(a, re, N, tol, maxit, restart)
%[x, flag, relres, iter, resvec] = solve_linear_system(a, re, N, tol, maxit, restart)
%   solves the linear system using gmsres method
    if nargin == 5
        restart = 0;
    end

    h = 1.0/N;
    b = ones(N,1);
    
    if nargin == 6
        [x, flag, relres, iter, resvec] = gmres(@(x)linear_operator(x, a, re, h, N), b, restart, tol, maxit, @(r)precond(r, a, re, h, N));
    else
        [x, flag, relres, iter, resvec] = bicgstabl(@(x)linear_operator(x, a, re, h, N), b, tol, maxit, @(r)precond(r, a, re, h, N));
    end
    fprintf("actual residual = %e\n", norm(linear_operator(x, a, re, h, N) - b) );

end

