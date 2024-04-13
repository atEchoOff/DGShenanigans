format longE

% Degree of polynomial 
tic
N = 3;

% Number of elements
M = 2048

% Time
TIME = 20480;

h = 0.7/TIME;

% Interval
a = -1;
b = 1;

dist = (b - a)/M;


[P, roots, weights] = gausslobotto(N);


D = diffM(P, N, roots);
L = L_create(N);
norm_direc = norm_create(M);
nodes = nodes_create(N, M, a, b, roots);

u = exp(-10*nodes.^2);
toc
%%
tic


for i = 1:60
    u = u+ h * rhs(u, norm_direc, L, weights, D, dist);
    % %%
    % if (mod(i, 10) == 0) 
    %     scatter(nodes, u, "filled");
    %     axis([a b 0 1.5]);
    %     drawnow limitrate
    % end
end
toc

%% RK4 Method
h = 0.001;
for i = 1:TIME
    k1 = rhs(u, norm_direc, L, weights, D, dist);
    k2 = rhs(u + h/2 * k1, norm_direc, L, weights, D, dist);
    k3 = rhs(u + h/2 * k2, norm_direc, L, weights, D, dist);
    k4 = rhs(u + h* k3, norm_direc, L, weights, D, dist);

    u = u + h/6 * (k1 + 2*k2 + 2*k3 + k4);
    if (mod(i, 10) == 0) 
        scatter(nodes, u, "filled");
        axis([a b 0 1.5]);
        drawnow limitrate
    end
end


function [P, roots, weights] = gausslobotto(N)
    syms x;
    P = diff(legendreP(N, x));
    roots = double(vpasolve(P == 0));
    roots = [-1;roots; 1];
    
    % weights;
    P1 = legendreP(N, x);
    for i = 1:N+1
        x = (roots(i));
        weights(1, i) = 2/(N+1)/(N)/ (subs(P1))^2;
    end
end

function D = diffM(P, N, roots) 
    D = zeros(N+1, N+1);
    psi = zeros(1, N+1);
    Dpsi = zeros(1, N+1);
    DDpsi = zeros(1, N+1);
    DP = diff(P);
    DDP = diff(DP);
    
    for i = 1:N+1
        x = roots(i);
        psi(i) = (x^2 - 1) * subs(P);
        Dpsi(i) = (x^2 - 1) * subs(DP) + ...
            2 * x * subs(P);
        DDpsi(i) = (x^2 - 1) * subs(DDP) + ...
            4 * x * subs(DP) + 2 * subs(P);
    end
    
    for i = 1:N+1
        for j = 1:N+1
            if (i ~= j) 
                D(j, i) = 1/(roots(j) - roots(i))^2 / ...
                    Dpsi(i) * (Dpsi(j) * ...
                    (roots(j) - roots(i)) -psi(j));
            else
                D(j, i) = DDpsi(i)/Dpsi(i)/2;
            end
        end
    end
end

function L = L_create(N)
    L = zeros(N+1, 2);
    L(1, 1) = 1;
    L(end, end) = 1;
    
end

function norm_direc = norm_create(M)    
    norm_direc = [-ones(1, M); ones(1, M)];
end

function nodes = nodes_create(N, M, a, b, roots)
    dist = (b - a)/M;
    nodes = zeros(N+1, M);
    for i = 1:M
        nodes(:, i) = (dist*(i-1) + a) + (1 + roots)/2 * dist;
    end
end

function out = rhs(u, normal_direc, L, weights, D, dist)
   
    uM = [u(1, :); u(end, :)];
    uP = [uM(1, 1) uM(2, 1:end-1); uM(1, 2:end) uM(end, end)];
    
    fuM = (uM.^2)/2;
    
    favg = 0.5 * (fuM + (uP.^2)/2);
    
    out = double((-2/dist) * (D * (u.^2)/2 + diag(1./weights) * L * ...
        ((favg - fuM) .* normal_direc - 0.5 * (uP - uM))));
end



        
