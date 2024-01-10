function [errors,price] = ExtrapSOR(s0,E,r,sigma,T,Smax,M,N,omega,tol)  
%
%%  Extrapolation-based SOR and finite difference methods for pricing 
%%  European put options    
%%
%  Author: Foued Saâdaoui, Date: 10-02-2024                               %
%  Cite: Accelerated Solutions for Discretized Black-Scholes Equations:   %
%  Numerical Experiments and Case Study, IMA J. Management Mathematics    %
%% INPUTS:
%%
% s0:     Initial Spot Price
% E:      Option's Strike
% r:      Instantaneous Risk-Free Rate
% sigma:  Black-Scholes Volatilityclear
% T:      Option's Maturity 
% Smax:   Maximum Value for the Space Grid (Twice/Thrice E)
% M:      Space Discretization Level
% N:      Time Discretization Level
% omega:  SOR Relaxation Parameter
% tol:    Tolerance
%% OUPUTS:
%%
% errors: Iteration Errors
% price:  Option Price
%
%% Example: 
%  s0=1498.06; E=2150; r=0.01543; sigma=0.055; T=0.1917;                  %                 
%  Smax=4300; M=250; N=250; omega=0.9; tol=10^-5;                         %
%%
tic
optiv=zeros(N+1,M+1);
dt=T/N;
ds=Smax/M;
ves=0:ds:Smax;
vet=T:-dt:0;
optiv(1,:)= max(E-ves,0); 
optiv(:,1)= E*exp(-r*(T-vet)); 
optiv(:,M+1)= 0; 
% Form the Tridiagonal Matrix
a = @(j) 0.5*dt*(r*j-sigma^2*j^2);
b = @(j) 1 + (sigma^2*j^2 + r)*dt;
c = @(j) -0.5*dt*(sigma^2*j^2+r*j);
acoeffs = zeros(M+1,1);bcoeffs = zeros(M+1,1);ccoeffs = zeros(M+1,1);
for j=1:M+1
acoeffs(j) = a(j-1);
bcoeffs(j) = b(j-1);
ccoeffs(j) = c(j-1);
end
Tri=diag(acoeffs(3:M),-1)+diag(bcoeffs(2:M))+diag(ccoeffs(2:M-1),+1);
% Successive over Relaxation 
for p=1:N
aux=zeros(M-1,1);
aux(1)=-a(0)*optiv(p+1,1);
aux(end)=-c(M)*optiv(p+1,M+1);
RHS=optiv(p,2:M)'+aux;
% Iteratively Solving the Linear System Ax=b 
A=Tri;b=RHS;
% Define x
x=optiv(p,2:M)'; 
% Initialize Vectors before Entering the loop
xold=100*x; 
x1=xold;   
x2=x1;       
n=length(x);
% Initialize k 
k=0;  
errors=zeros(); 
while norm(xold-x)>tol
xold=x; 
    for i=1:n % Calculating x1
    if i==1
    z=(b(i)-A(i,i+1)*x(i+1))/A(i,i);
    x1(i) = omega*z + (1-omega)*xold(i);
    elseif i==n
    z=(b(i)-A(i,i-1)*x(i-1))/A(i,i);
    x1(i) = omega*z +(1-omega)*xold(i);
    else
    z=(b(i)-A(i,i-1)*x(i-1)-A(i,i+1)*x(i+1))/ A(i,i);
    x1(i) = omega*z +(1-omega)*xold(i);
    end
    end
    for i=1:n % Calculating x2
    if i==1
    z1=(b(i)-A(i,i+1)*x(i+1))/A(i,i);
    x2(i) = omega*z1+(1-omega)*x1(i);
    elseif i==n
    z1=(b(i)-A(i,i-1)*x(i-1))/A(i,i);
    x2(i) = omega*z1 + (1-omega)*x1(i);
    else
    z1=(b(i)-A(i,i-1)*x(i-1)-A(i,i+1)*x(i+1))/ A(i,i);
    x2(i) = omega*z1 +(1-omega)*x1(i);
    end
    end
 Delta1 = x1-xold;
 Delta2 = x2-2*x1+xold;
 alpharre = dot(Delta1,Delta2)/dot(Delta2,Delta2);
 alphampe = dot(Delta1,Delta1)/dot(Delta1,Delta2);
 omeg=sqrt(abs(alpharre/alphampe));
 alphahyb=(omeg*alphampe)+((1-omeg)*alpharre);
 x=xold-(2*rand)*alphahyb*Delta1;
% x=xold-alpharre*Delta1;
 k=k+1;
 errors(k)=norm(xold-x);
end
optiv(p+1,(2:end-1))=x;
end
%% GRAPHICAL REPRESENTATION
subplot(2,1,1)
mesh(ves,vet,optiv)
xlabel('$S$','interpreter','latex');
ylabel('$t$','interpreter','latex');
zlabel('$V(S,t)$','interpreter', 'latex')
subplot(2,1,2)
plot(log10(errors))
xlabel('iterations');ylabel('log-residuals')
% Defining the Adjacent Value to s0 on the Grid and Calculating 
% the Price using Interpolation
price=interp1(ves,optiv(N+1,:),s0);
toc
