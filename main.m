%%%%%% Author(s): Regi Kusumaatmadja (Tinbergen Institute)
%%%%%% File name: main_start_v003.m
%%%%%% Last update: 2020-12-17

%%%%%%%%%%%%%%%%
%%% SETTINGS %%%
%%%%%%%%%%%%%%%%

clc;
clear;
clear all;

global W tol_inner Kbeta Ktheta Kgamma Total
global nn TM T prods
global A share price z
global IV v

DATA    = csvread('Data.csv');

%%% 10 variables
%%% market identifier, product identifier, product market share, product
%%% attributes (3 variables), price, cost shifters (3 variables)
IDmkt      = DATA(:,1);                 % Market identifier (m in slide)
IDprod     = DATA(:,2);                 % Product identifier (j in slide)
share      = DATA(:,3);                 % Market share 
A          = DATA(:,4:6);               % Product characteristics (three vars/cols)
price      = DATA(:,7);                 % Price
z          = DATA(:,8:10);              % Instruments (three instruments)
TM         = max(IDmkt);                % Number of markets (TM = 50 markets)
prods      = zeros(TM,1);               % # of products in each market (init zero)
for m=1:TM
   prods(m,1) = max(IDprod(IDmkt==m,1));    % filling maximum # of products in each market
end
T          = zeros(TM,2);               % generate 50x2
T(1,1)     = 1;
T(1,2)     = prods(1,1);                % 2nd col filled w/ 1st # of prod in m = 1
for i=2:TM
    T(i,1) = T(i-1,2)+1;                % 1st Column market starting point
    T(i,2) = T(i,1)+prods(i,1)-1;       % 2nd Column market ending point
end
Total      = T(TM,2);                   % # of obsevations. T(50,2) = total
TotalProd  = max(prods);                % Max # of products in a given market


%% Demand-side BLP

tol_inner = 1.e-14;                         % Tolerance for inner loop (NFXP)
randn('seed',10);                           % Reset normal random number generator
rand('seed',10);                            % Reset uniform random number generator
Kbeta   = 2+size(A,2);                      % # of parameters in mean utility
Ktheta  = 1+size(A,2);                      % # of parameters with random coefficient
nn      = 200;                              % # draws to simulate shares
v       = randn(Ktheta,nn);                 % Draws for share integrals during estimation
                                            % v_0i, vki (3x1)
%%%%% Set instruments and some vars
X_GMM   = [A price];
XX_GMM  = [ones(Total,1) A price];
IV      = [ones(Total,1) A z A.^2 z.^2];    % Instruments
nIV     = size(IV,2);                       % # of instruments
W       = (IV'*IV)\eye(nIV);                % Starting GMM weighting matrix

%%%%% Starting values and for fmincon
global delta_01
delta_01        = rand(Total,1);            % (970x1), random draw uniform
x0              = rand(Kbeta+Ktheta,1);     % Starting values for all parameters
opts            = optimset('Display','iter','TolCon',1E-10,'TolFun',1E-14,'TolX',1E-14);
x_L             = [-Inf*ones(Kbeta,1);zeros(Ktheta,1)];     % Lower bounds is zero for standard deviations of random coefficients
x_U             = Inf.*ones(Kbeta+Ktheta,1);                % Upper bounds for standard deviations of random coefficients

tic
obj_fun                 = @(parameter)GMMObjFun(parameter,X_GMM,XX_GMM);
[X,~,~,~,~,~,Hessian]   = fmincon(obj_fun,x0,[],[],[],[],x_L,x_U,[],opts);
toc

%%% Results
theta1_q1   = X(1:Kbeta,1);  % (5x1) _cons, A1,A2,A3, price
theta2_q1   = X(Kbeta+1:Kbeta+Ktheta,1); % (4x1) 

%%%%% Std error
%%%%% NOT CORRECTED STD ERROR!!!!
std_error_01            = sqrt(diag(inv(Hessian)));
std_error_theta1_01     = std_error_01(1:Kbeta,1);
std_error_theta2_01     = std_error_01(Kbeta+1:Kbeta+Ktheta,1);

%%% Now, we wanted to recover the correct standard errors.
Var_cov_hat = correctedDemandSE(X_GMM, XX_GMM, theta1_q1, theta2_q1);

%%%% Fourth, compute the standard error using info from Third
SE_01           = sqrt(diag(Var_cov_hat));
SE_theta1_01    = SE_01(1:Kbeta,1);
SE_theta2_01    = SE_01(Kbeta+1:Kbeta+Ktheta,1);

theta1_trueval      = [3;3;0.5;0.5;-2];      % use -price
theta2_trueval      = [0.8;0.5;0.5;0.5];  

theta1_results      = [theta1_q1 std_error_theta1_01 SE_theta1_01 theta1_trueval];
theta2_results      = [theta2_q1 std_error_theta2_01 SE_theta2_01 theta2_trueval];

disp('theta1')
disp(['  Estimates',' ','SE','  ','Corrected SE','  ',' True Value']);
disp(theta1_results);
disp('theta2')
disp(['  Estimates',' ','SE','  ','Corrected SE','  ',' True Value']);
disp(theta2_results);

%%%%%%%%%%%%%%%%%%%%
%%%% QUESTION 2 %%%%
%%%%%%%%%%%%%%%%%%%%

%% Demand + Supply sides: BLP
%%%%% First, redefine weighting, define parameters gamma, define cap Z
W           = kron(eye(2),(IV'*IV)\eye(nIV));   % starting GMM weighting matrix, 26x26
Kgamma      = 1 + size(z,2);                    
Z           = [ones(Total,1) z];                % Making life easier for estimating mc

%%%%% Second, init fmincon
global delta_02
delta_02    = rand(Total,1);
x1          = rand(Kbeta+Ktheta+Kgamma,1);
%x0        = [0;0;0;0;0;0.01;0.01;0.01;0;0;0;0;0];
opts        = optimset('Display','iter','TolCon',1E-6,'TolFun',1E-10,'TolX',1E-10);
x_L         = [-Inf*ones(Kbeta,1); zeros(Ktheta,1); -Inf*ones(Kgamma,1)];
x_U         = Inf.*ones(Kbeta+Ktheta+Kgamma,1);

tic
aux_fun    = @(pars)GMMsupply(pars,X_GMM,XX_GMM,Z);
[X,~,~,~,~,~,hessian] = fmincon(aux_fun,x1,[],[],[],[],x_L,x_U,[],opts);
toc

%%% Results
theta1_q2   = X(1:Kbeta,1);  % (5x1) _cons, A1,A2,A3, price
theta2_q2   = X(Kbeta+1:Kbeta+Ktheta,1); % (4x1) 
gamma_q2    = X(Kbeta+Ktheta+1:end,1); % (4x1)

%%%%% Std error
%%%%% NOT CORRECTED!!!
std_error_02           = sqrt(diag(inv(hessian)));
std_error_theta1_02    = std_error_02(1:Kbeta,1);
std_error_theta2_02    = std_error_02(Kbeta+1:Kbeta+Ktheta,1);
std_error_gamma_02     = std_error_02(Kbeta+Ktheta+1:end,1);

%%% Now, we wanted to recover the correct standard errors. 
Var_cov_hat2 = correctedSupplySE(X_GMM, XX_GMM, Z, theta1_q2,theta2_q2, gamma_q2);

%%%% Sixth, compute the standard error using info from Third
SE_02           = sqrt(diag(Var_cov_hat2));
SE_theta1_02    = SE_02(1:Kbeta,1);
SE_theta2_02    = SE_02(Kbeta+1:Kbeta+Ktheta,1);
SE_gamma_02     = SE_02(Kbeta+Ktheta+1:end,1);

theta1_trueval      = [3;3;0.5;0.5;-2];      % use -price
theta2_trueval      = [0.8;0.5;0.5;0.5];    
gamma_trueval       = [5;0.5;0.5;0.5];

theta1_results      = [theta1_q2 std_error_theta1_02 SE_theta1_02 theta1_trueval];
theta2_results      = [theta2_q2 std_error_theta2_02 SE_theta2_02 theta2_trueval];
gamma_results       = [gamma_q2 std_error_gamma_02 SE_gamma_02 gamma_trueval];

disp('theta1');
disp(['  Estimates',' ','SE','  ','Corrected SE','  ',' True Value']);
disp(theta1_results);
disp('theta2');
disp(['  Estimates',' ','SE','  ','Corrected SE','  ',' True Value']);
disp(theta2_results);
disp('gamma');
disp(['  Estimates',' ','SE','  ','Corrected SE','  ',' True Value']);
disp(gamma_results);

%% Some Counterfactuals

%%% First, retrieve parameters from earlier questions
theta1_q3   = theta1_q2;    % Demand-side
theta2_q3   = theta2_q2;    % Demand-side
gamma_q3    = gamma_q2;     % Supply-side
alpha_i     = theta1_q2(Kbeta,1) + theta2_q2(Ktheta,1) * v(Ktheta,:);
alpha_i_q3  = alpha_i;      % price coefficient for each indv

%%% Declare indexing
global xi mc merged_s_jm
merged_index    = zeros(Total,1);
merged_price    = zeros(Total,1);
merged_share    = zeros(Total,1);
Delta_CS        = zeros(TM,1);
iii             = 0;

%%% Loop for each market
%%%% Sort highest market share, assign ownership
for m = 1:TM
    N_prod      = prods(m);
    [~,index]   = sort(share(T(m,1):T(m,2),:),'descend'); %%% Sort by highest market share to identify merger
    
    %%% Define ownership \Omega
    Omega                       = eye(N_prod);  % (prods(m) x prods(m))
    Omega(index(1),index(2))    = 1;
    Omega(index(2),index(1))    = 1;

    
    %%% Now, construct an indicator to identify merged products
    merged_index(iii + index(1),1)    = 1;
    merged_index(iii + index(2),1)    = 1;

    
    iii   = iii + N_prod;
    
    %%% Define characteristics of merged products
    merged_A    = A(T(m,1):T(m,2),:);
    merged_xi   = xi(T(m,1):T(m,2),:);
    merged_mc   = mc(T(m,1):T(m,2),:);
    
    %%% Define price of merged products
    initial_price   = price(T(m,1):T(m,2),1);
    aux_fun         = @(param)merger(param, theta1_q3, theta2_q3, N_prod, merged_A, merged_xi, merged_mc, alpha_i_q3, Omega);
    opts            = optimoptions('fsolve','Display','off');
    p               = fsolve(aux_fun,initial_price,opts);
    
    merged_price(T(m,1):T(m,2),:)   = p;    % Counterfactual/Merged price
    merged_share(T(m,1):T(m,2),:)   = merged_s_jm;   % Counterfactual/Merged shares
    
    %%% Re-define utility in the Baseline and Merged cases
    baseline_u  = [ones(N_prod,1) merged_A initial_price]*theta1_q3 + [merged_A initial_price]*(diag(theta2_q3)*v) + merged_xi;
    merged_u    = [ones(N_prod,1) merged_A p]*theta1_q3 + [merged_A p]*(diag(theta2_q3)*v) + merged_xi;
    
    %%% Some intermediate steps
    int_baseline_u  = sum(exp(baseline_u),1);
    int_merged_u    = sum(exp(merged_u),1);
    
    %%% Define \Delta E(CS_i)
    %%% Mean across individuals i
    %%% Minus sign such that easier to calculate/interpret welfare
    %%% Remember that alpha is negative!
    Delta_CS(m,:)   = mean( ( log(int_merged_u) - log(int_baseline_u) ) ./ -alpha_i_q3, 2 ); 
%    Delta_CS(m,:)   = mean( ( log(int_merged_u) - log(int_baseline_u) ) ./ alpha_i_q3,2 );     
end

%%% Define Profit for both cases
baseline_Pi     = (price - mc) .* share;
merged_Pi       = (merged_price - mc) .* merged_share;

%%% Define changes in price, share, and profit
Delta_price     = merged_price - price;
Delta_share     = merged_share - share;
Delta_Pi        = merged_Pi - baseline_Pi;

%%% Convert index to logicals
merged_index    = logical(merged_index);

%%% Specify changes based on the case of whether products merged or non-merged
Delta_price_m   = Delta_price(merged_index,1);
Delta_price_n   = Delta_price(~merged_index,1);
Delta_share_m   = Delta_share(merged_index,1);
Delta_share_n   = Delta_share(~merged_index,1);
Delta_Pi_m      = Delta_Pi(merged_index,1);
Delta_Pi_n      = Delta_Pi(~merged_index,1);

%%% Consumer Surplus and Welfare
Delta_ConsSurp  = sum(Delta_CS);    %%% Changes in Consumer Surplus
Delta_Profit    = sum(Delta_Pi);    %%% Changes in profit of firm f
Delta_Welfare   = Delta_ConsSurp + Delta_Profit;     %%% Changes in total welfare

%%% Some descriptive statistics
%%%%% Price
mean_price      = [mean(Delta_price) mean(Delta_price_m) mean(Delta_price_n)];
median_price    = [median(Delta_price) median(Delta_price_m) median(Delta_price_n)];
std_price       = [std(Delta_price) std(Delta_price_m) std(Delta_price_n)];
result_price    = [mean_price; median_price; std_price];
disp('Summary Statistics Price (mean;median;std)');
disp(['All','   ','Merged','    ','Non-Merged']);
disp(result_price);

%%%%% Share
mean_share      = [mean(Delta_share) mean(Delta_share_m) mean(Delta_share_n)];
median_share    = [median(Delta_share) median(Delta_share_m) median(Delta_share_n)];
std_share       = [std(Delta_share) std(Delta_share_m) std(Delta_share_n)];
result_share    = [mean_share; median_share; std_share];
disp('Summary Statistics Share (mean;median;std)');
disp(['All','   ','Merged','    ','Non-Merged']);
disp(result_share);

%%%%% Profit
mean_Pi         = [mean(Delta_Pi) mean(Delta_Pi_m) mean(Delta_Pi_n)];
median_Pi       = [median(Delta_Pi) median(Delta_Pi_m) median(Delta_Pi_n)];
std_Pi          = [std(Delta_Pi) std(Delta_Pi_m) std(Delta_Pi_n)];
result_Pi       = [mean_Pi; median_Pi; std_Pi];
disp('Summary Statistics \Pi (mean;median;std)');
disp(['All','   ','Merged','    ','Non-Merged']);
disp(result_Pi);

%%%%% CS, Profit, Welfare
result_Welfare  = [Delta_ConsSurp Delta_Profit Delta_Welfare];
disp(['Consumer Surp.','    ','Profit','    ','Welfare'])
disp(result_Welfare);
