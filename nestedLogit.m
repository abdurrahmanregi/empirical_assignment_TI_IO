%%%%%% Author(s): Regi Kusumaatmadja (Tinbergen Institute)
%%%%%% File name: main_start_v003.m
%%%%%% Last update: 2020-12-17

clc;
clear all;

DATA        = csvread('nestedData.csv');
IDmkt       = DATA(:,1);                 % Market identifier
IDprod      = DATA(:,2);                 % Product identifier
share       = DATA(:,3);                 % Market share 
A           = DATA(:,4:6);               % Product characteristics (three vars/cols)
price       = DATA(:,7);                 % Price
z           = DATA(:,8:10);              % Instruments (three instruments)
group1      = DATA(:,11);                % Group identifier for Nested Logit
group2      = DATA(:,12);
group3      = DATA(:,13);
group4      = DATA(:,14);
group5      = DATA(:,15);
TM          = max(IDmkt);                % Number of markets (TM = 50 markets)
prods       = zeros(TM,1);               % # of products in each market (init zero)
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
Ngroups1    = max(group1);                % # of groups
Ngroups2    = max(group2);
Ngroups3    = max(group3);
Ngroups4    = max(group4);
Ngroups5    = max(group5);

%% Nested Logit
productSums_share = accumarray(IDmkt,share);
share_outside = 1 - productSums_share(IDmkt);     % define S_0m (970 x 1)
delta = log(share) - log(share_outside);        % define delta (970 x 1)

for i=1:Total
    group1_share(i,1) = sum( share( (IDmkt==IDmkt(i,1))&(group1==group1(i,1)) ,1) ,1);
    group2_share(i,1) = sum( share( (IDmkt==IDmkt(i,1))&(group2==group2(i,1)) ,1) ,1);
    group3_share(i,1) = sum( share( (IDmkt==IDmkt(i,1))&(group3==group3(i,1)) ,1) ,1);
    group4_share(i,1) = sum( share( (IDmkt==IDmkt(i,1))&(group4==group4(i,1)) ,1) ,1);
    group5_share(i,1) = sum( share( (IDmkt==IDmkt(i,1))&(group5==group5(i,1)) ,1) ,1);
end

%%%% Define log(s_j | g(j) m)
within_group1 = share./group1_share;
within_group2 = share./group2_share;
within_group3 = share./group3_share;
within_group4 = share./group4_share;
within_group5 = share./group5_share;
log_within_gr1 = log(within_group1);
log_within_gr2 = log(within_group2);
log_within_gr3 = log(within_group3);
log_within_gr4 = log(within_group4);
log_within_gr5 = log(within_group5);

X_IV    = [ones(Total,1) A];    % Define X_jm (970 x 4)
IV      = [X_IV z];             % Define IV_over (970 x 7); o = over-identified
Y_NL1   = [price log_within_gr1];
Y_NL2   = [price log_within_gr2];
Y_NL3   = [price log_within_gr3];
Y_NL4   = [price log_within_gr4];
Y_NL5   = [price log_within_gr5];

XX_NL1  = [ones(Total,1) A -price log_within_gr1];
XX_NL2  = [ones(Total,1) A -price log_within_gr2];
XX_NL3  = [ones(Total,1) A -price log_within_gr3];
XX_NL4  = [ones(Total,1) A -price log_within_gr4];
XX_NL5  = [ones(Total,1) A -price log_within_gr5];

W_IV    = inv(IV'*IV);
PZ_IV   = IV*W_IV*IV';

%%% Group 1
NL_XIVhat_1     = PZ_IV*XX_NL1;
NL_beta_g1      = (NL_XIVhat_1'*NL_XIVhat_1)\(NL_XIVhat_1'*delta);
se_g1           = sqrt(diag(mean((delta-XX_NL1*NL_beta_g1).^2)*((XX_NL1'*PZ_IV*XX_NL1)\eye(size(XX_NL1,2)))));
str_g1          = [NL_beta_g1 se_g1]

%%% Group 2
NL_XIVhat_2     = PZ_IV*XX_NL2;
NL_beta_g2      = (NL_XIVhat_2'*NL_XIVhat_2)\(NL_XIVhat_2'*delta);
se_g2           = sqrt(diag(mean((delta-XX_NL2*NL_beta_g2).^2)*((XX_NL2'*PZ_IV*XX_NL2)\eye(size(XX_NL2,2)))));
str_g2          = [NL_beta_g2 se_g2]

%%% Group 3
NL_XIVhat_3     = PZ_IV*XX_NL3;
NL_beta_g3      = (NL_XIVhat_3'*NL_XIVhat_3)\(NL_XIVhat_3'*delta);
se_g3           = sqrt(diag(mean((delta-XX_NL3*NL_beta_g3).^2)*((XX_NL3'*PZ_IV*XX_NL3)\eye(size(XX_NL3,2)))));
str_g3          = [NL_beta_g3 se_g3]

%%% Group 4
NL_XIVhat_4     = PZ_IV*XX_NL4;
NL_beta_g4      = (NL_XIVhat_4'*NL_XIVhat_4)\(NL_XIVhat_4'*delta);
se_g4           = sqrt(diag(mean((delta-XX_NL4*NL_beta_g4).^2)*((XX_NL4'*PZ_IV*XX_NL4)\eye(size(XX_NL4,2)))));
str_g4          = [NL_beta_g4 se_g4]

%%% Group 5
NL_XIVhat_5     = PZ_IV*XX_NL5;
NL_beta_g5      = (NL_XIVhat_5'*NL_XIVhat_5)\(NL_XIVhat_5'*delta);
se_g5           = sqrt(diag(mean((delta-XX_NL5*NL_beta_g5).^2)*((XX_NL5'*PZ_IV*XX_NL5)\eye(size(XX_NL5,2)))));
str_g5          = [NL_beta_g5 se_g5]


