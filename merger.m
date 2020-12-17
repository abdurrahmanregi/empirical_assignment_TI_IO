function f = merger(initial_price, theta1_q3, theta2_q3, N_prod, merged_A, merged_xi, merged_mc, alpha_i_q3, Omega)

global nn v merged_s_jm

%%% Define \delta and \mu for the utility
delta   = [ones(N_prod,1) merged_A initial_price]*theta1_q3 + merged_xi ;
mu      = [merged_A initial_price] * (diag(theta2_q3) * v);    

%%% Numerator and denominator
numerator       = exp(repmat(delta,1,nn) + mu);
denominator     = 1 + sum(numerator);
merged_s_ijm       = numerator./denominator;
merged_s_jm        = mean(merged_s_ijm,2);

%%% Define the partial derivatives (denominator of markup)
merged_denom_mkup  = zeros(N_prod, N_prod);
for i = 1:N_prod
    for j = 1:N_prod
        if i == j
            %%% In case of OWN product
            int_denom   = alpha_i_q3 .* merged_s_ijm(i,:) .* ( 1 - merged_s_ijm(i,:) ); 
            merged_denom_mkup(i,j)  = mean(int_denom,2);
        else
            %%% In case of CROSS product
            int_denom   = - alpha_i_q3 .* merged_s_ijm(i,:) .* merged_s_ijm(j,:);
            merged_denom_mkup(i,j)  = mean(int_denom,2);
        end
    end
end

%%% Define Omega_star 
Omega_star  = Omega .* (- merged_denom_mkup);

%%% f := s_m - \Omega_m [p_m - mc_m] = 0
f           = merged_s_jm - Omega_star*(initial_price - merged_mc);

end