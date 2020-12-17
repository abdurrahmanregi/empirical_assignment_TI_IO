function f = GMMsupply(x1,X_GMM,XX_GMM,Z)

global W tol_inner Kbeta Ktheta TM prods T Total share price nn v IV delta_02 xi mc 

theta1     = x1(1:Kbeta,1);
theta2     = x1(Kbeta+1:Kbeta+Ktheta,1);
gamma      = x1(Kbeta+Ktheta+1:end,1);
ii         = 0;
norm_max   = 1;

%%% Define initial delta
delta      = delta_02;
mu         = X_GMM * (repmat(theta2,1,nn).* v);

while (norm_max > tol_inner) && (ii < 10000)
    %%% the better way to tackle the problem!
    delta_02 = delta;
    numerator   = exp(repmat(delta_02,1,nn) + mu);        
    denominator   = zeros(Total,nn);
    for m = 1:TM
        denominator(T(m,1):T(m,2),:) = repmat(1 + sum(numerator(T(m,1):T(m,2),:),1),prods(m),1);
    end

    %%% Define share_ijm and share_jm p31/78
    share_ijm  = numerator./denominator;
    share_jm   = mean(share_ijm,2);
    
    %%% Now the contraction mapping
    delta  = delta_02 + log(share) - log(share_jm);
    norm_max = max(abs(delta - delta_02));
    
    ii     = ii + 1;
end

%%%% DEMAND-SIDE MOMENT
xi     = delta - XX_GMM * theta1;

%%%% SUPPLY-SIDE MOMENT
%%%%%% First, compute \alpha_i
%%%%%% Then, recover the denominator of the markup
alpha_i = theta1(Kbeta,1) + theta2(Ktheta,1) * v(Ktheta,:);
denominator_markup = mean(alpha_i .* share_ijm .* (1-share_ijm),2);

%%%%%% Use definition of marginal cost, without log transformation
%%%%%% Retrieve \omega
mc     = price - share_jm ./ (-denominator_markup);
omega  = mc - Z * gamma;

%%%%%% Define moments
%%%%%% As a stack
g      = [IV' * xi ; IV' * omega];

%%%%%% Optimize
f      = g' * W * g;

end