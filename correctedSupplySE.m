function [Var_cov_hat2] = correctedSupplySE(X_GMM, XX_GMM, Z, theta1_q2, theta2_q2, gamma_q2)

global W tol_inner Kbeta Ktheta Total nn TM T share prods IV v delta_01 delta_02
global price

%%% Now, we wanted to recover the correct standard errors. We need to
%%% recover moment \xi
%%% Source: Sudhir's Slide (2013)
%%%% First, do while-loop for the demand side
ii          = 0;
norm_max    = 1;
delta       = delta_02;                % Making life easier (plus, making it as global)
mu          = X_GMM*(diag(theta2_q2)*v);       % (970xnn), p30/78 slides

while norm_max > tol_inner && ii < 10000    % Loop until convergence
    %%% the better way to tackle the problem!
    delta_02        = delta;
    numerator       = exp(repmat(delta_02,1,nn) + mu);
    denominator     = zeros(Total,nn);  % (970xnn), init denominator
    for m=1:TM          % loop for each market
        denominator(T(m,1):T(m,2),:)    = repmat(1 + sum(numerator(T(m,1):T(m,2),:),1),prods(m),1);
    end
    
    %%% Define share_ijm and share_jm p31/78
    share_ijm = numerator./denominator; %(970x100)
    share_jm = mean(share_ijm,2);   %(970x1)
    
    %%% Now the contraction mapping
    delta           = delta_01 + log(share) - log(share_jm);
    norm_max        = max(abs(delta - delta_01));  % return one num to update the norm_max 
    
    ii           = ii + 1;
end

xi  = delta - XX_GMM*theta1_q2;     % estimated \xi

%%%% Second, do for the supply side
%%%%%% First, compute \alpha_i
%%%%%% Then, recover the denominator of the markup
alpha_i = theta1_q2(Kbeta,1) + theta2_q2(Ktheta,1) * v(Ktheta,:);
denominator_markup = mean(alpha_i .* share_ijm .* (1-share_ijm),2);

%%%%%% Use definition of marginal cost, without log transformation
%%%%%% Retrieve \omega
mc     = price - share_jm ./ (-denominator_markup);
omega  = mc - Z * gamma_q2;

%%%% Third, retrieve Jacobian matrix w.r.t. theta
for m = 1:TM
    for j = 1:prods(m,1)
        for l = 1:prods(m,1)
            if j == l
                K(T(m,1)+j-1, T(m,1)+l-1) = mean(share_ijm(T(m,1)+l-1,:).*(1 - share_ijm(T(m,1)+l-1,:)));
            else
                K(T(m,1)+j-1, T(m,1)+l-1) = mean(share_ijm(T(m,1)+l-1,:).*share_ijm(T(m,1)+j-1,:));
            end
        end
    end
end

for m = 1:TM
    for j = 1:prods(m,1)
        for l = 1:Ktheta
            L(T(m,1)+j-1,:,l) = X_GMM(T(m,1)+j-1,l)*share_ijm(T(m,1)+j-1,:);
        end
    end
end

M = sum(L,1);

for m=1:TM
    for j = 1:prods(m,1)
        for l=1:Ktheta
            N(T(m,1)+j-1,l) = mean(v(l,:).*share_ijm(T(m,1)+j-1,:).*( X_GMM(T(m,1)+j-1,l) - M(:,:,l) ));
        end
    end
end

Jacobian_delta2     = K\N;      % Jacobian of the moment conditions w.r.t theta2
Jacobian_D1         = [ones(Total,1) X_GMM Jacobian_delta2];     % Jacobian of the moment conditions to theta and supply side!

%%%% Fourth, define new IV
IV_02               = blkdiag(IV,IV);   % Make block diagonal, 2xsize(IV)
Jacobian_D2         = blkdiag(Jacobian_D1,Z);

%%%% Fifth, define \Sigma and Var-covar matrix
Sigma_hat2          = IV_02' * [ xi; omega] * [xi;  omega]' * IV_02;
Var_cov_hat2        = inv(Jacobian_D2'*IV_02*W*IV_02'*Jacobian_D2)*(Jacobian_D2'*IV_02*W*Sigma_hat2*W*IV_02'*Jacobian_D2)*inv(Jacobian_D2'*IV_02*W*IV_02'*Jacobian_D2);



end