function [Var_cov_hat] = correctedDemandSE(X_GMM, XX_GMM, theta1_q1, theta2_q1)

%%% Corrected Standard Errors

global W tol_inner Kbeta Ktheta Total nn TM T share prods IV v delta_01

%%% Now, we wanted to recover the correct standard errors. We need to
%%% recover moment \xi
%%% Source: Sudhir's Slide (2013)
%%%% First, do while-loop
ii          = 0;
norm_max    = 1;
delta       = delta_01;                % Making life easier (plus, making it as global)
mu          = X_GMM*(diag(theta2_q1)*v);       % (970xnn), p30/78 slides

while norm_max > tol_inner && ii < 10000    % Loop until convergence
    %%% the better way to tackle the problem!
    delta_01        = delta;
    numerator       = exp(repmat(delta_01,1,nn) + mu);
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

xi  = delta - XX_GMM*theta1_q1;     % estimated \xi

%%%% Second, retrieve Jacobian matrix w.r.t. theta
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

Jacobian_delta  = K\N;      % Jacobian of the moment conditions w.r.t theta2
Jacobian_D      = [ones(Total,1) X_GMM Jacobian_delta];     % Jacobian of the moment conditions w.r.t. theta1

%%%% Third, compute \hat{\Sigma} and \hat{variance-covariance matrix}
Sigma_hat       = IV' * xi * xi' * IV;
Var_cov_hat     = inv(Jacobian_D'*IV*W*IV'*Jacobian_D)*(Jacobian_D'*IV*W*Sigma_hat*W*IV'*Jacobian_D)*inv(Jacobian_D'*IV*W*IV'*Jacobian_D);

end