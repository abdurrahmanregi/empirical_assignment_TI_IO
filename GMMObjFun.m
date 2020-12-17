function f = GMMObjFun(x0,X_GMM,XX_GMM)

global W tol_inner Kbeta Ktheta Total nn TM T share prods IV v delta_01

theta1      = x0(1:Kbeta,1);                % Mean of tastes
theta2      = x0(Kbeta+1:Kbeta+Ktheta,1);   % Std deviation of tastes
ii          = 0;
norm_max    = 1;

%%% Define initial delta
delta       = delta_01;                % Making life easier (plus, making it as global)
mu          = X_GMM*(diag(theta2)*v);       % (970xnn), p30/78 slides

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

%%% Define eps (or xi, IDK) as in p43/78
xi  = delta - XX_GMM*theta1;
g   = IV'*xi;                   % 13x13 = (13x970) x (970x13) (moments)

f       = g'*W*g;               % Objective function

end