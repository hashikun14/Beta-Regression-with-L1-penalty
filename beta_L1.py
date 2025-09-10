import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, digamma, expit

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def soft_threshold(z, tau):
    return np.sign(z)*np.maximum(np.abs(z)-tau,0.0)

def negloglik_beta_regression(beta0, beta, phi, X, y, eps=1e-12):
    n, p=X.shape
    eta=beta0 + X.dot(beta)
    mu=sigmoid(eta)
    mu=np.clip(mu, 1e-12, 1 - 1e-12)
    y=np.clip(y,eps,1-eps)
    # log-likelihood per observation
    # l_i=log Gamma(phi)-log Gamma(phi*mu)-log Gamma(phi*(1-mu))+(phi*mu-1)*log(y)+(phi*(1-mu)-1)*log(1-y)
    a=gammaln(phi)-gammaln(phi*mu)-gammaln(phi*(1-mu))
    b=(phi*mu-1.0)*np.log(y)+(phi*(1-mu)-1.0)*np.log(1-y)
    loglik=np.sum(a+b)
    return -loglik

# Finite-difference gradient (central)
def grad_Rn(beta0, beta, phi, X, y):
    eta=beta0+X@beta
    mu=1/(1+np.exp(-eta))
    mu=np.clip(mu,1e-12,1-1e-12)
    y=np.clip(y,1e-12,1-1e-12)

    alpha=phi*mu
    beta_par=phi*(1-mu)

    # dR/dmu
    dR_dmu=phi*(digamma(alpha)-digamma(beta_par)-np.log(y)+np.log(1-y))
    dmu_deta=mu*(1-mu)

    # gradient wrt beta0
    grad_beta0=np.sum(dR_dmu*dmu_deta)
    # gradient wrt beta
    grad_beta=X.T@(dR_dmu*dmu_deta)
    return grad_beta0, grad_beta


# Fit phi by minimizing R_n over phi > 0
def fit_phi(beta0, beta, phi_init, X, y):
    def obj(log_phi):
        phi = np.exp(log_phi)  # optimize in log-space for stability
        return negloglik_beta_regression(beta0, beta, phi, X, y)
    res = minimize_scalar(obj, bounds=(np.log(1e-6), np.log(1e6)), method='bounded', options={'xatol':1e-4})
    phi_hat = float(np.exp(res.x)) if res.success else phi_init
    return phi_hat

# -------------------------
# Main proximal-gradient algorithm with backtracking and phi re-fit
# -------------------------
def proximal_gradient_beta_regression(
    X, y,
    lambda_,          # L1 penalty weight applied to beta (not beta0)
    phi_init=10.0,    # initial phi (precision)
    s_init=1e-2,      # initial stepsize
    M=10,             # re-fit phi every M iterations
    tol=1e-6,
    max_iter=1000,
    verbose=False
):
    n, p=X.shape
    # initialize parameters
    beta0=0.0
    beta=np.zeros(p)
    phi=float(phi_init)
    s=float(s_init)

    R_prev=negloglik_beta_regression(beta0, beta, phi, X, y)

    for t in range(1, max_iter + 1):
        # compute gradient at current theta
        d_beta0, d_beta = grad_Rn(beta0, beta, phi, X, y)

        # candidate update (gradient step for beta0, prox for beta)
        while True:
            beta0_prime=beta0-s*d_beta0
            beta_candidate=soft_threshold(beta-s*d_beta, lambda_*s)

            # prepare theta' and check condition (7)
            R_candidate=negloglik_beta_regression(beta0_prime, beta_candidate, phi, X, y)
            # inner product <∇_{beta0,beta} R_n (theta), theta' - theta>
            delta_beta0=beta0_prime-beta0
            delta_beta=beta_candidate-beta
            inner=d_beta0*delta_beta0+np.dot(d_beta, delta_beta)
            norm_sq=delta_beta0**2+np.dot(delta_beta, delta_beta)
            rhs =R_prev+inner+(1.0 / (2.0 * s))*norm_sq

            if R_candidate<=rhs+1e-12:  # condition (7) satisfied (small tol)
                break
            else:
                # backtrack step size
                s*=0.9
                if s < 1e-12:
                    # cannot reduce s further; accept candidate anyway to avoid infinite loop
                    break

        # accept updates
        beta0=beta0_prime
        beta=beta_candidate

        # optionally re-fit phi every M iterations
        if (t % M) == 0:
            phi=fit_phi(beta0, beta, phi, X, y)

        # evaluate objective and stopping
        R_curr=negloglik_beta_regression(beta0, beta, phi, X, y)
        if verbose and (t % 10 == 0 or t <= 5):
            print(f"iter {t:4d}  R={R_curr:.6f}  phi={phi:.6f}  s={s:.3e}")

        if (R_prev - R_curr) < tol:
            if verbose:
                print(f"Converged at iter {t}.")
            break
        R_prev = R_curr

    return {'beta0': beta0, 'beta': beta, 'phi': phi, 'R': R_curr, 'iterations': t}
if __name__ == "__main__":
    np.random.seed(0)
    n, p=200, 10
    X=np.random.randn(n, p)

    beta_true=np.array([1, -0.75] + [0.0] * (p - 2))
    mu=expit(X@beta_true)
    phi_true=5.0

    from scipy.stats import beta as beta_dist
    y=beta_dist.rvs(mu*phi_true, (1-mu)*phi_true)
    result1=proximal_gradient_beta_regression(X, y, lambda_=1.0,verbose=True)