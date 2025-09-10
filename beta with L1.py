import numpy as np
from scipy.special import gammaln, digamma, expit
from scipy.optimize import minimize
def beta_loglik(beta, phi, X, y):
    mu=expit(X@beta)
    ll=(gammaln(phi)-gammaln(mu*phi)-gammaln((1-mu)*phi)+(mu*phi-1)*np.log(y)+((1-mu)*phi-1)*np.log(1-y))
    return np.sum(ll)

def beta_grad(beta, phi, X, y):
    mu=expit(X@beta)
    dmu=(mu*(1-mu))[:,None]*X
    d_ll_dmu=(-phi*digamma(mu*phi)+phi*digamma((1-mu)*phi)+phi*(np.log(y)-np.log(1-y)))
    grad=np.sum(d_ll_dmu[:,None]*dmu, axis=0)
    return grad


def soft_threshold(z, lam):
    return np.sign(z)*np.maximum(np.abs(z)-lam, 0.0)


# --------- Main optimizer ----------

def beta_regression_l1(X, y, alpha=0.01, lr=1e-6, max_iter=100000, tol=1e-6):
    n, p=X.shape
    beta=np.zeros(p)
    eps=1e-8
    y=np.clip(y, eps, 1 - eps)
    phi=1.0  # initialize

    for it in range(max_iter):
        beta_old=beta.copy()
        phi_old=phi

        grad=-beta_grad(beta, phi, X, y)  # negative log-likelihood gradient
        beta=soft_threshold(beta-lr*grad, lr*alpha)

        def phi_obj(log_phi):
            phi_val=np.exp(log_phi)
            return -beta_loglik(beta, phi_val, X, y)

        res=minimize(phi_obj,np.log(phi),method="L-BFGS-B")
        phi=np.exp(res.x[0])

        if np.linalg.norm(beta-beta_old, ord=1)<tol and abs(phi-phi_old)<tol:
            print(f"Converged at iteration {it}")
            break

    return beta, phi

if __name__=="__main__":
    np.random.seed(0)
    n, p=200, 10
    X=np.random.randn(n, p)

    beta_true=np.array([1, -0.75] + [0.0] * (p - 2))
    mu=expit(X @ beta_true)
    phi_true=5.0

    from scipy.stats import beta as beta_dist
    y=beta_dist.rvs(mu*phi_true, (1-mu)*phi_true)
    beta_est, phi_est=beta_regression_l1(X, y, alpha=1.0, lr=1e-6)

    print("Estimated beta:", np.round(beta_est, 3))
    print("Estimated phi:", np.round(phi_est, 3))

