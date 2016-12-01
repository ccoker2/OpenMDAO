""" Surrogate model based on Kriging. """
from __future__ import print_function
from six import iteritems

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize
from six.moves import zip, range

from openmdao.surrogate_models.surrogate_model import SurrogateModel
from openmdao.test.util import set_pyoptsparse_opt

MACHINE_EPSILON = np.finfo(np.double).eps

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


def snopt_opt(objfun, desvar, lb, ub, title=None, options=None,
              sens=None, jac=None):
    """ Wrapper function for running a SNOPT optimization through
    pyoptsparse."""

    if OPTIMIZER:
        from pyoptsparse import Optimization
    else:
        raise(RuntimeError, 'Need pyoptsparse to run the SNOPT sub optimizer.')

    opt_prob = Optimization(title, objfun)

    ndv = len(desvar)

    opt_prob.addVarGroup('thetas', ndv, type='c', value=desvar.flatten(), lower=lb.flatten(),
                         upper=ub.flatten())
    opt_prob.addObj('obj')

    # Fall back on SLSQP if SNOPT isn't there
    _tmp = __import__('pyoptsparse', globals(), locals(), [OPTIMIZER], 0)
    opt = getattr(_tmp, OPTIMIZER)()


    if options:
        for name, value in iteritems(options):
            opt.setOption(name, value)

    opt.setOption('Major iterations limit', 100)
    opt.setOption('Verify level', -1)
    opt.setOption('iSumm', 0)
    #opt.setOption('iPrint', 0)

    sol = opt(opt_prob, sens=sens, sensStep=1.0e-6)
    #print(sol)

    x = sol.getDVs()['thetas']
    f = sol.objectives['obj'].value
    success_flag = sol.optInform['value'] < 2
    msg = sol.optInform['text']

    return x, f, success_flag, msg


class KrigingSurrogate(SurrogateModel):
    """Surrogate Modeling method based on the simple Kriging interpolation.
    Predictions are returned as a tuple of mean and RMSE. Based on Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams. (see also: scikit-learn).

    Args
    ----
    nugget : double or ndarray, optional
        Nugget smoothing parameter for smoothing noisy data. Represents the variance of the input values.
        If nugget is an ndarray, it must be of the same length as the number of training points.
        Default: 10. * Machine Epsilon
    eval_rmse : bool
        Flag indicating whether the Root Mean Squared Error (RMSE) should be computed. Set to False
        by default.
    """

    def __init__(self, nugget=10. * MACHINE_EPSILON, eval_rmse=False):
        super(KrigingSurrogate, self).__init__()

        self.n_dims = 0       # number of independent
        self.n_samples = 0       # number of training points
        self.thetas = np.zeros(0)
        self.nugget = nugget     # nugget smoothing parameter from [Sasena, 2002]

        self.c_r = np.zeros(0)
        self.L = np.zeros(0)
        self.SigmaSqr = np.zeros(0)

        # Normalized Training Values
        self.X = np.zeros(0)
        self.Y = np.zeros(0)
        self.X_mean = np.zeros(0)
        self.X_std = np.zeros(0)
        self.Y_mean = np.zeros(0)
        self.Y_std = np.zeros(0)

        self.use_snopt = False
        self.eval_rmse = eval_rmse

        self.Wstar = np.identity(0)
        self.pcom=0

    def train(self, x, y, KPLS_status=False):
        """
        Train the surrogate model with the given set of inputs and outputs.

        Args
        ----
        x : array-like
            Training input locations

        y : array-like
            Model responses at given inputs.

        KPLS_status : Boolean
            False when KPLS is not added to Kriging (default)
            True Adds KPLS method to Kriging to reduce the number of hyper-parameters
        """

        super(KrigingSurrogate, self).train(x, y)

        x, y = np.atleast_2d(x, y)

        self.n_samples, self.n_dims = x.shape

        if self.n_samples <= 1:
            raise ValueError(
                'KrigingSurrogate require at least 2 training points.'
            )

        # Normalize the data
        X_mean = np.mean(x, axis=0)
        X_std = np.std(x, axis=0)
        Y_mean = np.mean(y, axis=0)
        Y_std = np.std(y, axis=0)

        X_std[X_std == 0.] = 1.
        Y_std[Y_std == 0.] = 1.

        X = (x - X_mean) / X_std
        Y = (y - Y_mean) / Y_std

        self.X = X
        self.Y = Y
        self.X_mean, self.X_std = X_mean, X_std
        self.Y_mean, self.Y_std = Y_mean, Y_std

        if KPLS_status:
            pcom_max = 3 #Maximum number of hyper-parameters we want to afford
            self.pcom = min([pcom_max,self.n_dims]) #TODO Use some criteria to find optimal number of hyper-parameters.
            self.Wstar = self.KPLS_reg()
            if self.pcom >= 3:
                num_start = 5*self.pcom
                #TODO: Read this from a file
                start_point = [[0.5, 0.5714, 0.5714],
                               [0.6429, 0.0714, 0.3571],
                               [0.9286, 0.7857, 0.7857],
                               [0.3571, 0.7143, 0.1429],
                               [0.8571, 0.2857, 0.6429],
                               [0.7857, 0.8571, 0.2857],
                               [0.0714, 0.6429, 0.8571],
                               [1.0, 0.0, 0.0714],
                               [0.2857, 0.2143, 0.7143],
                               [0., 0.4286, 0.4286],
                               [0.5714, 0.3571, 1.0],
                               [0.2143, 1.0, 0.5],
                               [0.1429, 0.1429, 0.2143],
                               [0.7143, 0.5, 0.0],
                               [0.4286, 0.9286, 0.9286]]
            else:
                num_start = 3
                start_point = [[0.25], [0.5], [0.75]]
        else:
            self.Wstar = np.identity(self.n_dims)
            self.pcom = self.n_dims
            num_start = 3
            start_point = [[0.25], [0.5], [0.75]]

        # Multi-start approach (starting from 10*pcom_max different locations)
        #FIXME: Parallelize this multi-start
        best_loglike = np.inf
        #Start from random locations
        for point in start_point:
            x0 = -3.0*np.ones((self.pcom, )) + point*(5.0*np.ones((self.pcom, )))
            if self.use_snopt:
                def _calcll(dv_dict):
                    """ Callback function"""
                    fail = 0
                    thetas = dv_dict['thetas']
                    x = np.dot((self.Wstar**2),(10**thetas).T).flatten()
                    loglike = self._calculate_reduced_likelihood_params(x)[0]

                    # Objective
                    func_dict = {}
                    func_dict['obj'] = -loglike

                    return func_dict, fail

                low = -3.0*np.ones([self.pcom, 1])
                high = 2.0*np.ones([self.pcom, 1])
                # print("Using SNOPT!")
                opt_x, opt_f, succ_flag, msg = snopt_opt(_calcll, x0, low, high, title='kriging',
                                                         options={'Major optimality tolerance' : 1.0e-6})

                if not succ_flag:
                    print("SNOPT failed to converge.", msg)
                    opt_f = 1.0
                    pass
                    #raise ValueError('Kriging Hyper-parameter optimization failed: {0}'.format(msg))

                thetas = np.asarray(10**opt_x).flatten()
                fval = opt_f
            else:

                def _calcll(thetas):
                    """ Callback function"""
                    x = np.dot((self.Wstar**2),(10**thetas).T).flatten()
                    loglike = self._calculate_reduced_likelihood_params(x)[0]
                    return -loglike

                bounds = [(-3.0, 2.0) for _ in range(self.pcom)]
                # print("Using Cobyla")
                optResult = minimize(_calcll, x0, method='cobyla',
                                     options={'ftol': 1e-6},
                                     bounds=bounds)

                if not optResult.success:
                    print("Cobyla failed to converge", optResult.success)
                    optResult.fun = 1.0
                    pass
                    #raise ValueError('Kriging Hyper-parameter optimization failed: {0}'.format(optResult.message))

                thetas = 10**optResult.x.flatten()
                fval = optResult.fun

            if best_loglike > fval:
                best_loglike = fval*1.0
                self.thetas = np.dot((self.Wstar**2),thetas.T).flatten()

        print("BestLogLike: ",best_loglike)

        _, params = self._calculate_reduced_likelihood_params()
        self.c_r = params['c_r']
        self.U = params['U']
        self.S_inv = params['S_inv']
        self.Vh = params['Vh']
        self.mu = params['mu']
        self.SigmaSqr = params['SigmaSqr']
        self.R_inv = params['R_inv']

    def _calculate_reduced_likelihood_params(self, thetas=None):
        """
        Calculates a quantity with the same maximum location as the log-likelihood for a given theta.

        Args
        ----
        thetas : ndarray, optional
            Given input correlation coefficients. If none given, uses self.thetas from training.
        """
        if thetas is None:
            thetas = self.thetas

        X, Y = self.X, self.Y
        params = {}

        # Correlation Matrix
        distances = np.zeros((self.n_samples, self.n_dims, self.n_samples))
        for i in range(self.n_samples):
            distances[i, :, i+1:] = np.abs(X[i, ...] - X[i+1:, ...]).T
            distances[i+1:, :, i] = distances[i, :, i+1:].T

        R = np.exp(-thetas.flatten().dot(np.square(distances)))
        R[np.diag_indices_from(R)] = 1. + self.nugget

        [U,S,Vh] = linalg.svd(R)

        # Penrose-Moore Pseudo-Inverse:
        # Given A = USV^* and Ax=b, the least-squares solution is
        # x = V S^-1 U^* b.
        # Tikhonov regularization is used to make the solution significantly more robust.
        h = 1e-8 * S[0]
        inv_factors = S / (S ** 2. + h ** 2.)

        # Using the approach suggested on 1. EGO by D.R.Jones et.al and
        # 2. Engineering Deisgn via Surrogate Modeling-A practical guide
        # by Alexander Forrester, Dr. Andras Sobester, Andy Keane
        one = np.ones([self.n_samples,1])
        R_inv = Vh.T.dot(np.einsum('i,ij->ij',
                                              inv_factors,
                                              U.T))
        mu = np.dot(one.T,np.dot(R_inv,Y))/np.dot(one.T,np.dot(R_inv,one))
        c_r = Vh.T.dot(np.einsum('j,kj,kl->jl', inv_factors, U, (Y - mu*one)))
        logdet = -np.sum(np.log(inv_factors))
        SigmaSqr = np.dot((Y - mu*one).T, c_r).sum(axis=0) / self.n_samples
        reduced_likelihood = -(np.log(np.sum(SigmaSqr)) + logdet / self.n_samples)

        params['c_r'] = c_r
        params['S_inv'] = inv_factors
        params['U'] = U
        params['Vh'] = Vh
        params['R_inv'] = R_inv
        params['mu'] = mu
        params['SigmaSqr'] = SigmaSqr #This is wrt normalized y

        return reduced_likelihood, params

    def predict(self, x):
        """
        Calculates a predicted value of the response based on the current
        trained model for the supplied list of inputs.

        Args
        ----
        x : array-like
            Point at which the surrogate is evaluated.
        normalize : bool
            Normalize the training data to lie on [-1, 1]. Default is True, but
            some applications like Branch and Bound require False.
        """

        super(KrigingSurrogate, self).predict(x)

        X, Y = self.X, self.Y
        thetas = self.thetas
        if isinstance(x, list):
            x = np.array(x)
        x = np.atleast_2d(x)
        n_eval = x.shape[0]

        x_n = (x - self.X_mean) / self.X_std

        r = np.zeros((n_eval, self.n_samples), dtype=x.dtype)
        for r_i, x_i in zip(r, x_n):
            r_i[:] = np.exp(-thetas.dot(np.square((x_i - X).T)))

        if r.shape[1] > 1: #Ensure r is always a column vector
            r = r.T

        # Predictor
        y_t = self.mu + np.dot(r.T, self.c_r)
        y = self.Y_mean + self.Y_std * y_t


        if self.eval_rmse:
            one = np.ones([self.n_samples,1])
            mse  = self.SigmaSqr*(1.0 - np.dot(r.T,np.dot(self.R_inv,r)) + \
            ((1.0 - np.dot(one.T,np.dot(self.R_inv,r)))**2/np.dot(one.T,np.dot(self.R_inv,one))))

            # Forcing negative RMSE to zero if negative due to machine precision
            mse[mse < 0.] = 0.
            return y, np.sqrt(mse)

        return y

    def linearize(self, x):
        """
        Calculates the jacobian of the Kriging surface at the requested point.

        Args
        ----
        x : array-like
            Point at which the surrogate Jacobian is evaluated.
        """

        thetas = self.thetas

        # Normalize Input
        x_n = (x - self.X_mean) / self.X_std

        r = np.exp(-thetas.dot(np.square((x_n - self.X).T)))

        # Z = einsum('i,ij->ij', X, Y) is equivalent to, but much faster and
        # memory efficient than, diag(X).dot(Y) for vector X and 2D array Y.
        # I.e. Z[i,j] = X[i]*Y[i,j]
        gradr = r * -2 * np.einsum('i,ij->ij', thetas, (x_n - self.X).T)
        jac = np.einsum('i,j,ij->ij', self.Y_std, 1./self.X_std, gradr.dot(self.c_r).T)
        return jac

    def KPLS_reg(self):
        def power_iter(X,y):
            A = np.dot(np.dot(X.T,y),np.dot(y.T,X))
            qk = np.zeros([A.shape[0],1])
            qk[0] = 1.0
            kk=0.
            delta = 1.0
            qk_prev = qk
            while delta>1.0e-6:
                kk = kk + 1.
                zk = np.dot(A,qk)
                qk = zk/np.linalg.norm(zk)
                delta = np.linalg.norm(qk - qk_prev)
                qk_prev = qk
            return qk

        Xl = self.X
        yl = self.Y
        k = self.n_dims
        for l in range(self.pcom):
            wl = power_iter(Xl,yl)
            tl = np.dot(Xl,wl)
            tl_hat = tl/(np.dot(tl.T,tl))
            pl = (np.dot(Xl.T,tl_hat)).T
            cl = np.dot(yl.T,tl_hat)
            if l == 0: #FIXME: Re-code this
                W = wl*1.0
                P = pl.T*1.0
            else:
                W = np.concatenate((W,wl), axis=1)
                P = np.concatenate((P,pl.T), axis=1)
            Xl = Xl - np.dot(tl,pl)
            yl = yl - cl*tl
        Wstar = np.dot(W,np.linalg.inv(np.dot(P.T,W))) #TODO: See if there are better ways to do inverse
        return Wstar

class FloatKrigingSurrogate(KrigingSurrogate):
    """Surrogate model based on the simple Kriging interpolation. Predictions are returned as floats,
    which are the mean of the model's prediction."""

    def predict(self, x):
        dist = super(FloatKrigingSurrogate, self).predict(x)
        return dist[0]  # mean value
