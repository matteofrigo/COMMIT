# cython: profile=False
cimport cython
import numpy as np
cimport numpy as np
from math import sqrt
import sys
eps = np.finfo(float).eps

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def init_regularisation(thenorm = 2):
    """Initialise the structure that describes the regularisation term
    $ \Omega(x) = \lambda \sum_{g\inG} w_g \| x_{|g} \|_n $

    Parameters
    ----------
    thenorm : integer
        n-norm to be used in the sum (default: 2).

    Returns
    ----------
    Dictionary structure with the following fields
        startIC : integer
            index from which the IntraCellular (IC) compartment starts.
        sizeIC : np.array of integers
            list of sizes of each group of the IC compartment (check gnnls).
        structureIC : np.array of objects
            list of indices of each group of the structured sparsity penalty
            term (check hnnls).
        weightsIC: np.array of float64
            weights associated to each group encoded in sizeIC or structureIC
        lambdaIC : float64
            regularisation parameter for the IC compartment
        normIC : integer
            n-norm for the regularisation of the IC compartment

    ~The following are intended to be useful in future verisons of the package~
        startEC : integer
            index from which the ExtraCellular (EC) compartment starts.
        sizeEC : np.array of integers
            list of sizes of each group of the EC compartment (check gnnls).
        weightsEC: np.array of float64
            weights associated to each group encoded in sizeEC
        lambdaEC : float64
            regularisation parameter for the EC compartment
        normEC : integer
            n-norm for the regularisation of the EC compartment
        startISO : integer
            index from which the isotropic (ISO) compartment starts.
        sizeISO : np.array of integers
            list of sizes of each group of the ISO compartment (check gnnls).
        weightsISO: np.array of float64
            weights associated to each group encoded in sizeISO
        lambdaISO : float64
            regularisation parameter for the ISO compartment
        normISO : integer
            n-norm for the regularisation of the ISO compartment

    Notes
    -----
    Author: Matteo Frigo - lts5 @ EPFL
    """
    regularisation = {}
    regularisation['startIC'] = 0
    regularisation['sizeIC'] = np.array([0], dtype=np.int64)
    regularisation['structureIC'] = None
    regularisation['weightsIC'] = np.zeros_like(regularisation['sizeIC'], dtype=np.float64)

    regularisation['startEC'] = 0
    regularisation['sizeEC'] = np.array([0], dtype=np.int64)
    regularisation['weightsEC'] = np.zeros_like(regularisation['sizeEC'], dtype=np.float64)

    regularisation['startISO'] = 0
    regularisation['sizeISO'] = np.array([0], dtype=np.int64)
    regularisation['weightsISO'] = np.zeros_like(regularisation['sizeISO'], dtype=np.float64)

    regularisation['lambdaIC'] = 0.0
    regularisation['lambdaEC'] = 0.0
    regularisation['lambdaISO'] = 0.0

    regularisation['normIC'] = int(thenorm)
    regularisation['normEC'] = int(thenorm)
    regularisation['normISO'] = int(thenorm)

    return regularisation

## Interface for NNLS solver
def nnls(y, A, At, tol_fun = 1e-4, tol_x = 1e-6, max_iter = 1000, verbose = 1):
    """Solve Non-Negative Least Squares (NNLS)

       min 0.5 * || y - Ax ||_2^2,          s.t. x >= 0

    Parameters
    ----------
    y : 1-d array of doubles.
        Signal to be fit.

    A : matrix or object endowed with the .dot() method.
        Dictionary describing the forward model.

    At : matrix or class exposing the .dot() method.
        Adjoint operator of A.

    tol_fun : double, optional (default: 1e-4).
        Minimum relative change of the objective value. The algorithm stops if:
               | f(x(t)) - f(x(t-1)) | / f(x(t)) < tol_fun,
        where x(t) is the estimate of the solution at iteration t.

    tol_x : double, optional (default: 1e-6).
        Minimum relative change of the solution x. The algorithm stops if:
               || x(t) - x(t-1) || / || x(t) || < tol_x,
        where x(t) is the estimate of the solution at iteration t.

    max_iter : integer, optional (default: 1000).
        Maximum number of iterations.

    verbose : integer, optional (default: 1).
        0 no log, 1 print each iteration results.

    Returns
    -------
    x : 1-d array of doubles.
        Minimizer of the NNLS problem.

    Notes
    -----
    Author: Matteo Frigo - lts5 @ EPFL
    Acknowledgment: Rafael Carrillo - lts5 @ EPFL
    """
    x0 = np.zeros( A.shape[1], dtype=np.float64 )

    def prox(x):
        x[x<0.0] = 0.0
        return x
    omega = lambda x: 0.0

    return __fista(y, A, At, tol_fun, tol_x, max_iter, verbose, x0, omega, prox)

## Interface for GNNLS solver
def gnnls(y, A, At, tol_fun = 1e-4, tol_x = 1e-6, max_iter = 1000, verbose = 1, regularisation = None) :
    """Solve Group-structured Non-Negative Least Squares (GNNLS)

       min 0.5 * || y - Ax ||_2^2 + \Omega(x),      s.t. x >= 0
        where \Omega(x) is a regularisation functional of the form
            \Omega(x) = \lambda \sum_{g\in G} \|x_{g}\|.
        The groups defined in G must not overlap.
        If \lambda=0 the solver is equivalent to NNLS.

    Parameters
    ----------
    y : 1-d array of doubles.
        Signal to be fit.

    A : matrix or object endowed with the .dot() method.
        Dictionary describing the forward model.

    At : matrix or class exposing the .dot() method.
        Adjoint operator of A.

    tol_fun : double, optional (default: 1e-4).
        Minimum relative change of the objective value. The algorithm stops if:
               | f(x(t)) - f(x(t-1)) | / f(x(t)) < tol_fun,
        where x(t) is the estimate of the solution at iteration t.

    tol_x : double, optional (default: 1e-6).
        Minimum relative change of the solution x. The algorithm stops if:
               || x(t) - x(t-1) || / || x(t) || < tol_x,
        where x(t) is the estimate of the solution at iteration t.

    max_iter : integer, optional (default: 1000).
        Maximum number of iterations.

    verbose : integer, optional (default: 1).
        0 no log, 1 print each iteration results.

    regularisation : python dictionary
        Structure initialised by init_regularisation(). The necessary fields are:
            *lambdaIC
            *normIC
            *sizeIC
            *weightsIC
            *startIC
        Field sizeIC must be a list that encodes the length of each group, e.g.
            sizeIC = np.array([1,23,4,13,48])
        means that starting from startIC we have five groups represented by the
        fibres associated to indices (python-like) 0, 1:23, 23:27, 27:40, 40:88.

    Returns
    -------
    x : 1-d array of doubles.
        Minimizer of the GNNLS problem.

    Notes
    -----
    Author: Matteo Frigo - lts5 @ EPFL
    Acknowledgment: Rafael Carrillo - lts5 @ EPFL
    """
    x0 = np.zeros( A.shape[1], dtype=np.float64 )

    lambdaIC = regularisation.get('lambdaIC')
    lambdaEC = regularisation.get('lambdaEC')
    lambdaISO = regularisation.get('lambdaISO')
    if lambdaIC == 0.0: # and lambdaEC == 0.0 and lambdaISO == 0.0:
        return nnls(y, A, At, tol_fun, tol_x, max_iter, verbose)

    normIC      = regularisation.get('normIC')
    sizeIC      = regularisation.get('sizeIC')
    weightsIC   = regularisation.get('weightsIC')
    startIC     = regularisation.get('startIC')

    # normEC    = regularisation.get('normEC')
    # sizeEC    = regularisation.get('sizeEC')
    # weightsEC = regularisation.get('weightsEC')
    # startEC   = regularisation.get('startEC')
    #
    # normISO    = regularisation.get('normISO')
    # sizeISO    = regularisation.get('sizeISO')
    # weightsISO = regularisation.get('weightsISO')
    # startISO   = regularisation.get('startISO')

    omega = lambda x: omega_group(x, startIC,  sizeIC,  weightsIC,  lambdaIC,  normIC)# + \
                    #   omega_group(x, startEC,  sizeEC,  weightsEC,  lambdaEC,  normEC) + \
                    #   omega_group(x, startISO, sizeISO, weightsISO, lambdaISO, normISO)
    prox  = lambda x:  prox_group(x, startIC,  sizeIC,  weightsIC,  lambdaIC,  normIC)# + \
                    #    prox_group(x, startEC,  sizeEC,  weightsEC,  lambdaEC,  normEC) + \
                    #    prox_group(x, startISO, sizeISO, weightsISO, lambdaISO, normISO)

    return __fista(y, A, At, tol_fun, tol_x, max_iter, verbose, x0, omega, prox)



## Interface for HNNLS solver
def hnnls(y, A, At, tol_fun = 1e-4, tol_x = 1e-6, max_iter = 1000, verbose = 1, regularisation = None) :
    """Solve Hierarchical Non-Negative Least Squares (GNNLS)

       min 0.5 * || y - Ax ||_2^2 + \Omega(x),      s.t. x >= 0
        where \Omega(x) is a regularisation functional of the form
            \Omega(x) = \lambda \sum_{g\in G} \|x_{g}\|
        and the groups in G define an hierarchical structure.

        E.g.: the tree structure
                         [a]
                       /     \
                    [b]       [e]
                   /   \     /   \
                 [c]   [d] [f]   [h]
        corresponds to the structure
        G = {[c],[d],[b,c,d],[f],[h],[e,f,h],[a,b,c,d,e,f,h], <<},
        where << is a large order relation. See [1] for an extended description
        of the hierarchical structure required by the solver.

        If \lambda=0 the solver is equivalent to NNLS.

    Parameters
    ----------
    y : 1-d array of doubles.
        Signal to be fit.

    A : matrix or object endowed with the .dot() method.
        Dictionary describing the forward model.

    At : matrix or class exposing the .dot() method.
        Adjoint operator of A.

    tol_fun : double, optional (default: 1e-4).
        Minimum relative change of the objective value. The algorithm stops if:
               | f(x(t)) - f(x(t-1)) | / f(x(t)) < tol_fun,
        where x(t) is the estimate of the solution at iteration t.

    tol_x : double, optional (default: 1e-6).
        Minimum relative change of the solution x. The algorithm stops if:
               || x(t) - x(t-1) || / || x(t) || < tol_x,
        where x(t) is the estimate of the solution at iteration t.

    max_iter : integer, optional (default: 1000).
        Maximum number of iterations.

    verbose : integer, optional (default: 1).
        0 no log, 1 print each iteration results.

    regularisation : python dictionary
        Structure initialised by init_regularisation(). The necessary fields are:
            *lambdaIC
            *normIC
            *structureIC
            *weightsIC
            *startIC
        Field structureIC is supposed to be given as np.array of lists, which
        will result in a dtype=object np.array.

    Returns
    -------
    x : 1-d array of doubles.
        Minimizer of the HNNLS problem.

    Notes
    -----
    Author: Matteo Frigo - lts5 @ EPFL
    Acknowledgment: Rafael Carrillo - lts5 @ EPFL
    References:
        [1] `Jenatton et al. - Proximal Methods for Hierarchical Sparse Coding`
    """
    x0 = np.ones( A.shape[1], dtype=np.float64 )

    if regularisation is None:
        raise ValueError('The given tree structure is empty. Check the documentation.')

    # TODO: regularisation of EC and ISO compartments
    lambdaIC = regularisation.get('lambdaIC')
    if lambdaIC == 0.0:
        return nnls(y, A, At, tol_fun, tol_x, max_iter, verbose)

    normIC = regularisation.get('normIC')
    structureIC = regularisation.get('structureIC')
    weightsIC = regularisation.get('weightsIC')
    omega = lambda x: omega_tree( x, structureIC, weightsIC, lambdaIC, normIC )
    prox = lambda x: prox_tree( x, structureIC, weightsIC, lambdaIC, normIC )

    return __fista(y, A, At, tol_fun, tol_x, max_iter, verbose, x0, omega, prox)

## Regularisers for HNNLS
# Penalty term
cpdef omega_tree(np.ndarray[np.float64_t] v, np.ndarray[object] subtree, np.ndarray[np.float64_t] weight, double lam, double n) :
    cdef:
        int nG = weight.size
        size_t k, i
        double xn, tmp = 0.0

    if lam != 0:
        if n == 1:
            for k in range(nG) :
                idx = subtree[k]
                tmp += weight[k] * max( v[idx] )
        elif n == 2:
            for k in range(nG):
                idx = subtree[k]
                xn = 0.0
                for i in idx:
                    xn += v[i]*v[i]
                    tmp += weight[k] * sqrt( xn )
        elif n == np.Inf:
            for k in range(nG):
                idx = subtree[k]
                tmp += weight[k] * sum( v[idx] )
    return lam*tmp

# Proximal operator of the penalty term
cpdef np.ndarray[np.float64_t] prox_tree( np.ndarray[np.float64_t] x, np.ndarray[object] subtree, np.ndarray[np.float64_t] weight, double lam, double n ) :
    cdef:
        np.ndarray[np.float64_t] v
        int nG = weight.size, N, rho
        size_t k, i
        double r, xn, theta

    v = x.copy()
    v[v<0] = 0.0

    if lam != 0:
        if n == 1 :
            for k in range(nG) :
                idx = subtree[k]
                # xn = max( v[idx] )
                r = weight[k] * lam
                for i in idx :
                    if v[i] <= r:
                        v[i] = 0.0
                    else :
                        v[i] -= r
        if n == 2:
            for k in range(nG):
                idx = subtree[k]
                xn = 0.0
                for i in idx:
                    xn += v[i]*v[i]
                    xn = sqrt(xn)
                r = weight[k] * lam
                if xn > r:
                    r = (xn-r)/xn
                    for i in idx :
                        v[i] *= r
                else:
                    for i in idx:
                        v[i] = 0.0
        # if n == np.Inf:
        #   for k in range(nG):
        #     idx = subtree[k]
        #     xn = max(v[idx])
        #     r = lam*weight[k]
        #     if xn <= r:
        #       for i in idx:
        #         v[i] = 0.0
        #     else:
        #       # Duchi, Shalen-Shwartz, Singer, Chandra - Efficient Projections onto the L1-Ball for Learning in High Dimensions
        #       N = len(idx)
        #       mu = np.sort(v[idx])[::-1]
        #       csmu = np.cumsum(mu)
        #       rho = np.nonzero( 0<mu*np.arange(1,N+1)-(csmu-r) )[0][-1]
        #       theta = (csmu[rho]-r)/(rho+1.0)
        #       for i in idx:
        #         v[i] = max(0.0,v[i]-theta)
    return v


## Regularisers for GNNLS
# Penalty term
cpdef double omega_group(np.ndarray[np.float64_t] v, np.int64_t startC, np.ndarray[np.int64_t] sizeC, np.ndarray[np.float64_t] weightsC, np.float64_t lambdaC, np.int64_t normC) :
    # Input:
    #     *v: vector of which we want to compute the proximal
    #     *startC: index on which the compartment starts
    #     *sizeC: ndarray with the length of each group in the compartment
    #     *weightsC: ndarray of the same dimension of sizeC with the weights associated to each group of the compartment
    #     *lambdaC: regularisation parameter of the compartment
    #     *normC: only 1-norm and 2-norm are allowed


    if lambdaC == 0.0 or sizeC[0]  == 0:
        return 0.0

    cdef long idx1, idx2 = startC
    cdef double xn, r, retval = 0.0
    cdef size_t i, j, k

    if normC == 1 :
        for j in range(0,sizeC.size) :
            idx1 = idx2
            idx2 += sizeC[j]
            xn = 0.0
            for k in range(idx1,idx2) :
                xn += v[k]
            retval += weightsC[j]*xn
    if normC == 2:
        for j in range(0,sizeC.size):
            idx1 = idx2
            idx2 += sizeC[j]
            xn = 0.0
            for i in range(idx1,idx2):
                xn += v[i]*v[i]
            retval += weightsC[j]*sqrt(xn)
    return retval


# Proximal operator of the penalty term
cpdef np.ndarray[np.float64_t] prox_group(np.ndarray[np.float64_t] x, np.int64_t startC, np.ndarray[np.int64_t] sizeC, np.ndarray[np.float64_t] weightsC, np.float64_t lambdaC, np.int64_t normC) :
    # Input:
    #     *v: vector of which we want to compute the proximal
    #     *sizeC: ndarray with the length of each group in the compartment
    #     *weightsC: ndarray of the same dimension of sizeC with the weights associated to each group of the compartment
    #     *lambdaC: regularisation parameter of the compartment
    #     *normC: only 1-norm and 2-norm are allowed

    if lambdaC == 0.0 or sizeC[0] == 0:
        return x.copy()

    cdef np.ndarray[np.float64_t] v
    cdef long idx1, idx2 = startC
    cdef double xn, r, t
    cdef size_t i, j, k

    v = x.copy()
    v[v<0] = 0.0
    if normC == 1 :
        for j in range(0,sizeC.size) :
            idx1 = idx2 + 0
            idx2 += sizeC[j]
            r = weightsC[j] * lambdaC
            for k in range(idx1,idx2) :
                if v[k] <= r:
                    v[k] = 0.0
                else :
                    v[k] -= r
    if normC == 2:
        for j in range(0,sizeC.size):
            idx1 = idx2
            idx2 += sizeC[j]
            xn = 0.0
            for i in range(idx1,idx2) :
                xn += v[i]*v[i]
            xn = sqrt(xn)
            if xn == 0 :
                xn = 1.0

            r = weightsC[j] * lambdaC

            # Soft thresholding
            t = max( xn-r, 0.0 )/xn
            for i in range(idx1,idx2) :
                x[i] *= t
    return v


## General solver
cpdef np.ndarray[np.float64_t] __fista( np.ndarray[np.float64_t] y, A, At, double tol_fun, double tol_x, int max_iter, float verbose, np.ndarray[np.float64_t] x0, omega, proximal) :
    # Solve the regularised least squares problem
    #
    #     argmin_x 0.5*||y - A x||_2^2 + \Omega(x)
    #
    # with the FISTA algorithm described in [1].
    #
    # Notes
    # -----
    # Author: Matteo Frigo - lts5 @ EPFL
    # Acknowledgment: Rafael Carrillo - lts5 @ EPFL
    # References:
    #     [1] `Beck & Teboulle - A Fast Iterative Shrinkage Thresholding
    #         Algorithm for Linear Inverse Problems`



    # Initialization
    cdef:
        size_t iter
        np.ndarray[np.float64_t] xhat
        np.ndarray[np.float64_t] x
        np.ndarray[np.float64_t] res
        np.ndarray[np.float64_t] prev_x
        np.ndarray[np.float64_t] grad
        double prev_obj , told , beta , q , qfval , L , mu , abs_obj , rel_obj , abs_x , rel_x

    res = -y.copy()
    xhat = x0.copy()
    x = np.zeros_like(xhat)
    res += A.dot(xhat)
    xhat = proximal( xhat )
    reg_term = omega( xhat )
    prev_obj = 0.5 * np.linalg.norm(res)**2 + reg_term

    told = 1
    beta = 0.9
    prev_x = xhat.copy()
    grad = np.asarray(At.dot(res))
    qfval = prev_obj

    # Step size computation
    L = ( np.linalg.norm( A.dot(grad) ) / np.linalg.norm(grad) )**2
    mu = 1.9 / L

    # Main loop
    if verbose >= 1 :
        print
        print "      |     ||Ax-y||     |  Cost function    Abs error      Rel error    |     Abs x          Rel x"
        print "------|------------------|-----------------------------------------------|------------------------------"
    iter = 1
    while True :
        if verbose >= 1 :
            print "%4d  |" % iter,
            sys.stdout.flush()

        # Smooth step
        x = xhat - mu*grad

        # Non-smooth step
        x = proximal( x )
        reg_term_x = omega( x )

        # Check stepsize
        tmp = x-xhat
        q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2 + reg_term_x
        res = A.dot(x) - y
        res_norm = np.linalg.norm(res)
        curr_obj = 0.5 * res_norm**2 + reg_term_x

        # Backtracking
        while curr_obj > q :
            if verbose >= 1 :
                print "%4d  | BT" % iter

            # Smooth step
            mu = beta*mu
            x = xhat - mu*grad

            # Non-smooth step
            x = proximal( x )
            reg_term_x = omega( x )

            # Check stepsize
            tmp = x-xhat
            q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2 + reg_term_x
            res = A.dot(x) - y
            res_norm = np.linalg.norm(res)
            curr_obj = 0.5 * res_norm**2 + reg_term_x

        # Global stopping criterion
        abs_obj = abs(curr_obj - prev_obj)
        rel_obj = abs_obj / curr_obj
        abs_x   = np.linalg.norm(x - prev_x)
        rel_x   = abs_x / ( np.linalg.norm(x) + eps )
        if verbose >= 1 :
            print "  %13.7e  |  %13.7e  %13.7e  %13.7e  |  %13.7e  %13.7e" % ( res_norm, curr_obj, abs_obj, rel_obj, abs_x, rel_x )

        if abs_obj < eps :
            criterion = "Absolute tolerance on the objective"
            break
        elif rel_obj < tol_fun :
            criterion = "Relative tolerance on the objective"
            break
        elif abs_x < eps :
            criterion = "Absolute tolerance on the unknown"
            break
        elif rel_x < tol_x :
            criterion = "Relative tolerance on the unknown"
            break
        elif iter >= max_iter :
            criterion = "MAXIMUM NUMBER OF ITERATIONS"
            break

        # FISTA update
        t = 0.5 * ( 1 + sqrt(1+4*told**2) )
        xhat = x + (told-1)/t * (x - prev_x)

        # Gradient computation
        res = A.dot(xhat) - y
        xarr = np.asarray(x)

        grad = np.asarray(<np.ndarray[np.float64_t]> At.dot(res))

        # Update variables
        iter += 1
        prev_obj = curr_obj
        prev_x = x.copy()
        told = t
        qfval = 0.5 * np.linalg.norm(res)**2


    if verbose >= 1 :
        print "< Stopping criterion: %s >" % criterion

    return x
