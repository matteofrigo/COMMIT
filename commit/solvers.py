import numpy as np
from math import sqrt
import sys
eps = np.finfo(float).eps

from commit.proximals import (non_negativity,
                             omega_group_sparsity,
                             prox_group_sparsity,
                             soft_thresholding,
                             projection_onto_l2_ball)
group_sparsity = -1
non_negative = 0
norm1 = 1
norm2 = 2
norminf = np.inf
list_regnorms = [group_sparsity, non_negative, norm1, norm2]
list_group_sparsity_norms = [norm2, norminf]

def init_regularisation(commit_evaluation,
                        regnorms = (non_negative, non_negative, non_negative),
                        structureIC = None, weightsIC = None, group_norm = 2,
                        group_is_ordered = False,
                        lambdas = (.0,.0,.0) ):
    """
    Initialise the data structure that defines Omega in

        argmin_x 0.5*||Ax-y||_2^2 + Omega(x)


    Input
    -----
    commit_evaluation - commit.Evaluation object :
        dictionary and model have to be loaded beforehand.


    regnorms - tuple :
        this sets the penalty term to be used for each compartment.
            Default = (non_negative,non_negative,non_negative).

            regnorms[0] corresponds to the Intracellular compartment
            regnorms[1] corresponds to the Extracellular compartment
            regnorms[2] corresponds to the Isotropic compartment

            Each regnorms[k] must be one of commit.solvers.
                                {group_sparsity, non_negative, norm1, norm2}.

            commit.solvers.group_sparsity considers both the non-overlapping
                and the hierarchical group sparsity (see [1]). This option is
                allowed only in the IC compartment. The mathematical formulation
                of this term is
                $\Omega(x) = \lambda \sum_{g\in G} w_g |x_g|

            commit.solvers.non_negative puts a non negativity constraint on the
                coefficients corresponding to the compartment. This is the
                default option for each compartment

            commit.solvers.norm1 penalises with the 1-norm of the coefficients
                corresponding to the compartment.

            commit.solvers.norm2 penalises with the 2-norm of the coefficients
                corresponding to the compartment.


    structureIC - np.array(list(list)) :
        group structure for the IC compartment.
            This field is necessary only if regterm[0]=commit.solver.group_sparsity.
            Example:
                structureIC = np.array([[0,2,5],[1,3,4],[0,1,2,3,4,5],[6]])

                that is equivalent to
                            [0,1,2,3,4,5]        [6]
                              /       \
                        [0,2,5]       [1,3,4]
                which has two non overlapping groups, one of which is the union
                of two other non-overlapping groups.


    weightsIC - np.array(np.float64) :
        this defines the weights associated to each group of structure IC.


    group_norm - number :
        norm type for the commit.solver.group_sparsity penalisation of the IC compartment.
            Default: group_norm = commit.solver.norm2
            To be chosen among commit.solver.{norm2,norminf}.

    group_is_ordered - boolean :
        True if the streamlines are ordered group-wise, False otherwise.
            Defauls: False
            This option is given for back compatibility with older and internal
            version of the package that required an ordered version of the input
            tractogram.
            If you use QuickBundles(X) to define the group structure you
            shouldn't take care of this option.

            Note: this option will be deprecated in future release and gives a warning.

    lambdas - tuple :
        regularisation parameter for each compartment.
            Default: lambdas = (0.0, 0.0, 0.0)
            The lambdas correspond to the onse described in the mathematical
            formulation of the regularisation term
            $\Omega(x) = lambdas[0]*regnorm[0](x) + lambdas[1]*regnorm[1](x) + lambdas[2]*regnorm[2](x)$
    Notes
    -----
    Author: Matteo Frigo - athena @ inria
    References:
        [1] Jenatton et al. - 'Proximal Methods for Hierarchical Sparse Coding'
    """
    regularisation = {}

    regularisation['startIC']  = 0
    regularisation['sizeIC']   = int( commit_evaluation.DICTIONARY['IC']['nF'])#*len(commit_evaluation.model.ICVFs) )
    regularisation['startEC']  = int( regularisation['sizeIC'] )
    regularisation['sizeEC']   = int( commit_evaluation.DICTIONARY['EC']['nE'] )
    regularisation['startISO'] = int( regularisation['sizeIC'] + regularisation['sizeEC'] )
    regularisation['sizeISO']  = int( commit_evaluation.DICTIONARY['nV'])#*len(commit_evaluation.model.d_ISOs) )

    regularisation['normIC']  = regnorms[0]
    regularisation['normEC']  = regnorms[1]
    regularisation['normISO'] = regnorms[2]

    regularisation['lambdaIC']  = float( lambdas[0] )
    regularisation['lambdaEC']  = float( lambdas[1] )
    regularisation['lambdaISO'] = float( lambdas[2] )

    # Solver-specific fields
    regularisation['group_is_ordered'] = group_is_ordered  # This option will be deprecated in future release
    regularisation['structureIC']      = structureIC
    regularisation['weightsIC']        = weightsIC
    regularisation['group_norm']       = group_norm

    return regularisation


def regularisation2omegaprox(regularisation):
    lambdaIC  = float(regularisation.get('lambdaIC'))
    lambdaEC  = float(regularisation.get('lambdaEC'))
    lambdaISO = float(regularisation.get('lambdaISO'))
    if lambdaIC < 0.0 or lambdaEC < 0.0 or lambdaISO < 0.0:
        raise ValueError('Negative regularisation parameters are not allowed')

    normIC  = regularisation.get('normIC')
    normEC  = regularisation.get('normEC')
    normISO = regularisation.get('normISO')
    if not normIC in list_regnorms:
        raise ValueError('normIC must be one of commit.solvers.{group_sparsity,non_negative,norm1,norm2}')
    if not normEC in list_regnorms:
        raise ValueError('normEC must be one of commit.solvers.{group_sparsity,non_negative,norm1,norm2}')
    if not normISO in list_regnorms:
        raise ValueError('normISO must be one of commit.solvers.{group_sparsity,non_negative,norm1,norm2}')

    ## NNLS case
    if (lambdaIC == 0.0 and lambdaEC == 0.0 and lambdaISO == 0.0) or (normIC == non_negative and normEC == non_negative and normISO == non_negative):
        omega = lambda x: 0.0
        prox  = lambda x: non_negativity(x, 0, len(x))
        return omega, prox

    ## All other cases
    # Intracellular Compartment
    startIC = regularisation.get('startIC')
    sizeIC  = regularisation.get('sizeIC')
    if lambdaIC == 0.0:
        omegaIC = lambda x: 0.0
        proxIC  = lambda x: non_negativity(x, startIC, sizeIC)
    elif normIC == norm2:
        omegaIC = lambda x: lambdaIC * np.linalg.norm(x[startIC:sizeIC])
        proxIC  = lambda x: projection_onto_l2_ball(x, lambdaIC, startIC, sizeIC)
    elif normIC == norm1:
        omegaIC = lambda x: lambdaIC * sum( x[startIC:sizeIC] )
        proxIC  = lambda x: soft_thresholding(x, lambdaIC, startIC, sizeIC)
    elif normIC == non_negative:
        omegaIC = lambda x: 0.0
        proxIC  = lambda x: non_negativity(x, startIC, sizeIC)
    elif normIC == group_sparsity:
        weightsIC   = regularisation.get('weightsIC')
        structureIC = regularisation.get('structureIC')
        if regularisation.get('group_is_ordered'): # This option will be deprecated in future release
            raise DeprecationWarning('The ordered group structure will be deprecated. Consider using the structureIC field for defining the group structure.')
            bundles = np.cumsum(np.insert(sizeIC,0,0))
            structureIC = np.array([range(bundles[k],bundles[k+1]) for k in range(0,len(bundles)-1)])
            regularisation['structureIC'] = structureIC
            del bundles
        if not len(structureIC) == len(weightsIC):
            raise ValueError('Number of groups and weights do not coincide.')
        group_norm = regularisation.get('group_norm')
        if not group_norm in list_group_sparsity_norms:
            raise ValueError('Wrong norm in the structured sparsity term. Choose between %s.' % str(list_group_sparsity_norms))

        omegaIC = lambda x: omega_group_sparsity( x, structureIC, weightsIC, lambdaIC, normIC )
        proxIC  = lambda x:  prox_group_sparsity( x, structureIC, weightsIC, lambdaIC, normIC )
    else:
        raise ValueError('Type of regularisation for IC compartment not recognized.')


    # Extracellular Compartment
    startEC = regularisation.get('startEC')
    sizeEC  = regularisation.get('sizeEC')
    if lambdaEC == 0.0:
        omegaEC = lambda x: 0.0
        proxEC  = lambda x: non_negativity(x, startEC, sizeEC)
    elif normEC == norm2:
        omegaEC = lambda x: lambdaEC * np.linalg.norm(x[startEC:sizeEC])
        proxEC  = lambda x: projection_onto_l2_ball(x, lambdaEC, startEC, sizeEC)
    elif normEC == norm1:
        omegaEC = lambda x: lambdaEC * sum( x[startEC:sizeEC] )
        proxEC  = lambda x: soft_thresholding(x, lambdaEC, startEC, sizeEC)
    elif normEC == non_negative:
        omegaEC = lambda x: 0.0
        proxEC  = lambda x: non_negativity(x, startEC, sizeEC)
    else:
        raise ValueError('Type of regularisation for EC compartment not recognized.')

    # Isotropic Compartment
    startISO = regularisation.get('startISO')
    sizeISO  = regularisation.get('sizeISO')
    if lambdaISO == 0.0:
        omegaISO = lambda x: 0.0
        proxISO  = lambda x: non_negativity(x, startISO, sizeISO)
    elif normISO == norm2:
        omegaISO = lambda x: lambdaISO * np.linalg.norm(x[startISO:sizeISO])
        proxISO  = lambda x: projection_onto_l2_ball(x, lambdaISO, startISO, sizeISO)
    elif normISO == norm1:
        omegaISO = lambda x: lambdaISO * sum( x[startISO:sizeISO] )
        proxISO  = lambda x: soft_thresholding(x, lambdaISO, startISO, sizeISO)
    elif normISO == non_negative:
        omegaISO = lambda x: 0.0
        proxISO  = lambda x: non_negativity(x, startISO, sizeISO)
    else:
        raise ValueError('Type of regularisation for ISO compartment not recognized.')

    omega = lambda x: omegaIC(x) + omegaEC(x) + omegaISO(x)
    prox = lambda x: non_negative(proxIC(proxEC(proxISO(x)))) # non negativity is redunduntly forced

    return omega, prox

def solve(y, A, At, tol_fun = 1e-4, tol_x = 1e-6, max_iter = 1000, verbose = 1, x0 = None, regularisation = None, debug_mode = False, debug_parameters = None):
    """
    Solve the regularised least squares problem

        argmin_x 0.5*||Ax-y||_2^2 + Omega(x)

    with the Omega described by 'regularisation'.

    Check the documentation of commit.solvers.init_regularisation to see how to
    solve a specific problem.

    Notes
    -----
    Author: Matteo Frigo - athena @ inria
    """
    if regularisation is None:
        omega = lambda x: 0.0
        prox  = lambda x: non_negativity(x, 0, len(x))
    else:
        omega, prox = regularisation2omegaprox(regularisation)

    if x0 is None:
        x0 = np.ones(A.shape[1])

    if not debug_mode:
        return fista( y, A, At, tol_fun, tol_x, max_iter, verbose, x0, omega, prox)
    else:
        return fista_debug( y, A, At, tol_fun, tol_x, max_iter, verbose, x0, omega, prox, debug_parameters)



def fista( y, A, At, tol_fun, tol_x, max_iter, verbose, x0, omega, proximal) :
    """
    Solve the regularised least squares problem

        argmin_x 0.5*||Ax-y||_2^2 + Omega(x)

    with the FISTA algorithm described in [1].

    The penalty term and its proximal operator must be defined in such a way
    that they already contain the regularisation parameter.

    Notes
    -----
    Author: Matteo Frigo - athena @ inria
    Acknowledgment: Rafael Carrillo - lts5 @ EPFL
    References:
        [1] Beck & Teboulle - `A Fast Iterative Shrinkage Thresholding
            Algorithm for Linear Inverse Problems`
    """

    # Initialization
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
            criterion = "Maximum number of iterations"
            break

        # FISTA update
        t = 0.5 * ( 1 + sqrt(1+4*told**2) )
        xhat = x + (told-1)/t * (x - prev_x)

        # Gradient computation
        res = A.dot(xhat) - y
        xarr = np.asarray(x)

        grad = np.asarray(At.dot(res))

        # Update variables
        iter += 1
        prev_obj = curr_obj
        prev_x = x.copy()
        told = t
        qfval = 0.5 * np.linalg.norm(res)**2


    if verbose >= 1 :
        print "< Stopping criterion: %s >" % criterion

    opt_details = {}
    opt_details['residual'] = res_norm
    opt_details['cost_function'] = curr_obj
    opt_details['abs_cost'] = abs_obj
    opt_details['rel_cost'] = rel_obj
    opt_details['abs_x'] = abs_x
    opt_details['rel _x'] = rel_x
    opt_details['iterations'] = iter
    opt_details['stopping_criterion'] = criterion

    return x, opt_details

def fista_debug( y, A, At, tol_fun, tol_x, max_iter, verbose, x0, omega, proximal, debug_parameters) :
    """
    Debug mode for the fista solver.

    Notes
    -----
    Author: Matteo Frigo - athena @ inria
    """
    import cPickle
    from os.path import join as pjoin
    verbose = True

    debug_parameters = {} if debug_parameters is None else debug_parameters

    dosave       = False if debug_parameters.get('dosave') is None else debug_parameters.get('dosave')
    savestep     = 10 if debug_parameters.get('savestep') is None else debug_parameters.get('savestep')
    mit          = None if debug_parameters.get('commit_evaluation') is None else debug_parameters.get('commit_evaluation')
    savepath     = './debug_mode' if mit is None else pjoin(mit.get_config('TRACKING_path'), 'debug_mode')
    track_nzeros = False if debug_parameters.get('track_nzeros') is None else debug_parameters.get('track_nzeros')
    if track_nzeros:
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure()
        plt.axis([0,max_iter,0,A.shape[1]])
        i = 0
        x = list()
        y = list()

    print '#################'
    print 'DEBUG MODE SOLVER'
    print '#################'
    print 'Shape of the dictionary: \t%s' % A.shape
    print 'Length of x0:\t%s' % x0.shape
    print 'Length of y:\t%s' % y.shape
    print
    print 'Maximum number of iterations:\t%s' % max_iter

    print 'Verbose mode: on'
    print 'Testing omega ...'
    testx = np.ones(A.shape[1])
    try:
        tmp = omega(testx)
        del tmp
        print '\tTest passed'
    except:
        print '\tCannot call Omega. Check its definition'
        return
    print 'Testing proximal ...'
    try:
        tmp = proximal(testx)
        del tmp
        print '\tTest passed'
    except:
        print '\tCannot call Proximal. Check its definition'
        return
    del testx


    # Initialization
    res = -y.copy()
    xhat = x0.copy()
    x = np.zeros_like(xhat)
    res += A.dot(xhat)
    xhat = proximal( xhat )
    reg_term = omega( xhat )
    prev_obj = 0.5 * np.linalg.norm(res)**2 + reg_term

    told = 1
    beta = 0.9
    print 'Backtracking multiplication constant: %s' % beta
    prev_x = xhat.copy()
    grad = np.asarray(At.dot(res))
    qfval = prev_obj

    # Step size computation
    L = ( np.linalg.norm( A.dot(grad) ) / np.linalg.norm(grad) )**2
    mu = 1.9 / L

    # Main loop
    if verbose >= 1 :
        print
        print "      |     ||Ax-y||     |  Cost function    Abs error      Rel error    |     Abs x          Rel x     "
        print "------|------------------|-----------------------------------------------|------------------------------"
    end_regression = False
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
            print "BT  |",
            sys.stdout.flush()

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
            print "  %13.7e  |  %13.7e  %13.7e  %13.7e  |  %13.7e  %13.7e" % ( res_norm, curr_obj, abs_obj, rel_obj, abs_x, rel_x )

        # Global stopping criterion
        abs_obj = abs(curr_obj - prev_obj)
        rel_obj = abs_obj / curr_obj
        abs_x   = np.linalg.norm(x - prev_x)
        rel_x   = abs_x / ( np.linalg.norm(x) + eps )
        if verbose >= 1 :
            print "  %13.7e  |  %13.7e  %13.7e  %13.7e  |  %13.7e  %13.7e" % ( res_norm, curr_obj, abs_obj, rel_obj, abs_x, rel_x )

        if abs_obj < eps :
            criterion = "Absolute tolerance on the objective"
            end_regression = True
        elif rel_obj < tol_fun :
            criterion = "Relative tolerance on the objective"
            end_regression = True
        elif abs_x < eps :
            criterion = "Absolute tolerance on the unknown"
            end_regression = True
        elif rel_x < tol_x :
            criterion = "Relative tolerance on the unknown"
            end_regression = True
        elif iter >= max_iter :
            criterion = "Maximum number of iterations"
            end_regression = True

        if end_regression:
            break

        # FISTA update
        t = 0.5 * ( 1 + sqrt(1+4*told**2) )
        xhat = x + (told-1)/t * (x - prev_x)

        # Gradient computation
        res = A.dot(xhat) - y
        xarr = np.asarray(x)

        grad = np.asarray(At.dot(res))

        if dosave and iter//savestep == 0:
            np.savetxt(pjoin(savepath,'x_%d.txt' % iter), x, header='After %d iterations' % iter )

        if track_nzeros:
            x.append(iter);
            y.append(np.count_nonzero(x));
            plt.scatter(iter, np.count_nonzero(x))
            plt.show()
            plt.pause(1e-5)

        # Update variables
        iter += 1
        prev_obj = curr_obj
        prev_x = x.copy()
        told = t
        qfval = 0.5 * np.linalg.norm(res)**2

    if verbose >= 1 :
        print "< Stopping criterion: %s >" % criterion

    opt_details = {}
    opt_details['residual'] = res_norm
    opt_details['cost_function'] = curr_obj
    opt_details['abs_cost'] = abs_obj
    opt_details['rel_cost'] = rel_obj
    opt_details['abs_x'] = abs_x
    opt_details['rel _x'] = rel_x
    opt_details['iterations'] = iter
    opt_details['stopping_criterion'] = criterion

    return x, opt_details
