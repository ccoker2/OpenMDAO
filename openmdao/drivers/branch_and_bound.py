""" Class definition for the Branch_and_Bound driver. This driver can be run
standalone or plugged into the AMIEGO driver.

This is the branch and bound algorithm that maximizes the constrained
expected improvement function and returns an integer infill point. The
algorithm uses the relaxation techniques proposed by Jones et.al. on their
paper on EGO,1998. This enables the algorithm to use any gradient-based
approach to obtain a global solution. Also, to satisfy the integer
constraints, a new branching scheme has been implemented.

Developed by Satadru Roy
School of Aeronautics & Astronautics
Purdue University, West Lafayette, IN 47906
July, 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""

from __future__ import print_function

from collections import OrderedDict
from six import iteritems
from six.moves import range
from random import uniform

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

from openmdao.core.driver import Driver
from openmdao.core.mpi_wrap import debug
from openmdao.surrogate_models.kriging import KrigingSurrogate
from openmdao.test.util import set_pyoptsparse_opt
from openmdao.util.concurrent import concurrent_eval
from openmdao.util.record_util import create_local_meta, update_local_meta

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

class DummyComm(object):
    """ Do-nothing communicator for pyoptsparse so that we can run it
    independently on each process."""

    rank = 0

    def gather(self, name, root=None):
        """ Signature for gather"""
        return name

    def bcast(self, name, root=None):
        """ Signature for bcast"""
        return name

    def recv(self, name, tag=None):
        """ Signature for recv"""
        return name

def snopt_opt(objfun, desvar, lb, ub, ncon, title=None, options=None,
              sens=None, jac=None):
    """ Wrapper function for running a SNOPT optimization through
    pyoptsparse."""

    if OPTIMIZER:
        from pyoptsparse import Optimization
    else:
        raise(RuntimeError, 'Need pyoptsparse to run the SNOPT sub optimizer.')

    opt_prob = Optimization(title, objfun, comm=DummyComm())

    ndv = len(desvar)

    opt_prob.addVarGroup('x', ndv, type='c', value=desvar.flatten(), lower=lb.flatten(),
                         upper=ub.flatten())
    opt_prob.addConGroup('con', ncon, upper=np.zeros((ncon)))#, linear=True, wrt='x',
                         #jac={'x' : jac})
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

    x = sol.getDVs()['x']
    f = sol.objectives['obj'].value
    success_flag = sol.optInform['value'] < 2

    return x, f, success_flag


class Branch_and_Bound(Driver):
    """ Class definition for the Branch_and_Bound driver. This driver can be run
    standalone or plugged into the AMIEGO driver.

    This is the branch and bound algorithm that maximizes the constrained
    expected improvement function and returns an integer infill point. The
    algorithm uses the relaxation techniques proposed by Jones et.al. on
    their paper on EGO,1998. This enables the algorithm to use any
    gradient-based approach to obtain a global solution. Also, to satisfy the
    integer constraints, a new branching scheme has been implemented.
    """

    def __init__(self):
        """Initialize the Branch_and_Bound driver."""

        super(Branch_and_Bound, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = False
        self.supports['mixed_integer'] = False

        # Default surrogate. User can slot a modified one, but it essentially
        # has to provide what Kriging provides.
        self.surrogate = KrigingSurrogate

        # TODO: is 1e-3 the right and best value for ftol?

        # Options
        opt = self.options
        opt.add_option('active_tol', 1.0e-6, lower=0.0,
                       desc='Tolerance (2-norm) for triggering active set '
                       'reduction.')
        opt.add_option('atol', 1.0e-6, lower=0.0,
                       desc='Absolute tolerance (inf-norm) of upper minus '
                       'lower bound for termination.')
        opt.add_option('con_tol', 1.0e-6, lower=0.0,
                       desc='Constraint thickness.')
        opt.add_option('concave_EI', False,
                       desc='Set to True to apply a transformation to make the '
                       'objective function concave.')
        opt.add_option('disp', True,
                       desc='Set to False to prevent printing of iteration '
                       'messages.')
        opt.add_option('ftol', 1.0e-4, lower=0.0,
                       desc='Absolute tolerance for sub-optimizations.')
        opt.add_option('use_surrogate', False,
                       desc='Use surrogate model for the optimization. Training '
                       'data must be supplied.')
        opt.add_option('local_search', False,
                        desc='Set to True if local search needs to be performed '
                        'in step 2.')

        # Initial Sampling
        # TODO: Somehow slot an object that generates this (LHC for example)
        self.sampling = {}

        self.dvs = []
        self.size = 0
        self.idx_cache = {}
        self.obj_surrogate = None
        self.con_surrogate = []
        self.record_name = 'B&B'

        # Amiego will set this to True if we have found a minimum.
        self.eflag_MINLPBB = False

        # Amiego retrieves optimal design and optimum upon completion.
        self.xopt = None
        self.fopt = None

        # When this is slotted into AMIEGO, this will be set to False.
        self.standalone = True

        # Switch between pyoptsparse and scipy/slsqp
        self.pyopt = True

        # Declare stuff we need to pass to objective callback functions
        self.current_surr = None

    def _setup(self):
        """  Initialize whatever we need."""
        super(Branch_and_Bound, self)._setup()

        # Size our design variables.
        j = 0
        for name, val in iteritems(self.get_desvars()):
            self.dvs.append(name)
            try:
                size = len(np.asarray(val.val))
            except TypeError:
                size = 1
            self.idx_cache[name] = (j, j+size)
            j += size
        self.size = j

        # Lower and Upper bounds
        self.xI_lb = np.empty((self.size))
        self.xI_ub = np.empty((self.size))
        dv_dict = self._desvars
        for var in self.dvs:
            i, j = self.idx_cache[var]
            self.xI_lb[i:j] = dv_dict[var]['lower']
            self.xI_ub[i:j] = dv_dict[var]['upper']

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the
            min and max processors usable by this `Driver`.
        """
        return (1, None)

    def run(self, problem):
        """Execute the Branch_and_Bound method.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        obj_surrogate = self.obj_surrogate
        con_surrogate = self.con_surrogate
        n_i = self.size
        atol = self.options['atol']
        ftol = self.options['ftol']
        disp = self.options['disp']

        # Metadata Setup
        self.metadata = create_local_meta(None, self.record_name)
        self.iter_count = 1
        update_local_meta(self.metadata, (self.iter_count, ))

        # Use surrogates for speed.
        # Note, when not standalone, the parent driver provides the surrogate
        # model.
        if self.standalone and self.options['use_surrogate']:

            n_train = self.sampling[self.dvs[0]].shape[0]
            x_i = []
            x_i_hat = []
            obj = []
            cons = {}
            for con in self.get_constraint_metadata():
                cons[con] = []

            system = self.root
            metadata = self.metadata

            for i_train in range(n_train):

                xx_i = np.empty((self.size, ))
                xx_i_hat = np.empty((self.size, ))
                for var in self.dvs:
                    lower = self._desvars[var]['lower']
                    upper = self._desvars[var]['upper']
                    i, j = self.idx_cache[var]
                    x_i_0 = self.sampling[var][i_train, :]

                    xx_i[i:j] = np.round(lower + x_i_0 * (upper - lower))
                    xx_i_hat[i:j] = (xx_i[i:j] - lower)/(upper - lower)

                x_i.append(xx_i)
                x_i_hat.append(xx_i_hat)

            # Run each case and extract obj/con
            for i_run in range(len(x_i)):

                # Set design variables
                for var in self.dvs:
                    i, j = self.idx_cache[var]
                    self.set_desvar(var, x_i[i_run][i:j])

                with system._dircontext:
                    system.solve_nonlinear(metadata=metadata)

                # Get objectives and constraints
                current_objs = self.get_objectives()
                obj_name = list(current_objs.keys())[0]
                current_obj = current_objs[obj_name].copy()
                obj.append(current_obj)
                for name, value in iteritems(self.get_constraints()):
                    cons[name].append(value.copy())

            self.obj_surrogate = obj_surrogate = self.surrogate()
            obj_surrogate.use_snopt = True
            obj_surrogate.train(x_i_hat, obj, normalize=False)
            obj_surrogate.y = obj
            obj_surrogate.lb_org = self.xI_lb
            obj_surrogate.ub_org = self.xI_ub
            obj_surrogate.lb = np.zeros((n_i))
            obj_surrogate.ub = np.zeros((n_i))

            self.con_surrogate = con_surrogate = []
            for name, val in iteritems(cons):
                val = np.array(val)
                for j in range(val.shape[1]):
                    con_surr = self.surrogate()
                    con_surr.use_snopt = True
                    con_surr.train(x_i_hat, val[:, j:j+1], normalize=False)

                    con_surr.y = val[:, j:j+1]
                    con_surr.lb_org = self.xI_lb
                    con_surr.ub_org = self.xI_ub
                    con_surr.lb = np.zeros((n_i))
                    con_surr.ub = np.zeros((n_i))
                    con_surrogate.append(con_surr)

        # Calculate intermediate statistics. This stuff used to be stored in
        # the Modelinfo object, but more convenient to store it in the
        # Kriging surrogate.

        n_train = obj_surrogate.X.shape[0]
        one = np.ones([n_train,1])
        if obj_surrogate:

            # TODO: mvp in Kriging
            # R_inv = obj_surrogate.Vh.T.dot(np.einsum('i,ij->ij',
                                                            #    obj_surrogate.S_inv,
                                                            #    obj_surrogate.U.T))
            # obj_surrogate.R_inv = R_inv

            # obj_surrogate.mu = np.dot(one.T,np.dot(R_inv,obj_surrogate.Y))/np.dot(one.T,np.dot(R_inv,one))
            #
            # obj_surrogate.SigmaSqr = np.dot((obj_surrogate.Y - one*obj_surrogate.mu).T,np.dot(R_inv,(obj_surrogate.Y-one*obj_surrogate.mu)))/n_train

            # TODO: norm type, should probably always be 2 (Yes for sure)
            obj_surrogate.p = 2

            # obj_surrogate.c_r = np.dot(R_inv,(obj_surrogate.Y-one*obj_surrogate.mu))

            ## This is also done in Ameigo. TODO: just do it once.
            obj_surrogate.y_best = np.min(obj_surrogate.Y) #TODO If you are using Y value from obj_surrogate, no need to normalize in the calc_conEI_norm function

            ## This is the rest of the interface that any "surrogate" needs to contain.
            ##obj_surrogate.X = surrogate.X
            ##obj_surrogate.ynorm = surrogate.Y
            ##obj_surrogate.thetas = surrogate.thetas
            ##obj_surrogate.X_std = obj_surrogate.X_std.reshape(num_xI,1)
            ##obj_surrogate.X_mean = obj_surrogate.X_mean.reshape(num_xI,1)

        for con_surr in con_surrogate:

            # R_inv = con_surr.Vh.T.dot(np.einsum('i,ij->ij',
            #                                              con_surr.S_inv,
            #                                              con_surr.U.T))
            # con_surr.R_inv = R_inv
            # con_surr.mu = np.dot(one.T,np.dot(R_inv,con_surr.Y))/np.dot(one.T,np.dot(R_inv,one))
            # con_surr.SigmaSqr = np.dot((con_surr.y - one*con_surr.mu).T,np.dot(R_inv,(con_surr.y-one*con_surr.mu)))/n_train
            #
            con_surr.p = 2
            # con_surr.c_r = np.dot(R_inv,(con_surr.y-one*con_surr.mu))

        #----------------------------------------------------------------------
        # Step 1: Initialize
        #----------------------------------------------------------------------

        terminate = False
        num_des = len(self.xI_lb)
        node_num = 0

        # Initial B&B bounds are infinite.
        LBD = -np.inf
        LBD_prev =- np.inf

        # copy our desvars' user specified upper and lower bounds
        # FIXME: is this copy really needed here since we copy these again inside the loop?
        xL_iter = self.xI_lb.copy()
        xU_iter = self.xI_ub.copy()

        # Initial optimal objective and solution
        # Randomly generate an integer point
        # TODO Generate as many random points as number of design variables
        xopt = np.round(xL_iter + np.random.random(num_des)*(xU_iter - xL_iter)).reshape(num_des)
        # Use this one for verification against matlab
        # xopt = 2.0*np.ones((num_des))
        fopt = self.objective_callback(xopt)
        self.eflag_MINLPBB = True
        UBD = fopt

        # This stuff is just for printing.
        par_node = 0

        # Active set fields:
        #     Aset = [[NodeNumber, lb, ub, LBD, UBD], [], ..]
        # Each node is a list.
        active_set = []

        # Number of nodes to evaluate concurrently
        comm = problem.root.comm
        n_proc = comm.size - 1
        if n_proc < 2:
            comm = None
            n_proc = 1

        # # Initial node. This is the data structure we pass into the
        # # concurrent evaluator. TODO: wonder if we can clean this up.
        # args = [(xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
        #         xopt, node_num)]

        # # Initial number of nodes based on number of available procs
        args = init_nodes(n_proc, xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                xopt)

        while not terminate:

            # Branch and Bound evaluation of a set of nodes, starting with the initial one.
            # When executed in serial, only a single node is evaluted.
            cases = [(arg, None) for arg in args]
            results = concurrent_eval(self.evaluate_node, cases,
                                      comm, allgather=True)

            # Put all the new nodes into active set.
            for result in results:
                new_UBD, new_fopt, new_xopt, new_nodes = result[0]

                # Save stats for the best case.
                if new_UBD < UBD:
                    UBD = new_UBD
                    fopt = new_fopt
                    xopt = new_xopt

                # TODO: Should we extend the active set with all the cases we
                # ran, or just the best one. All for now.
                active_set.extend(new_nodes)
                node_num += len(new_nodes)

            # Update active set: Removes all nodes worse than the best new node.
            if len(active_set) >= 1:
                active_set = update_active_set(active_set, UBD)

            # Termination
            if len(active_set) >= 1:
                # Update LBD and select the current rectangle

                args = []

                # Grab the best nodes, as many as we have processors.
                n_nodes = np.min((n_proc, len(active_set)))
                for j in range(n_nodes):

                    # a. Set LBD as lowest in the active set
                    all_LBD = [item[3] for item in active_set]
                    LBD = min(all_LBD)

                    ind_LBD = all_LBD.index(LBD)
                    LBD_prev = LBD

                    # b. Select the lowest LBD node as the current node
                    par_node, xL_iter, xU_iter, _, _ = active_set[ind_LBD]
                    self.iter_count += 1

                    args.append((xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                                 xopt, node_num))

                    # c. Delete the selected node from the Active set of nodes
                    del active_set[ind_LBD]

                    #--------------------------------------------------------------
                    #Step 7: Check for convergence
                    #--------------------------------------------------------------
                    diff = np.abs(UBD - LBD)
                    if diff < atol:
                        terminate = True
                        if disp:
                            print("="*85)
                            print("Terminating! Absolute difference between the upper " + \
                                  "and lower bound is below the tolerence limit.")
            else:
                terminate = True
                if disp:
                    print("="*85)
                    print("Terminating! No new node to explore.")
                    print("Max Node", node_num)

        # Finalize by putting optimal value back into openMDAO
        if self.standalone:

            for var in self.dvs:
                i, j = self.idx_cache[var]
                self.set_desvar(var, xopt[i:j])

            update_local_meta(metadata, (self.iter_count, ))

            with system._dircontext:
                system.solve_nonlinear(metadata=metadata)

        else:
            self.xopt = xopt
            self.fopt = fopt

    def evaluate_node(self, xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD,
                      fopt, xopt, node_num):
        """ Branch and Bound step on a single node. This function
        encapsulates the portion of the code that runs in parallel."""

        active_tol = self.options['active_tol']
        local_search = self.options['local_search']
        disp = self.options['disp']
        obj_surrogate = self.obj_surrogate
        con_surrogate = self.con_surrogate
        num_des = len(self.xI_lb)

        new_nodes = []

        #Keep this to 0.49 to always round towards bottom-left
        xloc_iter = np.round(xL_iter + 0.49*(xU_iter - xL_iter))
        floc_iter = self.objective_callback(xloc_iter)

        efloc_iter = True
        if local_search:
            if np.abs(floc_iter) > active_tol: #Perform at non-flat starting point
                #--------------------------------------------------------------
                #Step 2: Obtain a local solution
                #--------------------------------------------------------------
                # Using a gradient-based method here.
                # TODO: Make it more pluggable.
                # TODO: Use SNOPT [Not a priority-Not going to use local search anytime soon]
                xC_iter = xloc_iter
                bnds = [(xL_iter[ii], xU_iter[ii]) for ii in range(num_des)]

                optResult = minimize(self.objective_callback, xC_iter,
                                     method='SLSQP', bounds=bnds,
                                     options={'ftol' : ftol})

                xloc_iter = np.round(optResult.x.reshape(num_des))
                floc_iter = self.objective_callback(xloc_iter)

                if not optResult.success:
                    efloc_iter = False
                    floc_iter = np.inf
                else:
                    efloc_iter = True

        #--------------------------------------------------------------
        # Step 3: Partition the current rectangle as per the new
        # branching scheme.
        #--------------------------------------------------------------
        child_info = np.zeros([2, 3])
        dis_flag = [' ', ' ']

        # Choose
        l_iter = (xU_iter - xL_iter).argmax()

        if xloc_iter[l_iter]<xU_iter[l_iter]:
            delta = 0.5 #0<delta<1
        else:
            delta = -0.5 #-1<delta<0

        for ii in range(2):
            lb = xL_iter.copy()
            ub = xU_iter.copy()
            if ii == 0:
                ub[l_iter] = np.floor(xloc_iter[l_iter]+delta)
            elif ii == 1:
                lb[l_iter] = np.ceil(xloc_iter[l_iter]+delta)

            if np.linalg.norm(ub - lb) > active_tol: #Not a point
                #--------------------------------------------------------------
                # Step 4: Obtain an LBD of f in the newly created node
                #--------------------------------------------------------------
                S4_fail = False
                x_comL, x_comU, Ain_hat, bin_hat = gen_coeff_bound(lb, ub, obj_surrogate)
                sU, eflag_sU = self.maximize_S(x_comL, x_comU, Ain_hat, bin_hat,
                                               obj_surrogate)

                if eflag_sU:
                    yL, eflag_yL = self.minimize_y(x_comL, x_comU, Ain_hat, bin_hat,
                                                   obj_surrogate)

                    if eflag_yL:
                        NegEI = calc_conEI_norm([], obj_surrogate, SSqr=sU, y_hat=yL)

                        M = len(self.con_surrogate)
                        EV = np.zeros([M, 1])

                        # Expected constraint violation
                        for mm in range(M):
                            x_comL, x_comU, Ain_hat, bin_hat = gen_coeff_bound(lb, ub, con_surrogate[mm])
                            sU_g, eflag_sU_g = self.maximize_S(x_comL, x_comU, Ain_hat,
                                                               bin_hat, con_surrogate[mm])

                            if eflag_sU_g:
                                yL_g, eflag_yL_g = self.minimize_y(x_comL, x_comU, Ain_hat,
                                                                   bin_hat, con_surrogate[mm])
                                if eflag_yL_g:
                                    EV[mm] = calc_conEV_norm(None,
                                                             con_surrogate[mm],
                                                             gSSqr=-sU_g,
                                                             g_hat=yL_g)
                                else:
                                    S4_fail = True
                                    break
                            else:
                                S4_fail = True
                                break
                    else:
                        S4_fail = True
                else:
                    S4_fail = True

                # Convex approximation failed!
                if S4_fail:
                    if efloc_iter:
                        LBD_NegConEI = LBD_prev
                    else:
                        LBD_NegConEI = np.inf
                    dis_flag[ii] = 'F'
                else:
                    EV_mean = np.mean(EV, axis=0)
                    EV_std = np.std(EV, axis=0)
                    EV_std[EV_std == 0.] = 1.
                    EV_norm = (EV - EV_mean) / EV_std
                    LBD_NegConEI = max(NegEI/(1.0 + np.sum(EV_norm)), LBD_prev)

                #--------------------------------------------------------------
                # Step 5: Store any new node inside the active set that has LBD
                # lower than the UBD.
                #--------------------------------------------------------------

                if LBD_NegConEI < UBD - 1.0e-6:
                    node_num += 1
                    new_node = [node_num, lb, ub, LBD_NegConEI, floc_iter]
                    new_nodes.append(new_node)
                    child_info[ii] = np.array([node_num, LBD_NegConEI, floc_iter])
                else:
                    child_info[ii] = np.array([par_node, LBD_NegConEI, floc_iter])
                    dis_flag[ii] = 'X' #Flag for child created but not added to active set (fathomed)
            else:
                if ii == 1:
                    xloc_iter = ub
                    floc_iter = self.objective_callback(xloc_iter)
                child_info[ii] = np.array([par_node, np.inf, floc_iter])
                dis_flag[ii] = 'x' #Flag for No child created

            #Update the active set whenever better solution found
            if floc_iter < UBD:
                UBD = floc_iter
                fopt = floc_iter
                xopt = xloc_iter.copy().reshape(num_des)

        if disp:
            if (self.iter_count-1) % 25 == 0:
                # Display output in a tabular format
                print("="*85)
                print("%19s%12s%14s%21s" % ("Global", "Parent", "Child1", "Child2"))
                template = "%s%8s%10s%8s%9s%11s%10s%11s%11s"
                print(template % ("Iter", "LBD", "UBD", "Node", "Node1", "LBD1",
                                  "Node2", "LBD2", "Flocal"))
                print("="*85)
            template = "%3d%10.2f%10.2f%6d%8d%1s%13.2f%8d%1s%13.2f%9.2f"
            print(template % (self.iter_count, LBD, UBD, par_node, child_info[0, 0],
                              dis_flag[0], child_info[0, 1], child_info[1, 0],
                              dis_flag[1], child_info[1, 1], child_info[1, 2]))

        return UBD, fopt, xopt, new_nodes

    def objective_callback(self, xI):
        """ Callback for main problem evaluation."""
        obj_surrogate = self.obj_surrogate

        # When run stanalone, the objective is the model objective.
        if self.standalone:
            if self.options['use_surrogate']:

                #FIXME : This will change if used standalone
                x0I_hat = (xI - self.xI_lb)/(self.xI_ub - self.xI_lb).reshape((len(xI), 1))

                f = obj_surrogate.predict(x0I_hat, normalize=False)[0]

            else:
                raise NotImplementedError()

        # When run under AMEIGO, objective is the expected improvment
        # function with modifications to make it concave.
        else:
            #ModelInfo_obj=param[0];ModelInfo_g=param[1];con_fac=param[2];flag=param[3]

            X = obj_surrogate.X
            k = np.shape(X)[1]
            lb = obj_surrogate.lb
            ub = obj_surrogate.ub

            # Normalized as per the convention in Kriging of openmdao
            xval = (xI - obj_surrogate.X_mean.flatten())/obj_surrogate.X_std.flatten()
            # xval = (xI - obj_surrogate.lb_org.flatten())/(obj_surrogate.ub_org.flatten() - obj_surrogate.lb_org.flatten())


            NegEI = calc_conEI_norm(xval, obj_surrogate)

            con_surrogate = self.con_surrogate
            M = len(con_surrogate)
            EV = np.zeros([M, 1])
            if M>0:
                for mm in range(M):
                    EV[mm] = calc_conEV_norm(xval, con_surrogate[mm])

            EV_mean = np.mean(EV, axis=0)
            EV_std = np.std(EV, axis=0)
            EV_std[EV_std == 0.] = 1.
            EV_norm = (EV - EV_mean) / EV_std
            conNegEI = NegEI/(1.0 + np.sum(EV_norm))

            P = 0.0

            if self.options['concave_EI']: #Locally makes ei concave to get rid of flat objective space
                for ii in range(k):
                    P += con_fac[ii]*(lb[ii] - xval[ii])*(ub[ii] - xval[ii])

            f = conNegEI + P

        #print(xI, f)
        return f

    def maximize_S(self, x_comL, x_comU, Ain_hat, bin_hat, surrogate):
        """This method finds an upper bound to the SigmaSqr Error, and scales
        up 'r' to provide a smooth design space for gradient-based approach.
        """
        R_inv = surrogate.R_inv
        SigmaSqr = surrogate.SigmaSqr
        X = surrogate.X

        n, k = X.shape
        one = np.ones([n, 1])

        xhat_comL = x_comL.copy()
        xhat_comU = x_comU.copy()
        xhat_comL[k:] = 0.0
        xhat_comU[k:] = 1.0

        # Calculate the convexity factor alpha
        rL = x_comL[k:]
        rU = x_comU[k:]

        dr_drhat = np.zeros([n, n])
        for ii in range(n):
            dr_drhat[ii, ii] = rU[ii, 0] - rL[ii, 0]

        T2_num = np.dot(np.dot(R_inv, one),np.dot(R_inv, one).T)
        T2_den = np.dot(one.T, np.dot(R_inv, one))
        d2S_dr2 = 2.0*SigmaSqr*(R_inv - (T2_num/T2_den))
        H_hat = np.dot(np.dot(dr_drhat, d2S_dr2), dr_drhat.T)

        # Use Gershgorin's circle theorem to find a lower bound of the
        # min eigen value of the hessian
        eig_lb = np.zeros([n, 1])
        for ii in range(n):
            dia_ele = H_hat[ii, ii]
            sum_rw = 0.0
            sum_col = 0.0
            for jj in range(n):
                if ii != jj:
                    sum_rw += np.abs(H_hat[ii,jj])
                    sum_col += np.abs(H_hat[jj,ii])

                eig_lb[ii] = dia_ele - np.min(np.array([sum_rw, sum_col]))

        eig_min = np.min(eig_lb)
        alpha = np.max(np.array([0.0, -0.5*eig_min]))

        # Just storing it here to pull it out in the callback?
        surrogate._alpha = alpha

        # Maximize S
        x0 = 0.5*(xhat_comL + xhat_comU)
        #bnds = [(xhat_comL[ii], xhat_comU[ii]) for ii in range(len(xhat_comL))]

        ##Note: Python defines constraints like g(x) >= 0
        #cons = [{'type' : 'ineq',
                 #'fun' : lambda x : -np.dot(Ain_hat[ii, :], x) + bin_hat[ii],
                 #'jac' : lambda x : -Ain_hat[ii, :]} for ii in range(2*n)]

        if self.pyopt:

            self.x_comL = x_comL
            self.x_comU = x_comU
            self.xhat_comL = xhat_comL
            self.xhat_comU = xhat_comU
            self.Ain_hat = Ain_hat
            self.bin_hat = bin_hat
            self.current_surr = surrogate

            opt_x, opt_f, succ_flag = snopt_opt(self.calc_SSqr_convex, x0, xhat_comL,
                                                xhat_comU, len(bin_hat),
                                                title='Maximize_S',
                                                options={'Major optimality tolerance' : self.options['ftol']},
                                                jac=Ain_hat,
                                                ) #sens=self.calc_SSqr_convex_grad)

            Neg_sU = opt_f
            # if not succ_flag:
            #     eflag_sU = False
            # else:
            #     eflag_sU = True
            eflag_sU = True
            tol = self.options['con_tol']
            for ii in range(2*n):
                if np.dot(Ain_hat[ii, :], opt_x) > (bin_hat[ii ,0] + tol):
                    eflag_sU = False
                    break

        else:
            optResult = minimize(self.calc_SSqr_convex_old, x0,
                                 args=(x_comL, x_comU, xhat_comL, xhat_comU, surrogate),
                                 method='SLSQP', constraints=cons, bounds=bnds,
                                 options={'ftol' : self.options['ftol'],
                                          'maxiter' : 100})

            Neg_sU = optResult.fun
            if not optResult.success:
                eflag_sU = False
            else:
                eflag_sU = True
                tol = self.options['con_tol']
                for ii in range(2*n):
                    if np.dot(Ain_hat[ii, :], optResult.x) > (bin_hat[ii ,0] + tol):
                        eflag_sU = False
                        break

        sU = - Neg_sU
        return sU, eflag_sU

    def calc_SSqr_convex(self, dv_dict):
        """ Callback function for minimization of mean squared error."""
        fail = 0

        x_com = dv_dict['x']
        surrogate = self.current_surr
        x_comL = self.x_comL
        x_comU = self.x_comU
        xhat_comL = self.xhat_comL
        xhat_comU = self.xhat_comU

        X = surrogate.X
        R_inv = surrogate.R_inv
        SigmaSqr = surrogate.SigmaSqr
        alpha = surrogate._alpha

        n, k = X.shape

        one = np.ones([n, 1])

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = x_com[k:].reshape(n, 1)

        r = rL + rhat*(rU - rL)
        rhat_L = xhat_comL[k:]
        rhat_U = xhat_comU[k:]

        term0 = np.dot(R_inv, r)
        term1 = -SigmaSqr*(1.0 - r.T.dot(term0) + \
        ((1.0 - one.T.dot(term0))**2/(one.T.dot(np.dot(R_inv, one)))))

        term2 = alpha*(rhat-rhat_L).T.dot(rhat-rhat_U)
        S2 = term1 + term2

        # Objectives
        func_dict = {}
        func_dict['obj'] = S2[0, 0]

        # Constraints
        Ain_hat = self.Ain_hat
        bin_hat = self.bin_hat

        func_dict['con'] = np.dot(Ain_hat, x_com) - bin_hat.flatten()
        #print('x', dv_dict)
        #print('obj', func_dict['obj'])
        return func_dict, fail

    def calc_SSqr_convex_grad(self, dv_dict, func_dict):
        """ Callback function for gradient of mean squared error."""
        fail = 0

        x_com = dv_dict['x']
        surrogate = self.current_surr
        x_comL = self.x_comL
        x_comU = self.x_comU
        xhat_comL = self.xhat_comL
        xhat_comU = self.xhat_comU

        X = surrogate.X
        R_inv = surrogate.R_inv
        SigmaSqr = surrogate.SigmaSqr
        alpha = surrogate._alpha

        n, k = X.shape
        nn = len(x_com)

        one = np.ones([n, 1])

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = x_com[k:].reshape(n, 1)

        r = rL + rhat*(rU - rL)
        rhat_L = xhat_comL[k:]
        rhat_U = xhat_comU[k:]

        dr_drhat = np.diag((rU-rL).flat)

        term0 = np.dot(R_inv, r) #This should be nx1 vector
        term1 = ((1.0 - one.T.dot(term0))/(one.T.dot(np.dot(R_inv, one))))*np.dot(R_inv, one) #This should be nx1 vector
        term = 2.0*SigmaSqr*(term0 + term1) #This should be nx1 vector

        #zdterm1a = (r.T.dot(R_inv) + r.T.dot(R_inv.T))
        #zdterm1b = 2.0*((1.0 - one.T.dot(term0))*np.sum(R_inv, 0)/(one.T.dot(np.dot(R_inv, one))))
        #zdterm1 = SigmaSqr*(zdterm1a.T + dr_drhat.dot(zdterm1b.T))

        dterm1 = np.dot(dr_drhat, term) #This should be nx1 vector
        dterm2 = alpha*(2.0*rhat - rhat_L - rhat_U) #This should be nx1 vector

        dobj_dr = (dterm1 + dterm2).T #This should be 1xn vector

        # Objectives
        sens_dict = OrderedDict()
        sens_dict['obj'] = OrderedDict()
        sens_dict['obj']['x'] = np.zeros((1, nn))
        sens_dict['obj']['x'][:, k:] = dobj_dr

        # Constraints
        Ain_hat = self.Ain_hat
        bin_hat = self.bin_hat

        sens_dict['con'] = OrderedDict()
        sens_dict['con']['x'] = Ain_hat

        #print('obj deriv', sens_dict['obj']['x'] )
        #print('con deriv', sens_dict['con']['x'])
        return sens_dict, fail

    def minimize_y(self, x_comL, x_comU, Ain_hat, bin_hat, surrogate):

        # 1- Formulates y_hat as LP (weaker bound)
        # 2- Uses non-convex relaxation technique (stronger bound) [Future release]
        app = 1

        X = surrogate.X
        n, k = X.shape

        xhat_comL = x_comL.copy()
        xhat_comU = x_comU.copy()
        xhat_comL[k:] = 0.0
        xhat_comU[k:] = 1.0

        if app == 1:
            x0 = 0.5*(xhat_comL + xhat_comU)
            #bnds = [(xhat_comL[ii], xhat_comU[ii]) for ii in range(len(xhat_comL))]

            #cons = [{'type' : 'ineq',
                     #'fun' : lambda x : -np.dot(Ain_hat[ii, :],x) + bin_hat[ii],
                     #'jac': lambda x: -Ain_hat[ii, :]} for ii in range(2*n)]

        if self.pyopt:

            self.x_comL = x_comL
            self.x_comU = x_comU
            self.Ain_hat = Ain_hat
            self.bin_hat = bin_hat
            self.current_surr = surrogate

            opt_x, opt_f, succ_flag = snopt_opt(self.calc_y_hat_convex, x0, xhat_comL,
                                                xhat_comU, len(bin_hat),
                                                title='minimize_y',
                                                options={'Major optimality tolerance' : self.options['ftol']},
                                                jac=Ain_hat)

            yL = opt_f
            # if not succ_flag:
            #     eflag_yL = False
            # else:
            #     eflag_yL = True
            eflag_yL = True
            tol = self.options['con_tol']
            for ii in range(2*n):
                if np.dot(Ain_hat[ii, :], opt_x) > (bin_hat[ii, 0] + tol):
                    eflag_yL = False
                    break

        else:
            optResult = minimize(self.calc_y_hat_convex_old, x0,
                                 args=(x_comL, x_comU, surrogate), method='SLSQP',
                                 constraints=cons, bounds=bnds,
                                 options={'ftol' : self.options['ftol'],
                                          'maxiter' : 100})

            yL = optResult.fun
            if not optResult.success:
                eflag_yL = False
            else:
                eflag_yL = True
                tol = self.options['con_tol']
                for ii in range(2*n):
                    if np.dot(Ain_hat[ii, :], optResult.x) > (bin_hat[ii, 0] + tol):
                        eflag_yL = False
                        break

        return yL, eflag_yL

    def calc_y_hat_convex_old(self, x_com, *param):
        x_comL = param[0]
        x_comU = param[1]
        surrogate = param[2]

        X = surrogate.X
        c_r = surrogate.c_r
        mu = surrogate.mu
        n, k = X.shape

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = np.array([x_com[k:]]).reshape(n, 1)
        r = rL + rhat*(rU - rL)

        y_hat = mu + np.dot(r.T, c_r)
        return y_hat[0, 0]

    def calc_y_hat_convex(self, dv_dict):
        fail = 0

        x_com = dv_dict['x']
        surrogate = self.current_surr
        x_comL = self.x_comL
        x_comU = self.x_comU

        X = surrogate.X
        c_r = surrogate.c_r
        mu = surrogate.mu
        n, k = X.shape

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = np.array([x_com[k:]]).reshape(n, 1)
        r = rL + rhat*(rU - rL)

        y_hat = mu + np.dot(r.T, c_r)

        # Objective
        func_dict = {}
        func_dict['obj'] = y_hat[0, 0]

        # Constraints
        Ain_hat = self.Ain_hat
        bin_hat = self.bin_hat

        func_dict['con'] = np.dot(Ain_hat, x_com) - bin_hat.flatten()
        #print('x', dv_dict)
        #print('obj', func_dict['obj'])
        return func_dict, fail

def update_active_set(active_set, ubd):
    """ Remove variables from the active set data structure if their current
    upper bound exceeds the given value.

    Args
    ----
    active_set : list of lists of floats
        Active set data structure of form [[NodeNumber, lb, ub, LBD, UBD], [], ..]
    ubd : float
        Maximum for bounds test.

    Returns
    -------
    new active_set
    """
    return [a for a in active_set if a[3] < ubd]


def gen_coeff_bound(xI_lb, xI_ub, surrogate):
    """This function generates the upper and lower bound of the artificial
    variable r and the coefficients for the linearized under estimator
    constraints. The version accepts design bound in the original design
    space, converts it to normalized design space.
    """

    #Normalized to 0-1 hypercube
    # xL_hat0 = (xI_lb - surrogate.lb_org.flatten())/(surrogate.ub_org.flatten() - surrogate.lb_org.flatten())
    # xU_hat0 = (xI_ub - surrogate.lb_org.flatten())/(surrogate.ub_org.flatten() - surrogate.lb_org.flatten())
    # xL_hat = xL_hat0
    # xU_hat = xU_hat0

    #Normalized as per Openmdao kriging model
    xL_hat = (xI_lb - surrogate.X_mean.flatten())/surrogate.X_std.flatten()
    xU_hat = (xI_ub - surrogate.X_mean.flatten())/surrogate.X_std.flatten()

    rL, rU = interval_analysis(xL_hat, xU_hat, surrogate)

    # Combined design variables for supbproblem
    num = len(xL_hat) + len(rL)
    x_comL = np.append(xL_hat, rL).reshape(num, 1)
    x_comU = np.append(xU_hat, rU).reshape(num, 1)

    # Coefficients of the linearized constraints of the subproblem
    Ain_hat, bin_hat = lin_underestimator(x_comL, x_comU, surrogate)

    return x_comL, x_comU, Ain_hat, bin_hat


def interval_analysis(lb_x, ub_x, surrogate):
    """ The module predicts the lower and upper bound of the artificial
    variable 'r' from the bounds of the design variable x r is related to x
    by the following equation:

    r_i = exp(-sum(theta_h*(x_h - x_h_i)^2))

    """

    X = surrogate.X
    thetas = surrogate.thetas
    p = surrogate.p
    n, k = X.shape

    t1L = np.zeros([n, k]); t1U = np.zeros([n, k])
    t2L = np.zeros([n, k]); t2U = np.zeros([n, k])
    t3L = np.zeros([n, k]); t3U = np.zeros([n, k])
    t4L = np.zeros([n, 1]); t4U = np.zeros([n, 1])
    lb_r = np.zeros([n, 1]); ub_r = np.ones([n, 1])

    if p % 2 == 0:
        for i in range(n):
            for h in range(k):
                t1L[i,h] = lb_x[h] - X[i, h]
                t1U[i,h] = ub_x[h] - X[i, h]
    #
                t2L[i,h] = np.max(np.array([0,np.min(np.array([t1L[i, h]*t1L[i, h],
                                                                t1L[i, h]*t1U[i, h],
                                                                t1U[i, h]*t1U[i, h]]))]))
                t2U[i,h] = np.max(np.array([0,np.max(np.array([t1L[i, h]*t1L[i, h],
                                                                t1L[i, h]*t1U[i, h],
                                                                t1U[i, h]*t1U[i, h]]))]))
    #
                t3L[i,h] = np.min(np.array([-thetas[h]*t2L[i, h], -thetas[h]*t2U[i, h]]))
                t3U[i,h] = np.max(np.array([-thetas[h]*t2L[i, h], -thetas[h]*t2U[i, h]]))
    #
            t4L[i] = np.sum(t3L[i, :])
            t4U[i] = np.sum(t3U[i, :])
    #
            lb_r[i] = np.exp(t4L[i])
            ub_r[i] = np.exp(t4U[i])
    else:
        print("\nWarning! Value of p should be 2. Cannot perform interval analysis")
        print("\nReturing global bound of the r variable")

    return lb_r, ub_r


def lin_underestimator(lb, ub, surrogate):
    X = surrogate.X
    thetas = surrogate.thetas
    p = surrogate.p
    n, k = X.shape

    lb_x = lb[:k]; ub_x = ub[:k]
    lb_r = lb[k:]; ub_r = ub[k:]

    a1 = np.zeros([n, n]); a3 = np.zeros([n, n])
    a1_hat = np.zeros([n, n]); a3_hat = np.zeros([n, n])
    a2 = np.zeros([n, k]); a4 = np.zeros([n, k])
    b2 = np.zeros([n, k]); b4 = np.zeros([n, k])
    b1 = np.zeros([n, 1]); b3 = np.zeros([n, 1])
    b1_hat = np.zeros([n, 1]); b3_hat = np.zeros([n, 1])

    for i in range(n):
        #T1: Linearize under-estimator of ln[r_i] = a1[i,i]*r[i] + b1[i]
        if ub_r[i] <= lb_r[i]:
            a1[i,i] = 0.0
        else:
            a1[i,i] = ((np.log(ub_r[i]) - np.log(lb_r[i]))/(ub_r[i] - lb_r[i]))

        b1[i] = np.log(ub_r[i]) - a1[i,i]*ub_r[i]
        a1_hat[i,i] = a1[i,i]*(ub_r[i]-lb_r[i])
        b1_hat[i] = a1[i,i]*lb_r[i] + b1[i]

        #T3: Linearize under-estimator of -ln[r_i] = a3[i,i]*r[i] + b3[i]
        r_m_i = (lb_r[i] + ub_r[i])/2.0
        a3[i,i] = -1.0/r_m_i
        b3[i] = -np.log(r_m_i) - a3[i,i]*r_m_i
        a3_hat[i,i] = a3[i,i]*(ub_r[i] - lb_r[i])
        b3_hat[i] = a3[i,i]*lb_r[i] + b3[i]

        for h in range(k):
            #T2: Linearize under-estimator of thetas_h*(x_h - X_h_i)^2 = a4[i,h]*x_h[h] + b4[i,h]
            x_m_h = (ub_x[h] + lb_x[h])/2.0
            a2[i,h] = p*thetas[h]*(x_m_h - X[i,h])**(p-1.0)
            yy = thetas[h]*(x_m_h - X[i,h])**p
            b2[i,h] = -a2[i,h]*x_m_h + yy

            #T4: Linearize under-estimator of -theta_h*(x_h - X_h_i)^2 = a4[i,h]*x_h[h] + b4[i,h]
            yy2 = -thetas[h]*(ub_x[h] - X[i,h])**p
            yy1 = -thetas[h]*(lb_x[h] - X[i,h])**p

            if ub_x[h] <= lb_x[h]:
                a4[i,h] = 0.0
            else:
                a4[i,h] = (yy2 - yy1)/(ub_x[h] - lb_x[h])

            b4[i,h] = -a4[i,h]*lb_x[h] + yy1

    Ain1 = np.concatenate((a2, a4), axis=0)
    Ain2 = np.concatenate((a1_hat, a3_hat), axis=0)
    Ain_hat = np.concatenate((Ain1, Ain2), axis=1)
    bin_hat = np.concatenate((-(b1_hat + np.sum(b2, axis=1).reshape(n,1)),
                              -(b3_hat + np.sum(b4, axis=1).reshape(n,1))), axis=0)

    return Ain_hat, bin_hat

def calc_conEI_norm(xval, obj_surrogate, SSqr=None, y_hat=None):
    """This function evaluates the expected improvement in the normalized
    design space.
    """
    # y_min = (obj_surrogate.y_best - obj_surrogate.Y_mean)/obj_surrogate.Y_std
    y_min = obj_surrogate.y_best #Ensure y_min is the minimum of the y used to train the surrogates (i.e centered/scaled/normalized y)

    if SSqr is None:
        X = obj_surrogate.X
        c_r = obj_surrogate.c_r
        thetas = obj_surrogate.thetas
        SigmaSqr = obj_surrogate.SigmaSqr
        R_inv = obj_surrogate.R_inv
        mu = obj_surrogate.mu
        p = obj_surrogate.p

        n = np.shape(X)[0]
        one = np.ones([n, 1])

        r = np.exp(-np.sum(thetas.T*(xval - X)**p, 1)).reshape(n, 1)

        y_hat = mu + np.dot(r.T, c_r)
        term0 = np.dot(R_inv, r)
        SSqr = SigmaSqr*(1.0 - r.T.dot(term0) + \
        ((1.0 - one.T.dot(term0))**2)/(one.T.dot(np.dot(R_inv, one))))

    if abs(SSqr) == 0.0:
        NegEI = 0.0
    else:
        dy = y_min - y_hat
        SSqr = abs(SSqr)
        ei1 = dy*(0.5+0.5*erf((1/np.sqrt(2))*(dy/np.sqrt(SSqr))))
        ei2 = np.sqrt(SSqr)*(1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*(dy**2/SSqr))
        NegEI = -(ei1 + ei2)
    return NegEI


def calc_conEV_norm(xval, con_surrogate, gSSqr=None, g_hat=None):
    """This modules evaluates the expected improvement in the normalized
    design sapce"""

    g_min = (1.0e-6 - con_surrogate.Y_mean)/con_surrogate.Y_std

    if gSSqr is None:
        X = con_surrogate.X
        c_r = con_surrogate.c_r
        thetas = con_surrogate.thetas
        SigmaSqr = con_surrogate.SigmaSqr
        R_inv = con_surrogate.R_inv
        mu = con_surrogate.mu
        p = con_surrogate.p
        n = np.shape(X)[0]
        one = np.ones([n, 1])

        r = np.exp(-np.sum(thetas.T*(xval - X)**p, 1)).reshape(n, 1)

        g_hat = mu + np.dot(r.T, c_r)
        term0 = np.dot(R_inv, r)
        gSSqr = SigmaSqr*(1.0 - r.T.dot(term0) + \
                          ((1.0 - one.T.dot(term0))**2)/(one.T.dot(np.dot(R_inv, one))))

    if abs(gSSqr) == 0.0:
        EV = 0.0
    else:
        # Calculate expected violation
        dg = g_hat - g_min
        gSSqr = abs(gSSqr)
        ei1 = dg*(0.5 + 0.5*erf((1.0/np.sqrt(2.0))*(dg/np.sqrt(gSSqr))))
        ei2 = np.sqrt(gSSqr)*(1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*(dg**2/gSSqr))
        EV = (ei1 + ei2)

    return EV

def init_nodes(N, xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt, xopt):
    pts = (xU_iter-xL_iter) + 1.0
    com_enum = np.prod(pts, axis=0)
    tot_pts = 0.0
    num_cut = min(N-1,com_enum-1)
    if num_cut>0:
        new_nodes = [[xL_iter, xU_iter, com_enum]]
        for cut in range(num_cut):
            all_area = [item[2] for item in new_nodes]
            maxA = max(all_area)
            ind_maxA = all_area.index(maxA)
            xL_iter, xU_iter, _ = new_nodes[ind_maxA]
            del new_nodes[ind_maxA]

            #Branching scheme stays same
            xloc_iter = np.round(xL_iter + 0.49*(xU_iter - xL_iter))
            # Choose the largest edge
            l_iter = (xU_iter - xL_iter).argmax()
            if xloc_iter[l_iter]<xU_iter[l_iter]:
                delta = 0.5 #0<delta<1
            else:
                delta = -0.5 #-1<delta<0
            for ii in range(2):
                lb = xL_iter.copy()
                ub = xU_iter.copy()
                if ii == 0:
                    ub[l_iter] = np.floor(xloc_iter[l_iter]+delta)
                elif ii == 1:
                    lb[l_iter] = np.ceil(xloc_iter[l_iter]+delta)
                pts = (ub-lb) + 1.0
                enum = np.prod(pts, axis=0)
                new_node = [lb, ub, enum]
                new_nodes.append(new_node)

        args = []
        n_nodes = len(new_nodes)
        for ii in range(n_nodes):
            xL_iter, xU_iter, enum = new_nodes[ii]
            tot_pts += enum
            args.append((xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                         xopt, ii+1))
    else:
        args = [(xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                xopt, 0)]

    return args
