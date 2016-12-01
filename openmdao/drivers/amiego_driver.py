""" Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).

This driver is based on the EGO-Like Framework (EGOLF) for the simultaneous
design-mission-allocation optimization problem. Handles
mixed-integer/discrete type design variables in a computationally efficient
manner and finds a near-global solution to the above MINLP/MDNLP problem.

Developed by Satadru Roy
Purdue University, West Lafayette, IN
July 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""

from __future__ import print_function
from copy import deepcopy
from time import time

from six import iteritems
from six.moves import range

import numpy as np

from openmdao.core.driver import Driver
from openmdao.core.vec_wrapper import _ByObjWrapper
from openmdao.drivers.branch_and_bound import Branch_and_Bound
from openmdao.drivers.scipy_optimizer import ScipyOptimizer
from openmdao.surrogate_models.kriging import KrigingSurrogate
from openmdao.util.record_util import create_local_meta, update_local_meta


class AMIEGO_driver(Driver):
    """ Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).
    This driver is based on the EGO-Like Framework (EGOLF) for the
    simultaneous design-mission-allocation optimization problem. It handles
    mixed-integer/discrete type design variables in a computationally
    efficient manner and finds a near-global solution to the above
    MINLP/MDNLP problem. The continuous optimization is handled by the
    optimizer slotted in self.cont_opt.

    AMIEGO_driver supports the following:
        integer_design_vars

    Options
    -------
    options['ei_tol_rel'] :  0.001
        Relative tolerance on the expected improvement.
    options['ei_tol_abs'] :  0.001
        Absolute tolerance on the expected improvement.
    options['max_infill_points'] : 10
        Ratio of maximum number of additional points to number of initial
        points.

    """

    def __init__(self):
        """Initialize the AMIEGO driver."""

        super(AMIEGO_driver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = True
        self.supports['mixed_integer'] = True

        # TODO - I started working on this, but needs tests and a bug fix
        self.supports['two_sided_constraints'] = False

        # Options
        opt = self.options
        opt.add_option('disp', True,
                       desc='Set to False to prevent printing of iteration '
                       'messages.')
        opt.add_option('ei_tol_rel', 0.001, lower=0.0,
                       desc='Relative tolerance on the expected improvement.')
        opt.add_option('ei_tol_abs', 0.001, lower=0.0,
                       desc='Absolute tolerance on the expected improvement.')
        opt.add_option('max_infill_points', 10, lower=1.0,
                       desc='Ratio of maximum number of additional points to number of initial points.')

        # The default continuous optimizer. User can slot a different one
        self.cont_opt = ScipyOptimizer()
        self.cont_opt.options['optimizer'] = 'SLSQP' #TODO Switch tp SNOPT

        # The default MINLP optimizer
        self.minlp = Branch_and_Bound()

        # Default surrogate. User can slot a modified one, but it essentially
        # has to provide what Kriging provides.
        self.surrogate = KrigingSurrogate

        self.c_dvs = []
        self.i_dvs = []
        self.i_size = 0
        self.i_idx = {}
        self.record_name = 'AMIEGO'

        # Initial Sampling
        # TODO: Somehow slot an object that generates this (LHC for example)
        self.sampling = {}

        # User can pre-load these to skip initial continuous optimization
        # in favor of pre-optimized points.
        # NOTE: when running in this mode, everything comes in as lists.
        self.obj_sampling = None
        self.con_sampling = None

    def _setup(self):
        """  Initialize whatever we need."""
        super(AMIEGO_driver, self)._setup()
        cont_opt = self.cont_opt
        cont_opt._setup()
        cont_opt.record_name = self.record_name + ':' + cont_opt.record_name
        cont_opt.dv_conversions = self.dv_conversions
        cont_opt.fn_conversions = self.fn_conversions

        if 'disp' in cont_opt.options:
            cont_opt.options['disp'] = self.options['disp']

        minlp = self.minlp
        minlp._setup()
        minlp.record_name = self.record_name + ':' +  minlp.record_name
        minlp.standalone = False
        minlp.options['disp'] = self.options['disp']

        # Identify and size our design variables.
        j = 0
        for name, val in iteritems(self.get_desvars()):
            if isinstance(val, _ByObjWrapper):
                self.i_dvs.append(name)
                try:
                    i_size = len(np.asarray(val.val))
                except TypeError:
                    i_size = 1
                self.i_idx[name] = (j, j+i_size)
                j += i_size
            else:
                self.c_dvs.append(name)
        self.i_size = j

        # Lower and Upper bounds for integer desvars
        self.xI_lb = np.empty((self.i_size, ))
        self.xI_ub = np.empty((self.i_size, ))
        dv_dict = self._desvars
        for var in self.i_dvs:
            i, j = self.i_idx[var]
            self.xI_lb[i:j] = dv_dict[var]['lower']
            self.xI_ub[i:j] = dv_dict[var]['upper']

        # Continuous Optimization only gets continuous desvars
        for name in self.c_dvs:
            cont_opt._desvars[name] = self._desvars[name]

        # MINLP Optimization only gets discrete desvars
        for name in self.i_dvs:
            minlp._desvars[name] = self._desvars[name]

        # It should be perfectly okay to 'share' obj and con with the
        # sub-optimizers.
        cont_opt._cons = self._cons
        cont_opt._objs = self._objs
        minlp._cons = self._cons
        minlp._objs = self._objs

    def set_root(self, pathname, root):
        """ Sets the root Group of this driver.

        Args
        ----
        root : Group
            Our root Group.
        """
        super(AMIEGO_driver, self).set_root(pathname, root)
        self.cont_opt.set_root(pathname, root)
        self.minlp.set_root(pathname, root)

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the
            min and max processors usable by this `Driver`.
        """
        return self.minlp.get_req_procs()

    def run(self, problem):
        """Execute the AMIEGO driver.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        n_i = self.i_size
        ei_tol_rel = self.options['ei_tol_rel']
        ei_tol_abs = self.options['ei_tol_abs']
        disp = self.options['disp']
        cont_opt = self.cont_opt
        minlp = self.minlp
        xI_lb = self.xI_lb
        xI_ub = self.xI_ub

        # Metadata Setup
        self.metadata = create_local_meta(None, self.record_name)
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count, ))

        #----------------------------------------------------------------------
        # Step 1: Generate a set of initial integer points
        # TODO: Use Latin Hypercube Sampling to generate the initial points
        # User supplied (in future use LHS). Provide num_xI+2 starting points
        #----------------------------------------------------------------------

        max_pt_lim = self.options['max_infill_points']*n_i

        # Since we need to add a new point every iteration, make these lists
        # for speed.
        x_i = []
        x_i_hat = []
        obj = []
        cons = {}
        best_int_design = {}
        best_cont_design = {}
        for con in self.get_constraint_metadata():
            cons[con] = []

        # Start with pre-optimized samples
        if self.obj_sampling:
            pre_opt = True
            n_train = len(self.sampling[self.i_dvs[0]])
            c_start = c_end = n_train

            for i_train in range(n_train):

                xx_i = np.empty((self.i_size, ))
                for var in self.i_dvs:
                    i, j = self.i_idx[var]
                    xx_i[i:j] = self.sampling[var][i_train]

                    # Save the best design too (see below)
                    if i_train == 0:
                        best_int_design[var] = self.sampling[var][i_train]

                x_i.append(xx_i)

            current_objs = self.get_objectives()
            obj_name = list(current_objs.keys())[0]
            obj = self.obj_sampling[obj_name]
            cons = self.con_sampling

            # Satadru's suggestion is that we start with the first point as
            # the best obj.
            best_obj = obj[0].copy()

        # Prepare to optimize the initial sampling points
        else:
            best_obj = 1.0e99
            pre_opt = False
            n_train = self.sampling[self.i_dvs[0]].shape[0]
            c_start = 0
            c_end = n_train

            for i_train in range(n_train):

                xx_i = np.empty((self.i_size, ))
                # xx_i_hat = np.empty((self.i_size, ))
                for var in self.i_dvs:
                    lower = self._desvars[var]['lower']
                    upper = self._desvars[var]['upper']
                    i, j = self.i_idx[var]

                    #Samples should be bounded in an unit hypercube [0,1]
                    x_i_0 = self.sampling[var][i_train, :]
                    del_fac = 0.0 #TODO This is the deafult. Set is to 2.0/3.0 for the des-alloc problem.
                    xx_i[i:j] = np.round(lower + x_i_0 * (upper - lower-del_fac))
                    # xx_i_hat[i:j] = (xx_i[i:j] - lower)/(upper - lower)
                x_i.append(xx_i)
                # x_i_hat.append(xx_i_hat)

        # Need to cache the continuous desvars so that we start each new
        # optimization back at the original initial condition.
        xc_cache = {}
        desvars = cont_opt.get_desvars()
        for var, val in iteritems(desvars):
            xc_cache[var] = val.copy()

        ei_max = 1.0
        term = 0.0
        terminate = False
        tot_newpt_added = 0
        tot_pt_prev = 0
        ec2 = 0

        # AMIEGO main loop
        while not terminate:
            self.iter_count += 1

            #------------------------------------------------------------------
            # Step 2: Perform the optimization w.r.t continuous design
            # variables
            #------------------------------------------------------------------

            if disp:
                print("======================ContinuousOptimization-Start=====================================")
                t0 = time()

            for i_run in range(c_start, c_end):
                if disp:
                    print("Optimizing for the given integer/discrete type design variables:", x_i[i_run])
                # Set Integer design variables
                for var in self.i_dvs:
                    i, j = self.i_idx[var]
                    self.set_desvar(var, x_i[i_run][i:j])

                # Restore initial condition for continuous vars.
                for var, val in iteritems(xc_cache):
                    cont_opt.set_desvar(var, val)

                # Optimize continuous variables
                cont_opt.run(problem)
                eflag_conopt = cont_opt.success
                if disp:
                    print("Exit flag: ",eflag_conopt)
                if not eflag_conopt:
                    self.minlp.bad_samples.append(x_i[i_run])

                if not eflag_conopt:
                    self.minlp.bad_samples.append(x_i[i_run])

                # Get objectives and constraints (TODO)
                current_objs = self.get_objectives()
                obj_name = list(current_objs.keys())[0]
                current_obj = current_objs[obj_name].copy()
                obj.append(current_obj)
                for name, value in iteritems(self.get_constraints()):
                    cons[name].append(value.copy())
                # If best solution, save it
                if eflag_conopt and current_obj < best_obj:
                    best_obj = current_obj
                    # Save integer and continuous DV
                    desvars = self.get_desvars()

                    for name in self.i_dvs:
                        val = desvars[name]
                        if isinstance(val, _ByObjWrapper):
                            val = val.val
                        best_int_design[name] = val.copy()

                    for name in self.c_dvs:
                        best_cont_design[name] = desvars[name].copy()

            if disp:
                print('Elapsed Time:', time() - t0)
                print("======================ContinuousOptimization-End=======================================")
            # exit()
            #------------------------------------------------------------------
            # Step 3: Build the surrogate models
            #------------------------------------------------------------------
            obj_surrogate = self.surrogate()
            obj_surrogate.use_snopt = True
            print(x_i)
            obj_surrogate.train(x_i, obj, KPLS_status=True)

            obj_surrogate.y = obj
            obj_surrogate.lb_org = xI_lb
            obj_surrogate.ub_org = xI_ub
            obj_surrogate.lb = np.zeros((n_i))
            obj_surrogate.ub = np.zeros((n_i))
            best_obj_norm = (best_obj - obj_surrogate.Y_mean)/obj_surrogate.Y_std

            con_surrogate = []
            for name, val in iteritems(cons):
                val = np.array(val)

                # Note, Branch and Bound defines constraints to be violated
                # when positive, so we need to transform from OpenMDAO's
                # freeform.
                meta = self._cons[name]
                upper = meta['upper']
                lower = meta['lower']
                double_sided = False
                if lower is None:
                    val = val - upper
                elif upper is None:
                    val = lower - val
                else:
                    double_sided = True
                    val_u = val - upper
                    val_l = lower - val

                for j in range(val.shape[1]):
                    con_surr = self.surrogate()
                    con_surr.use_snopt = True

                    if double_sided:
                        val_idx = max(val_u[:, j:j+1], val_l[:, j:j+1])
                    else:
                        val_idx = val[:, j:j+1]

                    con_surr.train(x_i, val_idx, True)
                    con_surr.y = val_idx
                    con_surr._name = name
                    con_surr.lb_org = xI_lb
                    con_surr.ub_org = xI_ub
                    con_surr.lb = np.zeros((n_i))
                    con_surr.ub = np.zeros((n_i))
                    con_surrogate.append(con_surr)

            if disp:
                print("\nSurrogate building of the objective is complete...")

            #------------------------------------------------------------------
            # Step 4: Maximize the expected improvement function to obtain an
            # integer infill point.
            #------------------------------------------------------------------

            if disp:
                print("EGOLF-Iter: %d" % self.iter_count)
                print("The best solution so far: yopt = %0.4f" % best_obj)

            tot_newpt_added += c_end - c_start
            if pre_opt or tot_newpt_added != tot_pt_prev:

                minlp.obj_surrogate = obj_surrogate
                minlp.con_surrogate = con_surrogate
                minlp.xI_lb = xI_lb
                minlp.xI_ub = xI_ub

                if disp:
                    t0 = time()
                    print("======================MINLPBB-Start=====================================")
                minlp.run(problem)
                if disp:
                    print('Elapsed Time:', time() - t0)
                    print("======================MINLPBB-End=======================================")

                eflag_MINLPBB = minlp.eflag_MINLPBB
                x0I = minlp.xopt
                ei_min = minlp.fopt

                if disp:
                    print("Eflag = ", eflag_MINLPBB)

                if eflag_MINLPBB >= 1:

                    # x0I_hat = (x0I - xI_lb)/(xI_ub - xI_lb)

                    ei_max = -ei_min
                    tot_pt_prev = tot_newpt_added

                    if disp:
                        print("New xI = ", x0I)
                        print("EI_min = ", ei_min)

                    # Prevent the correlation matrix being close to singular. No
                    # point allowed within the pescribed hypersphere of any
                    # existing point
                    rad = 0.5
                    for ii in range(len(x_i)):
                        dist = np.sum((x_i[ii] - x0I)**2)**0.5
                        if dist <= rad:
                            if disp:
                                print("Point already exists!")
                            ec2 = 1
                            break
                    x_i.append(x0I)
                    # x_i_hat.append(x0I_hat)

                else:
                    ec2 = 1
            else:
                ec2 = 1

            #------------------------------------------------------------------
            # Step 5: Check for termination
            #------------------------------------------------------------------

            c_start = c_end
            c_end += 1

            # # 1e-6 is the switchover from rel to abs.
            # if np.abs(best_obj)<= 1.0e-6:
            #     term = ei_tol_abs
            # else:
            #     term = np.max(np.array([np.abs(ei_tol_rel*best_obj), ei_tol_abs]))
            term  = np.abs(ei_tol_rel*best_obj_norm)
            if (not pre_opt and ei_max <= term) or ec2 == 1 or tot_newpt_added >= max_pt_lim:
                terminate = True
                if disp:
                    if ei_max <= term:
                        print("No Further improvement expected! Terminating algorithm.")
                    elif ec2 == 1:
                        print("No new point found that improves the surrogate. Terminating algorithm.")
                    elif tot_newpt_added >= max_pt_lim:
                        print("Maximum allowed sampling limit reached! Terminating algorithm.")

            pre_opt = False

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name, val in iteritems(best_int_design):
            if isinstance(val, _ByObjWrapper):
                self.set_desvar(name, val.val)
            else:
                self.set_desvar(name, val)
        for name, val in iteritems(best_cont_design):
            self.set_desvar(name, val)

        with self.root._dircontext:
            self.root.solve_nonlinear(metadata=self.metadata)

        if disp:
            print("\n===================Result Summary====================")
            print("The best objective: %0.4f" % best_obj)
            print("Total number of continuous minimization: %d" % len(x_i))
            #print("Total number of objective function evaluation: %d" % Tot_FunCount)
            print("Best Integer designs: ", best_int_design)
            print("Corresponding continuous designs: ", best_cont_design)
            print("=====================================================")
