""" Broyden non-linear solver implemented using source code from
scipy.optimize's broyden root solvers."""

import numpy as np
from numpy.linalg import norm

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.util.record_util import update_local_meta, create_local_meta


class BroydenSolver(NonLinearSolver):
    """MIMO Newton-Raphson Solver with Broyden approximation to the Jacobian.
    Algorithms are based on those found in ``scipy.optimize`` around version
    13 (the latest version of scipy is a complete rewrite of the Broyden
    stack.)

        A collection of general-purpose nonlinear multidimensional solvers.

        - broyden2: Broyden's second method -- the same as broyden1 but
          updates the inverse Jacobian directly
        - broyden3: Broyden's second method -- the same as broyden2 but
          instead of directly computing the inverse Jacobian,
          it remembers how to construct it using vectors.
          When computing inv(J)*F, it uses those vectors to
          compute this product, thus avoiding the expensive NxN
          matrix multiplication.
        - excitingmixing: The excitingmixing algorithm. J=-1/alpha

        The broyden2 is the best. For large systems, use broyden3; excitingmixing is
        also very effective. The remaining nonlinear solvers from SciPy are, in
        their own words, of "mediocre quality," so they were not implemented.
    """

    def __init__(self):
        super(BroydenSolver, self).__init__()

        opt = self.options
        opt.add_option('algorithm', 'broyden2',
                       values=['broyden2', 'broyden3', 'excitingmixing'],
                       desc='Algorithm to use. Choose from broyden2, broyden3, and excitingmixing.')
        opt.add_option('alpha', 0.4,
                       desc='Mixing Coefficient.')
        opt.add_option('alphamax', 1.0,
                       desc='Maximum Mixing Coefficient (only used with excitingmixing.')
        opt.add_option('atol', 0.00001,
                       desc='Convergence tolerance. If the norm of the independent'
                       'vector is lower than this, then terminate successfully.')
        opt.add_option('maxiter', 10,
                       desc='Maximum number of iterations before termination.')
        opt.add_option('state_var', '',
                       desc="name of the state-variable/residual the solver should with")
        #opt.add_option('state_var_idx', 0,
        #               desc="Index into state_var if it is a vector.")

        self.print_name = 'BROYDEN'

        self.xin = None
        self.F = None

    def setup(self, sub):
        """ Initialization

        Args
        ----
        sub: `System`
            System that owns this solver.
        """

        state_name = self.options['state_var']
        if state_name.strip() == '':
            pathname = 'root' if sub.pathname=='' else sub.pathname
            msg = "'state_var' option in Brent solver of %s must be specified" % pathname
            raise ValueError(msg)

        if sub.is_active():
            n = sub.unknowns.metadata(state_name)['size']
            self.xin = np.zeros((n))
            self.F = np.zeros((n))

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system using the Brent Method.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        self.sys = system
        self.metadata = metadata
        self.local_meta = create_local_meta(self.metadata, self.sys.pathname)

        self.xin[:] = system.unknowns[self.options['state_var']]

        # perform an initial run for self-consistency
        system.children_solve_nonlinear(self.local_meta)
        system.apply_nonlinear(params, unknowns, resids)
        self.recorders.record_iteration(system, self.local_meta)

        # get initial dependents
        self.F[:] = system.resids[self.options['state_var']]

        # pick solver algorithm
        algorithm =self.options['algorithm']
        if algorithm == 'broyden2':
            self.execute_broyden2(params, unknowns, resids, system)
        elif algorithm == 'broyden3':
            self.execute_broyden3(params, unknowns, resids, system)
        elif algorithm == 'excitingmixing':
            self.execute_excitingmixing(params, unknowns, resids, system)

    def execute_broyden2(self, params, unknowns, resids, system):
        """From SciPy, Broyden's First method.

        Updates inverse Jacobian by an optimal formula.
        There is NxN matrix multiplication in every iteration.

        The best norm(F(x))=0.003 achieved in ~20 iterations.
        """
        maxiter = self.options['maxiter']
        state_name = self.options['state_var']
        alpha = self.options['alpha']
        atol = self.options['atol']
        iprint = self.options['iprint']

        xm = self.xin
        Fxm = self.F.copy()
        Gm = -alpha*np.identity(len(xm))
        basenorm = abs(Fxm)

        for n in range(maxiter):

            deltaxm = -Gm*Fxm
            xm = xm + deltaxm.T

            # Update the new independents in the model
            system.unknowns._dat[state_name].val[:] = xm

            # Run the model
            system.children_solve_nonlinear(self.local_meta)
            system.apply_nonlinear(params, unknowns, resids)
            self.recorders.record_iteration(system, self.local_meta)

            # Get dependents
            self.F[:] = system.resids[state_name]

            if iprint == 2:
                normval = abs(self.F)
                self.print_norm(self.print_name, system, self.iter_count, normval,
                                basenorm)

            # Successful termination if independents are below tolerance
            if norm(self.F) < atol:
                return

            Fxm1 = np.matrix(self.F).T
            deltaFxm = Fxm1 - Fxm

            if norm(deltaFxm) == 0:
                msg = "Broyden iteration has stopped converging. Change in " + \
                      "input has produced no change in output. This could " + \
                      "indicate a problem with your component connections. " + \
                      "It could also mean that this solver method is " + \
                      "inadequate for your problem."
                raise RuntimeError(msg)

            Fxm = Fxm1.copy()
            Gm = Gm + (deltaxm-Gm*deltaFxm)*deltaFxm.T/norm(deltaFxm)**2

            self.iter_count += 1

    def execute_broyden3(self, params, unknowns, resids, system):
        """from scipy, Broyden's second (sic) method.

        Updates inverse Jacobian by an optimal formula.
        The NxN matrix multiplication is avoided.

        The best norm(F(x))=0.003 achieved in ~20 iterations.
        """
        maxiter = self.options['maxiter']
        state_name = self.options['state_var']
        alpha = self.options['alpha']
        atol = self.options['atol']
        iprint = self.options['iprint']

        zy = []

        def updateG(z, y):
            """G:=G+z*y.T'"""
            zy.append((z, y))

        def Gmul(f):
            """G=-alpha*1+z*y.T+z*y.T ..."""
            s = -alpha*f
            for z, y in zy:
                s = s + z*(y.T*f)
            return s

        xm = self.xin
        Fxm = self.F.copy()
        basenorm = abs(Fxm)

        for n in range(maxiter):

            deltaxm = Gmul(-Fxm)
            xm = xm + deltaxm.T

            # Update the new independents in the model
            system.unknowns._dat[state_name].val[:] = xm

            # Run the model
            system.children_solve_nonlinear(self.local_meta)
            system.apply_nonlinear(params, unknowns, resids)
            self.recorders.record_iteration(system, self.local_meta)

            # Get dependents
            self.F[:] = system.resids[state_name]

            if iprint == 2:
                normval = abs(self.F)
                self.print_norm(self.print_name, system, self.iter_count, normval,
                                basenorm)

            # successful termination if independents are below tolerance
            if norm(self.F) < atol:
                return

            Fxm1 = self.F
            deltaFxm = Fxm1 - Fxm

            if norm(deltaFxm) == 0:
                msg = "Broyden iteration has stopped converging. Change in " + \
                      "input has produced no change in output. This could " + \
                      "indicate a problem with your component connections. " + \
                      "It could also mean that this solver method is " + \
                      "inadequate for your problem."
                raise RuntimeError(msg)

            Fxm = Fxm1.copy()
            updateG(deltaxm - Gmul(deltaFxm), deltaFxm/norm(deltaFxm)**2)

            self.iter_count += 1

    def execute_excitingmixing(self, params, unknowns, resids, system):
        """from scipy, The excitingmixing method.

        J=-1/alpha

        The best norm(F(x))=0.005 achieved in ~140 iterations.

        Note: SciPy uses 0.1 as the default value for alpha for this algorithm.
        Ours is set at 0.4, which is appropriate for Broyden2 and Broyden3, so
        adjust it accordingly if there are problems.
        """
        maxiter = self.options['maxiter']
        state_name = self.options['state_var']
        alpha = self.options['alpha']
        alphamax = self.options['alphamax']
        atol = self.options['atol']
        iprint = self.options['iprint']

        xm = self.xin.copy()
        beta = alpha*np.ones(len(xm))
        Fxm = self.F.T.copy()
        basenorm = abs(Fxm)

        for n in range(maxiter):

            deltaxm = beta*Fxm
            xm = xm + deltaxm

            # Update the new independents in the model
            system.unknowns._dat[state_name].val[:] = xm

            # Run the model
            system.children_solve_nonlinear(self.local_meta)
            system.apply_nonlinear(params, unknowns, resids)
            self.recorders.record_iteration(system, self.local_meta)

            # Get dependents
            self.F[:] = system.resids[state_name]

            if iprint == 2:
                normval = abs(self.F)
                self.print_norm(self.print_name, system, self.iter_count, normval,
                                basenorm)

            # successful termination if independents are below tolerance
            if norm(self.F) < atol:
                return

            Fxm1 = self.F.T

            for i in range(len(xm)):
                if Fxm1[i]*Fxm[i] > 0:
                    beta[i] = beta[i] + alpha
                    if beta[i] > alphamax:
                        beta[i] = alphamax
                else:
                    beta[i] = alpha

            Fxm = Fxm1.copy()

            self.iter_count += 1
