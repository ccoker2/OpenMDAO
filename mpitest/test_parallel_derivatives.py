""" Test out some specialized parallel derivatives features"""

from __future__ import print_function

import numpy as np

from openmdao.api import Group, ParallelGroup, Problem, IndepVarComp, \
    LinearGaussSeidel, Component
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.simple_comps import FanOutGrouped, FanInGrouped, FanOut3Grouped
from openmdao.test.exec_comp_for_test import ExecComp4Test
from openmdao.test.util import assert_rel_error
from openmdao.util.array_util import evenly_distrib_idxs

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl


class MatMatTestCase(MPITestCase):

    N_PROCS = 2

    def test_fan_in_serial_sets_rev(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()

        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.sub.ln_solver.options['mode'] = 'rev'

        prob.driver.add_desvar('p1.x1')
        prob.driver.add_desvar('p2.x2')
        prob.driver.add_objective('comp3.y')

        prob.setup(check=False)
        prob.run()

        indep_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_in_serial_sets_fwd(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()

        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.sub.ln_solver.options['mode'] = 'fwd'

        prob.driver.add_desvar('p1.x1')
        prob.driver.add_desvar('p2.x2')
        prob.driver.add_objective('comp3.y')

        prob.setup(check=False)
        prob.run()

        indep_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_serial_sets_fwd(self):

        prob = Problem(impl=impl)
        prob.root = FanOutGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()

        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.sub.ln_solver.options['mode'] = 'fwd'

        prob.driver.add_desvar('p.x')
        prob.driver.add_constraint('c2.y', upper=0.0)
        prob.driver.add_constraint('c3.y', upper=0.0)

        prob.setup(check=False)
        prob.run()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['p.x']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_out_serial_sets_rev(self):

        prob = Problem(impl=impl)
        prob.root = FanOutGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()

        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.sub.ln_solver.options['mode'] = 'rev'

        prob.driver.add_desvar('p.x')
        prob.driver.add_constraint('c2.y', upper=0.0)
        prob.driver.add_constraint('c3.y', upper=0.0)

        prob.setup(check=False)
        prob.run()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['p.x']

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_in_parallel_sets_fwd(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()
        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.sub.ln_solver.options['mode'] = 'fwd'

        prob.driver.add_desvar('p1.x1')
        prob.driver.add_desvar('p2.x2')
        prob.driver.add_desvar('p3.x3')
        prob.driver.add_objective('comp3.y')

        # make sure we can't mix inputs and outputs in parallel sets
        try:
            prob.driver.parallel_derivs(['p1.x1', 'comp3.y'])
        except Exception as err:
            self.assertEqual(str(err),
                             "['p1.x1', 'comp3.y'] cannot be grouped because ['p1.x1'] are "
                             "design vars and ['comp3.y'] are not.")
        else:
            self.fail("Exception expected")

        prob.driver.parallel_derivs(['p1.x1', 'p2.x2'])

        if MPI:
            expected = [('p1.x1', 'p2.x2'), ('p3.x3',)]
        else:
            expected = [('p1.x1',), ('p2.x2',), ('p3.x3',)]

        self.assertEqual(prob.driver.desvars_of_interest(),
                         expected)

        # make sure we can't add a VOI to multiple groups
        if MPI:
            try:
                prob.driver.parallel_derivs(['p1.x1', 'p3.x3'])
            except Exception as err:
                self.assertEqual(str(err),
                                 "'p1.x1' cannot be added to VOI set ('p1.x1', 'p3.x3') "
                                 "because it already exists in VOI set: ('p1.x1', 'p2.x2')")
            else:
                self.fail("Exception expected")

        prob.setup(check=False)
        prob.run()

        indep_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_in_parallel_sets_rev(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.sub.ln_solver.options['mode'] = 'rev'

        prob.driver.add_desvar('p1.x1')
        prob.driver.add_desvar('p2.x2')
        prob.driver.add_desvar('p3.x3')
        prob.driver.add_objective('comp3.y')

        prob.driver.parallel_derivs(['p1.x1', 'p2.x2'])

        if MPI:
            expected = [('p1.x1', 'p2.x2'), ('p3.x3',)]
        else:
            expected = [('p1.x1',), ('p2.x2',), ('p3.x3',)]

        self.assertEqual(prob.driver.desvars_of_interest(),
                         expected)

        prob.setup(check=False)
        prob.run()

        indep_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_parallel_sets_fwd(self):

        prob = Problem(impl=impl)
        prob.root = FanOutGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()

        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.sub.ln_solver.options['mode'] = 'fwd'

        prob.driver.add_desvar('p.x')
        prob.driver.add_constraint('c2.y', upper=0.0)
        prob.driver.add_constraint('c3.y', upper=0.0)
        prob.driver.parallel_derivs(['c2.y', 'c3.y'])  # ignored in fwd

        if MPI:
            expected = [('c2.y', 'c3.y')]
        else:
            expected = [('c2.y',), ('c3.y',)]

        self.assertEqual(prob.driver.outputs_of_interest(),
                         expected)

        prob.setup(check=False)
        prob.run()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['p.x']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_out_parallel_sets_rev(self):

        prob = Problem(impl=impl)
        prob.root = FanOutGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()

        # need to set mode to rev before setup. Otherwise the sub-vectors
        # for the parallel set vars won't get allocated.
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.sub.ln_solver.options['mode'] = 'rev'

        prob.driver.add_desvar('p.x')
        prob.driver.add_constraint('c2.y', upper=0.0)
        prob.driver.add_constraint('c3.y', upper=0.0)
        prob.driver.parallel_derivs(['c2.y', 'c3.y'])

        if MPI:
            expected = [('c2.y', 'c3.y')]
        else:
            expected = [('c2.y',), ('c3.y',)]

        self.assertEqual(prob.driver.outputs_of_interest(),
                         expected)

        prob.setup(check=False)
        prob.run()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['p.x']

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)


class ParDeriv3TestCase(MPITestCase):

    N_PROCS = 3

    def test_fan_out_parallel_sets(self):

        prob = Problem(impl=impl)
        prob.root = FanOut3Grouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()

        # need to set mode to rev before setup. Otherwise the sub-vectors
        # for the parallel set vars won't get allocated.
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.sub.ln_solver.options['mode'] = 'rev'

        # Parallel Groups
        prob.driver.add_desvar('p.x')
        prob.driver.add_constraint('c2.y', upper=0.0)
        prob.driver.add_constraint('c3.y', upper=0.0)
        prob.driver.add_constraint('c4.y', upper=0.0)
        prob.driver.parallel_derivs(['c2.y', 'c3.y', 'c4.y'])

        if MPI:
            expected = [('c2.y', 'c3.y', 'c4.y')]
        else:
            expected = [('c2.y',), ('c3.y',), ('c4.y', )]

        self.assertEqual(prob.driver.outputs_of_interest(),
                         expected)

        prob.setup(check=False)
        prob.run()

        unknown_list = ['c2.y', 'c3.y', 'c4.y']
        indep_list = ['p.x']

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)
        assert_rel_error(self, J['c4.y']['p.x'][0][0], 33.0, 1e-6)


class MatMatIndicesTestCase(MPITestCase):

    N_PROCS = 2

    def setup_model(self, mode):
        asize = 3
        prob = Problem(root=Group(), impl=impl)
        root = prob.root
        root.ln_solver = LinearGaussSeidel()
        root.ln_solver.options['mode'] = mode

        p = root.add('p', IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        G1 = root.add('G1', ParallelGroup())
        G1.ln_solver = LinearGaussSeidel()
        G1.ln_solver.options['mode'] = mode

        c2 = G1.add('c2', ExecComp4Test('y = x * 2.0', lin_delay=1.0,
                                        x=np.zeros(asize), y=np.zeros(asize)))
        c3 = G1.add('c3', ExecComp4Test('y = numpy.ones(3).T*x.dot(numpy.arange(3.,6.))',
                                        lin_delay=1.0,
                                        x=np.zeros(asize), y=np.zeros(asize)))
        c4 = root.add('c4', ExecComp4Test('y = x * 4.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))
        c5 = root.add('c5', ExecComp4Test('y = x * 5.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))

        prob.driver.add_desvar('p.x', indices=[1, 2])
        prob.driver.add_constraint('c4.y', upper=0.0, indices=[1])
        prob.driver.add_constraint('c5.y', upper=0.0, indices=[2])
        prob.driver.parallel_derivs(['c4.y', 'c5.y'])

        root.connect('p.x', 'G1.c2.x')
        root.connect('p.x', 'G1.c3.x')
        root.connect('G1.c2.y', 'c4.x')
        root.connect('G1.c3.y', 'c5.x')

        prob.setup(check=False)
        prob.run()

        return prob

    def test_indices_fwd(self):
        prob = self.setup_model('fwd')

        J = prob.calc_gradient(['p.x'],
                               ['c4.y', 'c5.y'],
                               mode='fwd', return_format='dict')

        assert_rel_error(self, J['c5.y']['p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['c4.y']['p.x'][0], np.array([8., 0.]), 1e-6)

    def test_indices_rev(self):
        prob = self.setup_model('rev')
        J = prob.calc_gradient(['p.x'], ['c4.y', 'c5.y'],
                               mode='rev', return_format='dict')

        assert_rel_error(self, J['c5.y']['p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['c4.y']['p.x'][0], np.array([8., 0.]), 1e-6)


class DistComp(Component):
    """Uses 2 procs and has output var slices"""
    def __init__(self, arr_size=4):
        super(DistComp, self).__init__()
        self.arr_size = arr_size
        self.add_param('invec', np.ones(arr_size, float))
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):

        p1 = params['invec'][0]
        p2 = params['invec'][1]

        unknowns['outvec'][0] = p1**2 - 11.0*p2
        unknowns['outvec'][1] = 7.0*p2**2 - 13.0*p1

    def linearize(self, params, unknowns, resids):
        """ Derivatives"""

        p1 = params['invec'][0]
        p2 = params['invec'][1]

        J = {}
        jac = np.zeros((2, 2))
        jac[0][0] = 2.0*p1
        jac[0][1] = -11.0
        jac[1][0] = -13.0
        jac[1][1] = 7.0*p2

        J[('outvec', 'invec')] = jac
        return J

    def setup_distrib(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs. Returns a dict of
        index arrays keyed to variable names.
        """

        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        self.set_var_indices('invec', val=np.ones(sizes[rank], float),
                             src_indices=np.arange(start, end, dtype=int))
        self.set_var_indices('outvec', val=np.ones(sizes[rank], float),
                             src_indices=np.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class ElementwiseParallelDerivativesTestCase(MPITestCase):

    N_PROCS = 2

    def test_simple_adjoint(self):
        top = Problem(impl=impl)
        root = top.root = Group()
        root.add('dcomp', DistComp())
        root.add('p1', IndepVarComp('x', np.ones((4, ))))

        top.driver.add_desvar('p1.x', np.ones((4, )))
        top.driver.add_objective('dcomp.outvec')
        top.root.connect('p1.x', 'dcomp.invec')

        top.setup(check=False)

        top['p1.x'][0] = 1.0
        top['p1.x'][1] = 2.0
        top['p1.x'][2] = 3.0
        top['p1.x'][3] = 4.0

        top.run()

        J = top.calc_gradient(['p1.x'], ['dcomp.outvec'], mode='rev')

        assert_rel_error(self, J[0][0], 2.0, 1e-6)
        assert_rel_error(self, J[0][1], -11.0, 1e-6)
        assert_rel_error(self, J[0][2], 4.0, 1e-6)
        assert_rel_error(self, J[0][3], -11.0, 1e-6)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

