""" Testing the AMIEGO driver."""

import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp, pyOptSparseDriver
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.core.mpi_wrap import MPI
from openmdao.test.branin import BraninInteger
from openmdao.test.griewank import Greiwank
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.three_bar_truss import ThreeBarTruss, ThreeBarTrussVector
from openmdao.test.util import assert_rel_error

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl


class TestAMIEGOdriver(MPITestCase):

    N_PROCS = 3

    def test_three_bar_truss(self):

        prob = Problem(impl=impl)
        root = prob.root = Group()

        root.add('xc_a1', IndepVarComp('area1', 5.0), promotes=['*'])
        root.add('xc_a2', IndepVarComp('area2', 5.0), promotes=['*'])
        root.add('xc_a3', IndepVarComp('area3', 5.0), promotes=['*'])
        root.add('xi_m1', IndepVarComp('mat1', 1), promotes=['*'])
        root.add('xi_m2', IndepVarComp('mat2', 1), promotes=['*'])
        root.add('xi_m3', IndepVarComp('mat3', 1), promotes=['*'])
        root.add('comp', ThreeBarTruss(), promotes=['*'])

        prob.driver = AMIEGO_driver()
        #prob.driver.cont_opt.options['tol'] = 1e-12
        #prob.driver.options['disp'] = False
        root.deriv_options['type'] = 'fd'
        prob.driver.cont_opt = pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

        prob.driver.add_desvar('area1', lower=0.0005, upper=10.0)
        prob.driver.add_desvar('area2', lower=0.0005, upper=10.0)
        prob.driver.add_desvar('area3', lower=0.0005, upper=10.0)
        prob.driver.add_desvar('mat1', lower=1, upper=4)
        prob.driver.add_desvar('mat2', lower=1, upper=4)
        prob.driver.add_desvar('mat3', lower=1, upper=4)
        prob.driver.add_objective('mass')
        prob.driver.add_constraint('stress', upper=1.0)

        npt = 5
        samples = np.array([[4, 2, 3],
                            [1, 3, 1],
                            [3, 1, 2],
                            [3, 4, 2],
                            [1, 1, 4]])

        prob.driver.sampling = {'mat1' : samples[:, 0].reshape((npt, 1)),
                                'mat2' : samples[:, 1].reshape((npt, 1)),
                                'mat3' : samples[:, 2].reshape((npt, 1))}

        prob.setup(check=False)

        prob.run()

        assert_rel_error(self, prob['mass'], 5.287, 1e-3)
        assert_rel_error(self, prob['mat1'], 3, 1e-5)
        assert_rel_error(self, prob['mat2'], 3, 1e-5)
        #Material 3 can be anything


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
