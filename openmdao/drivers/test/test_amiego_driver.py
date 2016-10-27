""" Testing the AMIEGO driver."""

import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp, pyOptSparseDriver
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.test.branin import BraninInteger
from openmdao.test.griewank import Greiwank
from openmdao.test.three_bar_truss import ThreeBarTruss, ThreeBarTrussVector
from openmdao.test.util import assert_rel_error


class TestAMIEGOdriver(unittest.TestCase):

    def test_simple_branin_opt(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('xC', 7.5), promotes=['*'])
        root.add('p2', IndepVarComp('xI', 0), promotes=['*'])
        root.add('comp', BraninInteger(), promotes=['*'])

        prob.driver = AMIEGO_driver()
        #prob.driver.options['disp'] = False
        root.deriv_options['type'] = 'fd'
        prob.driver.cont_opt = pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

        prob.driver.add_desvar('xI', lower=-5, upper=10)
        prob.driver.add_desvar('xC', lower=0.0, upper=15.0)

        prob.driver.add_objective('f')

        prob.driver.sampling = {'xI' : np.array([[0.0], [0.33], [0.66]])}

        prob.setup(check=False)
        prob.run()

        # Optimal solution
        assert_rel_error(self, prob['f'], 0.49398, 1e-5)
        #assert_rel_error(self, prob['xI'], -3.0, 1e-5)

    def test_three_bar_truss(self):

        prob = Problem()
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
        samples = np.array([[1.0, 0.25, 0.75],
                            [0.0, 0.75, 0.0],
                            [0.75, 0.0, 0.25],
                            [0.75, 1.0, 0.49],
                            [0.25, 0.49, 1.0]])

        prob.driver.sampling = {'mat1' : samples[:, 0].reshape((npt, 1)),
                                'mat2' : samples[:, 1].reshape((npt, 1)),
                                'mat3' : samples[:, 2].reshape((npt, 1))}

        prob.setup(check=False)

        prob.run()

        assert_rel_error(self, prob['mass'], 5.287, 1e-3)
        assert_rel_error(self, prob['mat1'], 3, 1e-5)
        assert_rel_error(self, prob['mat2'], 3, 1e-5)
        #Material 3 can be anything

    def test_three_bar_truss_vector(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('xc_a', IndepVarComp('area', np.array([5.0, 5.0, 5.0])), promotes=['*'])
        root.add('xi_m', IndepVarComp('mat', np.array([1, 1, 1])), promotes=['*'])
        root.add('comp', ThreeBarTrussVector(), promotes=['*'])

        prob.driver = AMIEGO_driver()
        prob.driver.cont_opt.options['tol'] = 1e-12
        #prob.driver.options['disp'] = False
        root.deriv_options['type'] = 'fd'
        prob.driver.cont_opt = pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

        prob.driver.add_desvar('area', lower=0.0005, upper=10.0)
        prob.driver.add_desvar('mat', lower=1, upper=4)
        prob.driver.add_objective('mass')
        prob.driver.add_constraint('stress', upper=1.0)

        npt = 5
        samples = np.array([[1.0, 0.25, 0.75],
                            [0.0, 0.75, 0.0],
                            [0.75, 0.0, 0.25],
                            [0.75, 1.0, 0.5],
                            [0.25, 0.5, 1.0]])
        prob.driver.sampling = {'mat' : samples}

        prob.setup(check=False)

        prob.run()

        assert_rel_error(self, prob['mass'], 5.287, 1e-3)
        assert_rel_error(self, prob['mat'][0], 3, 1e-5)
        assert_rel_error(self, prob['mat'][1], 3, 1e-5)
        # Material 3 can be anything

    def test_simple_greiwank_opt(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('xC', np.array([0.0, 0.0, 0.0])), promotes=['*'])
        root.add('p2', IndepVarComp('xI', np.array([0, 0, 0])), promotes=['*'])
        root.add('comp', Greiwank(num_cont=3, num_int=3), promotes=['*'])

        prob.driver = AMIEGO_driver()
        prob.driver.cont_opt.options['tol'] = 1e-12
        prob.driver.options['disp'] = True
        root.deriv_options['type'] = 'fd'

        prob.driver.add_desvar('xI', lower=-5, upper=5)
        prob.driver.add_desvar('xC', lower=-5.0, upper=5.0)

        prob.driver.add_objective('f')
        samples = np.array([[1.0, 0.25, 0.75],
                            [0.0, 0.75, 0.0],
                            [0.75, 0.0, 0.25],
                            [0.75, 1.0, 0.5],
                            [0.25, 0.5, 1.0]])
        # prob.driver.sampling = {'xI' : np.array([[0.0], [.76], [1.0]])}
        prob.driver.sampling = {'xI' : samples}

        prob.setup(check=False)
        prob.run()

        # Optimal solution
        assert_rel_error(self, prob['f'], 0.0, 1e-5)
        assert_rel_error(self, prob['xI'], 0.0, 1e-5)


if __name__ == "__main__":
    unittest.main()
