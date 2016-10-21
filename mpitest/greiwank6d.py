""" Griewank 6D problem """
from __future__ import print_function

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, IndepVarComp
from openmdao.core.mpi_wrap import MPI
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.test.griewank import Greiwank

if MPI:
   from openmdao.core.petsc_impl import PetscImpl as impl
else:
   from openmdao.core.basic_impl import BasicImpl as impl

prob = Problem(impl=impl)
root = prob.root = Group()

root.add('p1', IndepVarComp('xC', np.array([0.0, 0.0, 0.0])), promotes=['*'])
root.add('p2', IndepVarComp('xI', np.array([0, 0, 0])), promotes=['*'])
root.add('comp', Greiwank(num_cont=3, num_int=3), promotes=['*'])

prob.driver = AMIEGO_driver()
prob.driver.cont_opt.options['tol'] = 1e-12
prob.driver.options['disp'] = True
root.deriv_options['type'] = 'fd'
prob.driver.cont_opt = pyOptSparseDriver()
prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

prob.driver.add_desvar('xI', lower=-5, upper=5)
prob.driver.add_desvar('xC', lower=-5.0, upper=5.0)

prob.driver.add_objective('f')

samples = np.array([[1.0, 0.25, 0.75],
                   [0.0, 0.75, 0.0],
                   [0.75, 0.0, 0.25],
                   [0.75, 1.0, 0.5],
                   [0.25, 0.5, 1.0]])

prob.driver.sampling = {'xI' : samples}

prob.setup(check=False)

from time import time
t0 = time()
prob.run()
print("Elapsed Time", time()-t0)
