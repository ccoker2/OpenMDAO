""" Greiwank function with N continuous desgin variables and M integer desgin
variables. The Griewank function is a function widely used to test the
convergence of optimization functions.
"""

from six.moves import range
import numpy as np

from openmdao.core.component import Component


class Greiwank(Component):
    """ Greiwank function with N continuous desgin variables and M integer
    desgin variables

    Args
    -----
    num_int : int(2)
        Number of integer design variables.
    num_cont : int(2)
        Number of continuous design variables.
    """

    def __init__(self, num_int=2, num_cont=2):
        super(Greiwank, self).__init__()

        # Inputs
        self.add_param('xI', np.zeros((num_int)))
        self.add_param('xC', np.zeros((num_cont)))

        # Outputs
        self.add_output('f', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Define the function f(xI, xC)
        Here xI is integer and xC is continuous"""

        xI = params['xI']
        xC = params['xC']        
        
        f1I = np.sum((xI**2/4000.0))
        f1C = np.sum((xC**2/4000.0))
        
        f2C = 1.0
        f2I = 1.0

        for ii in range(len(xC)):
            f2C *= np.cos(xC[ii]/np.sqrt(ii+1.))

        for ii in range(len(xI)):
            f2I *= np.cos(xI[ii]/np.sqrt(ii+1.))

        unknowns['f'] = ((f1C+f1I) - (f2C*f2I) + 1.0)
