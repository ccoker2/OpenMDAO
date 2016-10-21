
import sys

import unittest

from openmdao.util.concurrent import concurrent_eval_lb
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase


def funct(job, option=None):
    if job == 5:
        raise RuntimeError("Job 5 had an (intentional) error!")
    if MPI:
        rank = MPI.COMM_WORLD.rank
    else:
        rank = 0
    return (job*2, option, rank)


class Foo(object):
    def __init__(self, comm):
        if comm is None:
            self.rank = 0
        else:
            self.rank = comm.rank

    def mymethod(self, num):
        return self.rank * num, self.rank, num


class ConcurrentTestCase(MPITestCase):

    N_PROCS = 6

    def setUp(self):
        if MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.rank
        else:
            self.comm = None
            self.rank = 0

    def test_funct(self):
        # test for concurrent evaluation of a plain function
        
        comm, rank = self.comm, self.rank
        ncases = 10

        cases = [([i], {'option': 'foo_%d'%i}) for i in range(ncases)]

        results = concurrent_eval_lb(funct, cases, comm)

        if comm is None or comm.rank == 0:
            self.assertEqual(len(results), ncases)
            for ret, err in results:
                if ret is None:  # error case
                    self.assertTrue("Job 5 had an (intentional) error!" in err)
                else:
                    j2, opt, rank = ret
                    job = int(opt.split('_')[-1])
                    self.assertEqual(j2, job*2)
                    self.assertEqual(err, None)
        else:
            self.assertEqual(results, None)

    def test_method(self):
        # tests using an object method as the concurrently evaluated funct.

        comm, rank = self.comm, self.rank

        obj = Foo(comm)

        ncases = 10

        cases = [([i], None) for i in range(1, ncases+1)]

        results = concurrent_eval_lb(obj.mymethod, cases, comm)

        if comm is None or comm.rank == 0:
            self.assertEqual(len(results), ncases)
            for ret, err in results:
                tot, rnk, num = ret
                self.assertEqual(tot, rnk*num)
                self.assertEqual(err, None)
        else:
            self.assertEqual(results, None)

    def test_method_broadcast(self):
        # tests broadcasting of results to all procs in the comm
        comm, rank = self.comm, self.rank

        obj = Foo(comm)

        ncases = 10

        cases = [([i], None) for i in range(1, ncases+1)]

        results = concurrent_eval_lb(obj.mymethod, cases, comm, broadcast=True)

        self.assertEqual(len(results), ncases)
        for ret, err in results:
            tot, rnk, num = ret
            self.assertEqual(tot, rnk*num)
            self.assertEqual(err, None)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
