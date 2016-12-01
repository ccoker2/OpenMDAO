"""Aircraft Design and Airline Allocation using FLOPS to generate aircraft performance data.

- 6 + num_route*num_ac - continuous variables

- num_route*num_ac - inetger type variables

"""

import numpy as np

from openmdao.core.component import Component
import subprocess
import os
from mpi4py import MPI

def obj_cons_calc(xC, xI, xC_num_des, newac, existac, network):
    """ Calculate fleet-level profit of the airline and the aircraft performance constraints"""
    # xC = xClb + xC_val.flatten()*(xCub - xClb)
    Filename = newac.Filename
    num_route = network.num_route
    price = network.price
    dem = network.dem
    num_ac = len(existac.AC_name)+1
    trip = xI
    seat = np.concatenate((newac.seats,existac.seat_cap), axis=0)
    AC_num = np.concatenate((newac.AC_num_new,existac.AC_num), axis=0)
    # Continuous data extract
    pax = xC[xC_num_des:]
    x_con = xC[:xC_num_des]

    # Size the aircraft and fly the missions using FLOPS
    mission_route = network.route
    mission_pax = pax[:num_route]
    # os.chdir('/home/roger/a/roy10/OpenMDAO/openmdao/test/FLOPS_Files')
    FLOPSInputGen(x_con,mission_route,mission_pax,newac,Filename)
    os.chdir('/home/roger/a/roy10/AMIEGO_FLOPS/FLOPS_Files/FLOPS 8.12/FlopsLinux')
    cmmndline = "flops <"+Filename+".in > "+Filename+".out"
    out = subprocess.check_output(['bash','-c',cmmndline])
    # os.chdir('/home/roger/a/roy10/OpenMDAO/openmdao/test/FLOPS_Files')
    TD, LD, TOC, BH, acdata_count, nan_count = ReadFLOPSOutput(Filename)
    if nan_count > 0 and acdata_count != (1+num_route*num_ac):
        print('Mission Failed! Could not read FLOPS output.')
        TD = 10000.0
        LD = 10000.0
        TOC = 1.0e6*np.ones((num_route+1,1))
        BH = 25.0*np.ones((num_route+1,1))

    cost_1j = TOC[1:]
    BH_1j = BH[1:]
    # Calculate the profit & the constraints
    profit = 0.0
    g = np.zeros((2+num_ac,1))
    g_linineq = np.zeros((num_route,1))
    g[0,0] = (TD/8500.0)
    g[1,0] = (LD/7000.0)
    # Add more performance constraints (like fuselage fuel capacity, landing gear length etc) for the production run
    cc = 1
    for kk in range(num_ac):
        con_val = 0.0
        for jj in range(num_route):
            pax_kj = pax[kk*num_route + jj]
            x_kj = trip[kk*num_route + jj]
            if kk == 0: #New aircraft
                cost_kj = cost_1j[jj]
                BH_kj = BH_1j[jj]
                MH_FH_kj = newac.MH_new
            else:
                LF = int(10*round(pax_kj/seat[kk]))
                #TODO #This convention is different in matlab version (3D): dim1-routes,dim2-aircraft, dim3-LF
                #Extend TotCost_LF, BH to a 3D array
                cost_kj = existac.TotCost_LF[jj,LF]
                BH_kj = existac.BH[jj,LF]
                MH_FH_kj = existac.MH_FH[kk-1]

            profit += (price[jj]*x_kj*pax_kj) - (cost_kj*x_kj)
            con_val += x_kj*(BH_kj*(1.0+MH_FH_kj) + 1.0)

        g[cc+1,0] = (con_val/(12*AC_num[kk]))
        cc += 1

    # For the linear constraints.
    cc = 0
    for jj in range(num_route):
        pax_j = 0.0
        for kk in range(num_ac):
            x_kj = trip[kk*num_route + jj]
            pax_j += x_kj*pax[kk*num_route + jj]
        g_linineq[cc,0] = pax_j/dem[jj]
        cc += 1

    return profit, g, g_linineq

def FLOPSInputGen(x_con,mission_route,mission_pax,newac,Filename):
    '''Generate the input deck for FLOPS'''
    wt_pax = 165.0
    bag_pax = 30.0
    NPF = round(0.07*newac.seats)
    if np.mod(NPF,2) == 1:
        NPF = NPF-1
    NPT = newac.seats - NPF
    # Write in a new file
    titleline = 'Input deck ' + Filename
    fname = Filename + '.in'
    fid = open(fname,'w')
    fid.write(' '+titleline+' \r\n')
    fid.write(' $OPTION \r\n')
    fid.write('  IOPT=1, IANAL=3, ICOST=1, \r\n')
    fid.write(' $END \r\n')
    fid.write(' $WTIN \r\n')
    fid.write('  DGW='+str(newac.GW)+', \r\n')
    fid.write('  VMMO='+str(0.82)+', \r\n')
    fid.write('  DIH='+str(6.0)+', \r\n')
    fid.write('  HYDPR='+str(3000.0)+', \r\n')
    fid.write('  WPAINT='+str(0.033)+', \r\n')
    fid.write('  XL='+str(129.5)+', \r\n')
    fid.write('  WF='+str(12.33)+', \r\n')
    fid.write('  DF='+str(13.5)+', \r\n')
    fid.write('  XLP='+str(98.5)+', \r\n')
    fid.write('  SHT='+str(353.1)+', \r\n')
    fid.write('  SWPHT='+str(284.2)+', \r\n')
    fid.write('  ARHT='+str(353.1)+', \r\n')
    fid.write('  TRHT='+str(0.281)+', \r\n')
    fid.write('  TCHT='+str(0.09)+', \r\n')
    fid.write('  SVT='+str(284.2)+', \r\n')
    fid.write('  SWPVT='+str(39.4)+', \r\n')
    fid.write('  ARVT='+str(1.24)+', \r\n')
    fid.write('  TRVT='+str(0.386)+', \r\n')
    fid.write('  TCVT='+str(0.09)+', \r\n')
    fid.write('  NEW='+str(int(newac.NEW))+', \r\n')
    fid.write('  NPF='+str(int(NPF))+', \r\n')
    fid.write('  NPT='+str(int(NPT))+', \r\n')
    fid.write('  WPPASS='+str(wt_pax)+', \r\n')
    fid.write('  BPP='+str(bag_pax)+', \r\n')
    fid.write('  CARGOF='+str(5500)+', \r\n')
    fid.write('  WSRV=1.8, \r\n')
    fid.write('  IFUFU=1, \r\n')
    fid.write('  MLDWT=0,  WAPU=1.0,  WHYD=1.0,  \r\n')
    fid.write(' $END \r\n')

    fid.write(' $CONFIN \r\n')
    fid.write('  DESRNG='+str(newac.DESRNG)+', \r\n')
    fid.write('  GW='+str(newac.GW)+', \r\n')
    fid.write('  AR='+str(x_con[0])+', \r\n')
    fid.write('  TR='+str(x_con[1])+', \r\n')
    fid.write('  TCA='+str(x_con[2])+', \r\n')
    fid.write('  SW='+str(x_con[3])+', \r\n')
    fid.write('  SWEEP='+str(x_con[4])+', \r\n')
    fid.write('  THRUST='+str(x_con[5])+', \r\n')
    fid.write('  VCMN='+str(0.787)+', \r\n')
    fid.write('  CH='+str(41000.0)+', \r\n')
    fid.write('  HTVC='+str(2.84)+', \r\n')
    fid.write('  VTVC='+str(0.243)+', \r\n')
    fid.write('  OFG=1., OFF=0., OFC=0., \r\n')
    fid.write(' $END \r\n')

    fid.write(' $AERIN \r\n')
    fid.write('  VAPPR='+str(142.0)+', \r\n')
    fid.write('  AITEK='+str(1.819)+', \r\n')
    fid.write('  E=0.93365, \r\n')
    fid.write(' $END \r\n')

    fid.write(' $COSTIN \r\n')
    fid.write('  ROI='+str(7.0)+', \r\n')
    fid.write('  FARE='+str(0.10)+', \r\n')
    fid.write(' $END \r\n')

    fid.write(' $ENGDIN  \r\n')
    fid.write('  IDLE=1, IGENEN=1, \r\n')
    fid.write('  MAXCR=1, NGPRT=0, \r\n')
    fid.write(' $END  \r\n')

    fid.write(' $ENGINE \r\n')
    fid.write('  IENG=2, IPRINT=0, \r\n')
    fid.write('  OPRDES='+str(29.5)+', \r\n')
    fid.write('  FPRDES='+str(1.67)+', \r\n')
    fid.write('  TETDES='+str(2660.0)+', \r\n')
    fid.write(' $END \r\n')

    fid.write(' $MISSIN \r\n')
    fid.write('  IFLAG=2, \r\n')
    fid.write('  IRW=1, \r\n')
    fid.write('  ITTFF=1, \r\n')
    fid.write('  TAKOTM=0.4, \r\n')
    fid.write('  TAXOTM=10, \r\n')
    fid.write('  TAXITM=10, \r\n')
    fid.write('  FWF=-1, \r\n')
    fid.write('  THOLD=0.05, \r\n')
    fid.write('  RESRFU=0.05, \r\n')
    fid.write(' $END \r\n')

    fid.write('START \r\n')
    fid.write('CLIMB \r\n')
    fid.write('CRUISE \r\n')
    fid.write('DESCENT \r\n')
    fid.write('END \r\n')

    for jj in range(len(mission_route)):
        fid.write('$RERUN \r\n')
        fid.write('  mywts = 1, wsr = 0., twr = 0. , \r\n')
        fid.write('  desrng='+str(mission_route[jj])+', \r\n')
        payload = mission_pax[jj]*(wt_pax + bag_pax)
        fid.write('  paylod='+str(payload)+', \r\n')
        fid.write('$END \r\n');

        fid.write(' $MISSIN \r\n')
        fid.write('  IFLAG=0, \r\n')
        fid.write(' $END \r\n')

        fid.write('START \r\n')
        fid.write('CLIMB \r\n')
        fid.write('CRUISE \r\n')
        fid.write('DESCENT \r\n')
        fid.write('END \r\n')
    fid.close

def ReadFLOPSOutput(Filename):
    '''Read from the FLOPS output file'''
    TD = 0.0
    LD = 0.0
    TOC = np.array([0.0])
    BH = np.array([0.0])
    acdata_count = 0
    nan_count = 0
    fname = Filename + '.out'
    fid = open(fname,'r')
    nlines = 0
    count = 0
    line = fid.readline()
    while line != '':
        nlines = nlines + 1
        if len(line)>22 and line[:23] == ' DIRECT OPERATING COSTS':
            acdata_count += 1
            line = fid.readline()
            words = line.split()
            DOC = float(words[-1])
            line = fid.readline()
            words = line.split()
            DOCperBH = float(words[-1])
            for cc in range(24):
                line = fid.readline()
            words = line.split()
            IOC = float(words[-1])
            if count == 0: #This is for the design mission
                TOC = np.array([DOC + IOC])
                BH = np.array([DOC/DOCperBH])
            else: #Off design mission
                TOC = np.concatenate((TOC,np.array([DOC+IOC])),axis = 0)
                BH = np.concatenate((BH,np.array([DOC/DOCperBH])),axis = 0)
            if np.isnan(DOC) or np.isnan(IOC) or np.isnan(DOCperBH):
                nan_count += 1
                print('Nan found!')
        if len(line)>15 and line[:15] == '#OBJ/VAR/CONSTR' and count == 0:
            acdata_count += 1
            fid.readline()
            fid.readline()
            line = fid.readline()
            words = line.split()
            TD = float(words[3])
            LD = float(words[4])
            count = 1
            if np.isnan(TD) or np.isnan(LD):
                eflag = 0
        line = fid.readline()
    fid.close
    # os.remove(fname)
    # fnameIN = Filename + '.in'
    # os.remove(fnameIN)
    return TD, LD, TOC, BH, acdata_count, nan_count

class NewAC():
    def __init__(self):
        # Intermediate inputs about the 'yet-to-be-designed' aircraft
        # self.num_des = 6 #Number of aircraft design variables
        self.Filename = '/home/roger/a/roy10/Amiego_TestFiles/FLOPS_Files/AC_New' #A better version of B737-8ish aircraft
        self.AC_num_new = np.array([3])
        self.MH_new = 0.948
        self.DESRNG = 2940.0
        self.GW = 174900.0
        self.NEW = 2
        self.seats = np.array([162.0])

        #Estimates of costs & BH for failed cases
        self.cost_new = np.array([1.813e4, 1.4764e4, 1.1412e4])
        self.BH_new = np.array([5.158, 4.0455, 2.9284])

class ExistingAC():
    def __init__(self):
        self.AC_name = ['B757-200']
        self.AC_num=np.array([6])
        self.seat_cap=np.array([180.0])
        self.des_range=np.array([2800.0])
        self.MH_FH=np.array([0.948])
        #TODO: Extend below to a 3D array: dim1-routes, dim2-aircraft type, dim3-load factor
        self.TotCost_LF=np.array([[19610.25,	19672.35,	19735.97,	19801.21,	19865.22,	19931.52,	20001.05,	20073.62,	20147.71,	20224.14,	20302.37],\
                                            [15970.15,	16019.57,	16070.59,	16121.65,	16173.45,	16224.3,	16277.74,	16333.82,	16390.78,	16449.34,	16511.76],\
                                            [12338.39,	12379.84,	12419.3,	12458.03,	12497.9,	12534.93,	12576.46,	12617.69,	12659.47,	12702.61,	12748.24]])
        self.BH=np.array([[5.19228124923812,	5.19115497963286,	5.18972425626526,	5.18834917824564,	5.18578214484093,	5.18345047771080,	5.18118807187219,	5.17959335751242,	5.17795719607052,	5.17663719291697,	5.17541531322506],\
        [4.06265858539434,	4.06141998855617,	4.06017094017094,	4.05853762322404,	4.05669338781121,	4.05388093620411,	4.05174223096978,	4.05011302818046,	4.04813394919169,	4.04644578950029,	4.04574260866384],\
        [2.93144735623669,	2.93175768256619,	2.93065066132338,	2.92887222725733,	2.92730571625250,	2.92416112866325,	2.92246467860468,	2.92047842540803,	2.91851573104681,	2.91696139077071,	2.91578919331641]])

class NetworkData():
    def __init__(self,num_route):
        self.num_route = num_route
        if num_route == 3:
            self.route = np.array([2000.0, 1500.0, 1000.0])
            self.dem = np.array([300.0, 700.0, 220.0])
            self.price = np.array([463.1, 372.4, 282.9])
            self.num_route = num_route
        elif num_route == 11:
            self.route = np.array([2000.0, 1500.0, 1000.0])
            self.dem = np.array([300.0, 700.0, 220.0])
            self.price = np.array([463.1, 372.4, 282.9])
            self.num_route = num_route

class DesAllocFLOPS_1new1ex_3rt(Component):

    def __init__(self):
        super(DesAllocFLOPS_1new1ex_3rt, self).__init__()
        #Define the problem here
        self.num_ac = 2 #Total number of aircraft types (New + existing aircraft)
        self.num_route = 3 #Number of route
        self.xC_num_des = 6 # Number of aircraft design variables - 1.AR 2.TR 3.t2c 4.Area 5.Sweep[deg] 6.ThrustperEngine[lbs]
        self.existAC_index = [1] # Index of existing aircraft types - 1. B757-200 2.

        # Continuous Inputs
        self.add_param('xC', np.zeros((self.xC_num_des + self.num_ac*self.num_route,)),
                       desc='Continuous type design variables of the des-alloc problem.')

        # Integer Inputs
        self.add_param('xI', np.ones((self.num_ac*self.num_route,)), lower=0, upper=6,
                       desc='Integer type design variables of the des-alloc problem')

        # #Lower and upper bound of the continuous design-allocation variables
        # xClb_des = np.array([8.0,0.1,0.009,1000.0,0.5,20000.0])
        # xCub_des = np.array([12.0,0.5,0.17,2000.0,40.0,30000.0])
        #
        # xClb_alloc = np.zeros([self.num_ac*self.num_route])
        # xCub_alloc = np.array([162.0,162.0,162.0,180.0,180.0,180.0])
        #
        # self.xClb = np.concatenate((xClb_des,xClb_alloc),axis=0)
        # self.xCub = np.concatenate((xCub_des,xCub_alloc),axis=0)

        #TODO: In the future release read below from a file
        # New aircraft
        self.newac = NewAC()
        # Existing aircraft
        self.existac = ExistingAC()
        # Network data
        self.network = NetworkData(self.num_route)

        self.deriv_options['type'] = 'fd'

        # Outputs
        self.add_output('profit', val=0.0)
        self.add_output('g_val', val=np.zeros((2+self.num_ac,)))
        self.add_output('g_val_linineq', val=np.zeros((self.num_route,)))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Define the function f(xI, xC)
        Here xI is integer and xC is continuous"""

        xC = params['xC']
        xI = params['xI']

        if MPI.COMM_WORLD.rank == 0:
            # profit, g, g_linineq = obj_cons_calc(xC, xI, self.xC_num_des, self.xClb, self.xCub, self.newac, self.existac, self.network)
            profit, g, g_linineq = obj_cons_calc(xC, xI, self.xC_num_des, self.newac, self.existac, self.network)
        else:
            profit = 0.0
            g = np.zeros((2+self.num_ac,))
            g_linineq = np.zeros((self.num_route,))

        profit = MPI.COMM_WORLD.allgather(profit)[0]
        g = MPI.COMM_WORLD.allgather(g)[0]

        unknowns['profit'] = profit/-1.0e3
        unknowns['g_val'] = g
        unknowns['g_val_linineq'] = g_linineq
