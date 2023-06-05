import data
import vcl
import rsfvc
import segyvc
import asg
import npvc
import vcalg
import linalg
import numpy as np
import time
import func
import lsalg
import optimizationAlg
import os

start_time = time.time()

# bulk modulus with lens
data.model(bulkfile='mstar.rsf', bulk=4.0, nx=401, nz=201, dx=20, dz=20,
           lensfac=0.7)

# bandpass filter source at sx=4200, sz=3000 (single trace)
data.bpfilt(file='wstar.su',nt=251,dt=8.0,s=1.0,f1=1.0,f2=2.5,f3=7.5,f4=12,sx=4200,sz=3000)
                          
# # scale the source
linalg.scale('wstar.su',1000)


# create zero data file with same source position, rz=500, rx=[2000,6000]
data.rechdr(file='data.su',nt=626,dt=8.0,ntr=201,rx=2000,rz=1000,sx=4200,sz=3000,drx=20)
    
# domain and range spaces
bulksp = rsfvc.Space('mstar.rsf')
datasp = segyvc.Space('data.su')


# the number of vector pair {s,y} we want to keep
m = 3
k = 0


# wrap bulk modulus in Vector
mstar = vcl.Vector(bulksp,'mstar.rsf')


# instantial modeling operator
F = asg.fsbop(dom=bulksp, rng=datasp, \
            buoyancy='bymstar.rsf', source_p='wstar.su', \
            order=2, sampord=1, nsnaps=20,\
            cfl=0.5, cmin=1.0, cmax=3.0,dmin=0.8, dmax=3.0,\
            nl1=250, nr1=250, nl2=250, nr2=250, pmlampl=1.0)


# evaluate F[mstar], create noise-free data
dstar = F(mstar)

# homogeneous bulk modulus
data.model(bulkfile='m0.rsf', bulk=4.0, nx=401, nz=201, dx=20, dz=20,
           lensfac=1.0)
m0 = vcl.Vector(bulksp,'m0.rsf')

# reset initial m
mest = vcl.Vector(bulksp)
mest.copy(m0)
t = 0.7
# m = t*mstar + (1-t)*m0
mest.linComb(t,mstar,1.0-t)


# create header file to define expanded velo space
os.system('sfput < mstar.rsf label="Real" unit="None" > cx.rsf')
cxsp = rsfvc.Space('cx.rsf')


# inverse map expanded c from bulkmod
inv = asg.invrntobulkfb(bulksp, cxsp, 'bymstar.rsf',
cmin=1.0, cmax=3.0, dmin=0.8, dmax=3.0)
# initial expanded velo model, corresponding to m
cx7 = inv(mest)


# forward map expanded c to bulkmod
fwd = asg.rntobulkfb(cxsp, bulksp, 'bymstar.rsf',
                    cmin=1.0, cmax=3.0, dmin=0.8, dmax=3.0)

# composition: modeling on expanded velo space
Fmod = vcl.comp(F, fwd)

# least-squares function
JFWI = vcl.LeastSquares(Fmod,dstar)



alpha = 1.0
c = 0.0001
iterMax = 100
verbose = 1
beta = 0.5
eps = 0.00000000000001
Kmax = 1200
tol = 0.00001
plot_on = 0 #  plot for the solution
gamma = 1

cx7 = optimizationAlg.lbfgs(JFWI, cx7, m, bulksp, cxsp, iterMax,
                            alpha, beta, c, verbose, eps, gamma, tol, Kmax, plot_on)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"The total run time for program is {elapsed_time} sec.")
