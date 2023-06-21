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
import lsalg
import optimizationAlg_1
import os
import vpm

n   = 4
sp0 = npvc.Space(n)
sp1 = npvc.Space(n)
sp2 = npvc.Space(2*n)

sp = vcl.ProductSpace([sp0, sp1])


F = vpm.sepexpl1(sp,sp2)

# print(F.getRange().dim)
# print(F.getDomain()[0].dim)

x = vcl.Vector(sp0)

x.data[0] = 1.0
x.data[1] = 0.0
x.data[2] = 0.0
x.data[3] = 0.0


#print('matrix at [1.0, 0.0, 0.0, 0.0] = ')

op = F.opfcn(x)

# print(op.mat)


b = vcl.Vector(F.getRange())

for i in range(F.getDomain()[1].dim):
    b.data[i]=(-i-1)**(i+1)



m = 3
alpha = 1.0
c = 0.0001
iterMax = 100
verbose = 0
beta = 0.5
eps = 0.00000000000001
Kmax = 1200
tol = 0.5*0.0001
plot_on = 0 #  plot for the solution
gamma = 1

S = optimizationAlg_1.lbfgsvcl(iterMax, m, alpha, beta, c,
                             eps, gamma, tol, Kmax, plot_on=0, verbose=0)


Jx = vpm.vpmjet(x, F, b, S)

print('value at [1.0, 0.0, 0.0, 0.0] = ' + str(Jx.value()))
print('gradient at [1.0, 0.0, 0.0, 0.0] = ')
print(Jx.gradient().data)
