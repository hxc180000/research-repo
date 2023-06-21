# 06/21/2023
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
#import func
import lsalg

def lbfgs(JFWI, m0, m, bulksp, datasp, iterMax, alpha, beta, c, verbose, eps, gamma, tol, Kmax, plot_on):
    
    # total gradient evaluation
    gtot = 0
    #total backtracking steps
    ktot = 0
    #total function evaluation
    jtot = 0
    
    k = 0
    s_set = []
    y_set = []
#    s = vcl.Vector(datasp)
#    y = vcl.Vector(bulksp)
    df = vcl.Vector(bulksp)
#    df_old = vcl.Vector(bulksp)
    q = vcl.Vector(bulksp)
    p = vcl.Vector(bulksp)
    r = vcl.Vector(bulksp)
    #m_old = vcl.Vector(datasp)
    m_old = m0.dup()
    #mest = vcl.Vector(datasp)
    #mest = vcl.Vector(datasp)
    #mest.copy(m0)
    mest = m0.dup()
    #print(f"The initial value of the objective function is: {JFWI(m0)}")
    df = JFWI.gradient(mest)
    #print(f"The gradient of the objective function is: {df.norm()}")
    df_old = df.dup()
    s = mest.dup()
    y = df.dup()
    while df.norm() > tol and k <= Kmax:

        p = lsalg.descent_direction_1(df, k, m, s_set, y_set, gamma)
        m_old.copy(mest)
        [i, mest, jtot, ktot] = lsalg.bt_lineSearch_1(JFWI, mest , df, p, k, iterMax, alpha, beta, c, verbose, jtot, ktot)
    
        if plot_on > 0:
            print(" ")
            print("The updated slowness: ")
            print(" ")
            linalg.simplot(mest.data, addcb=True, minval=2.5, maxval=4.5)
        
        if i == iterMax:
            print("The line search failed.")
            break
        f_obj = JFWI(mest)
        df_old.copy(df)
        df = JFWI.gradient(mest)
        gtot += 1
        
        # s_k = x_{k+1} - x_k
        s.copy(mest)
        s.linComb(-1,m_old,1)
        # y_k = df_{k+1} - df_k
        y.copy(df)
        y.linComb(-1,df_old,1)

        if k < m:
            s_set.append(s.dup())
            y_set.append(y.dup())

        if k >= m and s.dot(y) > eps*s.norm()*y.norm():# and y.dot(y) > eps*y.norm()**2:
            s_set.pop(0)
            s_set.append(s.dup())
            y_set.pop(0)
            y_set.append(y.dup())
        gamma = s_set[-1].dot(y_set[-1])/y_set[-1].dot(y_set[-1])
        k += 1

        if k >= Kmax:
            print("The BFGFS algorithm reached the maximum number of iterations.")
            break
    print(f"total function evals = {jtot}")
    print(f"total gradient evals = {gtot}")
    print(f"total backtracking steps = {ktot}")
    return mest



class lbfgsvcl(vcl.LSSolver):
    '''
    object interface for lbfgsvcl algorithm, for use in VPM and other
    applications requiring solution of lease squares problem:
    min_x |Ax-b|
    in terms of this description, the args are
    JFWI = A
    rhs = b
    returns solution x and residual b-Ax as
    a vector list.
    '''
    def __init__(self, iterMax, m, alpha, beta, c, eps, gamma, tol, Kmax, plot_on=0, verbose=0):
        self.iterMax = iterMax
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.eps = eps
        self.gamma = gamma
        self.tol = tol
        self.Kmax = Kmax
        self.plot_on = plot_on
        self.verbose = verbose
        

    def solve(self, f, rhs):
        try:
            if not isinstance(f, vcl.LinearOperator):
                raise Exception('first arg not LinearOperator')
                
            x = vcl.Vector(f.getDomain())
            e = vcl.Vector(f.getRange())
            ffin = vcl.Vector(f.getRange())
            e.copy(rhs)
            bulksp = f.getDomain()
            datasp = f.getRange()
            JFWI = vcl.LeastSquares(f,rhs)

            x = lbfgs(JFWI, x, self.m, bulksp, datasp, self.iterMax, self.alpha, self.beta, self.c,
                    self.verbose, self.eps, self.gamma, self.tol, self.Kmax, self.plot_on)
            ffin = f(x)
            e.linComb(-1.0,ffin)
            return [x, e]
        except Exception as ex:
            print(ex)
            raise Exception('called from optimizationAlg.lbfgsvcl.solve')
