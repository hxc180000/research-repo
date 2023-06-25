# 06/24/2023
import gc
import os
import vcl
import math
import linalg
from vcl import Vector
from vcl import transp
import time

# s_set and y_set stored the last m vectors of s and y, respectively
def descent_direction_1(df, k, m, s_set, y_set,gamma):
    q = df.dup()
    
    rho_set = [0 for i in range(m)]
    alpha_set = [0 for i in range(m)]
    
    if k >= m:
        for i in range(m-1,-1,-1):
            
            rho = 1 / y_set[i].dot(s_set[i])
            rho_set[i] = rho
            alpha = rho*(s_set[i].dot(q))
            alpha_set[i] = alpha
            #q = q - alpha * y_set[i]
            q.linComb(-alpha,y_set[i],1)
                
    #r = H_k_0 * q
    q.scale(gamma)
    r = q.dup()
    
    if k >= m:
        for i in range(0,m):
           
            beta = rho_set[i] * (y_set[i].dot(r))
            #r = r + s_set[i]*(alpha_set[i] - beta)
            r.linComb((alpha_set[i]-beta), s_set[i], 1)

    #Get -H_k * df_k = -r
    r.linComb(0,r,-1)
    #r.scale(-1.0)
    return r


# alpha: step length. The initial step length can be chosen to be 1
#       in Newton and quasi-Newton methods
# beta: contraction factor
# c: Armijo condition parameter
def bt_lineSearch_1(f, x, df, p, k, iterMax, alpha, beta, c, verbose, jtot, ktot, jetargs):
    start_time = time.time()
    if verbose > 0:
        print(" ")
        print(f"Iteration {k+1} of LBFGS:")
        print(f"i is the iteration counter for line search.")
    fx = f(x,**jetargs)
    fxValue = fx.value()
    df = df.dup()
    dfp = df.dot(p)
    i = 0
    #total function evaluation
    xn=x.dup()
    xn.linComb(alpha,p)
    fxn = f(xn,**jetargs)
    fxnValue = fxn.value()
    pnorm = p.norm()
    if verbose > 0:
        print(" ")
        print(f"Before Line Search:")
        print(f"i = 0 alpha = 0 f(x) =  + {fxValue} dfp = {dfp}")
        print(" ")
        print(f"During Line Search:")
    while fxnValue > fxValue + c*alpha*dfp and i <= iterMax:
        if dfp > 0:
            print("ERROR: POSITIVE GRADIENT.")
            break
        jtot += 1
        if verbose > 0:
            print(" ")
            print(f"fxvalue is {fxValue}")
            print(f"i={i}    alpha={alpha}   f(x)={fxValue}    f(x)-f(xn)={fxValue-fxnValue}")
            print(f"||p||={pnorm}     ||df||={df.norm()}    dfp={dfp} ")
            print(f"c*alpha*dfp={c*alpha*dfp}")
            print(" ")
        
        alpha = alpha*beta
        ktot += 1
        xn.copy(x)
        xn.linComb(alpha,p)
        fxn = f(xn,**jetargs)
        fxnValue = fxn.value()
        i = i+1
        
    if i < iterMax:
        x.copy(xn)
        if verbose > 0:
            print(" ")
            print(f"i={i}    alpha={alpha}   f(x)={fx}    f(x)-f(xn)={fxValue-fxnValue}")
            print(f"||p||={pnorm}     ||df||={df.norm()}    dfp={dfp} ")
            print(f"c*alpha*dfp={c*alpha*dfp}")
            print(" ")
            fxn = f(xn,**jetargs)
            fxnValue = fxn.value()
            print(f"k = {k+1} alpha = {alpha} f(xn) = {fxnValue} ||df|| = {fxn.gradient().norm()}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose > 0:
        print(f"The total run time for this line search is: {elapsed_time} sec.")
        print("-----------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------")
    return [i,x,jtot,ktot]


