'''
计算视电阻率曲线，包括理论计算和有限差分法计算
'''
import numpy as np
import cmath as cm
from scipy.linalg import lu


# FD 方法计算TE模式电场
def mt1dtm(freq, thk,sigma,z):
    # constants
    mu  = 4.0e-7 * np.pi
    II  = cm.sqrt(-1)
    # Arguments:
    #   - freq: in Hz.
    #   - thk  : the thickness of each layer in m.
    #   - sigma : conductvity of each layer in sims/m.
    #   - z  : output position

    omega = 2.0 * np.pi * freq  
    h = np.hstack((0.0,np.cumsum(thk)))+z[0] 
    # 最后一个应该大一些，便于后面的分段函数映射sig值
    h[-1] = h[-1] +10.0
    nh = len(h) 
    sig = np.piecewise(z,[(z>=h[i])&(z<h[i+1]) for i in range(nh-1)],[sigma[i] for i in range(nh-1)])
    dz0 = z[1:] - z[:-1]
    nz = len(dz0)
    dz = np.hstack((dz0,np.sqrt(2.0/(sig[-1]*omega*mu))))

    # initialization
    diagA = np.zeros(nz,dtype=complex)
    offdiagA = np.zeros(nz-1,dtype=complex)
    for ki in range(nz):
        diagA[ki] = II*omega*mu*(dz[ki]+dz[ki+1])-2.0/(sig[ki]*dz[ki])-2.0/(sig[ki+1]*dz[ki+1])
    
    for ki in range(nz-1):
        offdiagA[ki] = 2.0/(sig[ki+1]*dz[ki+1])
        
    mtxA = (np.diag(diagA) +np.diag(offdiagA,-1) + np.diag(offdiagA,1))
    
    # BCs
    rhs = np.zeros((nz,1))
    rhs[0] = -2.0/(dz[0]*sig[0])
    
    P,L,U = lu(mtxA)
    hy  =  np.linalg.solve(U,np.linalg.solve(P.dot(L), rhs))
    hy0 = np.vstack((np.complex(1,0),hy))
    return hy0    

# 计算视电阻率曲线
def MT_1d(freq,thk,sigma,z,nza):
    # constants
    mu  = 4.0e-7 * np.pi
    II  = cm.sqrt(-1)
    omega = 2*np.pi*freq
    nf  = len(freq)
    # 3 initialize the returning arguments.
    rho = np.zeros(nf)
    phs = np.zeros(nf)
    zxy = np.zeros(nf,dtype=complex)
    for k in range(len(freq)):
        ex = mt1dte(freq[k],thk,sigma,z)
        exs = ex[nza]
        hys = (ex[nza+1] - ex[nza]) / (z[nza+1]-z[nza]) / II/omega[k]/mu
        zxy[k] = exs/hys
        rho[k] = abs(zxy[k]*zxy[k])/omega[k]/mu
        phs[k] = cm.phase(zxy[k]) * 180.0 / np.pi
        
    return rho, phs, zxy