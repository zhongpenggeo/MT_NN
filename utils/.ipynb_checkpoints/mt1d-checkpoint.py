'''
计算视电阻率曲线，包括理论计算和有限差分法计算
'''
import numpy as np
import cmath as cm
from scipy.linalg import lu

# constants
mu  = 4.0e-7 * np.pi
II  = cm.sqrt(-1)

# 理论视电阻率曲线
def mt1dan(freq, dz, sig):
    # Arguments:
    #   - freq: in Hz.
    #   - dz  : the thickness of each layer in m.
    #   - dz  : except the layer of air and the half-infinite layer. N-2
    #   - sig : conductvity of each layer in sims/m.
    #   - sig : except air layer N-1
    #
    # 1 define the constants.

    # 2 get the number of the layers and frequencies.
    nf  = len(freq)
    nz  = len(sig)
    # 3 initialize the returning arguments.
    rho = np.zeros(nf)
    phs = np.zeros(nf)
    zxy = np.zeros(nf,dtype=complex)
    # 4 loop over all frequencies.
    for kf in range(nf):
        omega = 2.0 * np.pi * freq[kf]
        Z     = np.sqrt(-II*omega*mu/sig[nz-1])
        for m in range(nz-2,-1,-1):
            km = np.sqrt(-II*omega*mu*sig[m])
            Z0 = -II*omega*mu/km
            Z  = np.exp(-2.0*km*dz[m]) * (Z-Z0) / (Z+Z0)
            Z  = Z0 * (1+Z) / (1-Z)
        zxy[kf] = Z
        rho[kf] = abs(zxy[kf]*zxy[kf]) / omega / mu
        phs[kf] = cm.phase(zxy[kf]) * 180.0 / np.pi
    # 5 return
    return rho, phs, zxy

# FD 方法计算TE模式电场
def mt1dte(freq, dz0, sig0):
    # Arguments:
    #   - freq: in Hz.
    #   - dz  : the thickness of each layer in m.
    #   - sig : conductvity of each layer in sims/m.

    omega = 2.0 * np.pi * freq    
    nz = len(sig0)
    sig = np.hstack((sig0,sig0[-1]))
    dz = np.hstack((dz0,np.sqrt(2.0/(sig[nz]*omega*mu))))

    # initialization
    diagA = np.zeros(nz,dtype=complex)
    offdiagA = np.zeros(nz-1,dtype=complex)
    for ki in range(nz):
        diagA[ki] = II*omega*mu*(sig[ki]*dz[ki]+sig[ki+1]*dz[ki+1])-2.0/dz[ki]-2.0/dz[ki+1]
    
    for ki in range(nz-1):
        offdiagA[ki] = 2.0/dz[ki+1]
        
    mtxA = (np.diag(diagA) +np.diag(offdiagA,-1) + np.diag(offdiagA,1))
    
    # BCs
    rhs = np.zeros((nz,1))
    rhs[0] = -2.0/dz[0]
    
    P,L,U = lu(mtxA)
    ex =  np.linalg.solve(U,np.linalg.solve(P.dot(L), rhs))
    ex0 = np.vstack((np.complex(1,0),ex))
    return ex0    

# 计算视电阻率曲线
def mt1d(freq,dz0,sig0,nza):
    omega = 2*np.pi*freq
    nf  = len(freq)
    # 3 initialize the returning arguments.
    rho = np.zeros(nf)
    phs = np.zeros(nf)
    zxy = np.zeros(nf,dtype=complex)
    for k in range(len(freq)):
        ex = mt1dte(freq[k],dz0,sig0)
        exs = ex[nza]
        hys = (ex[nza+1] - ex[nza]) / dz0[nza] / II/omega[k]/mu
        zxy[k] = exs/hys
        rho[k] = abs(zxy[k]*zxy[k])/omega[k]/mu
        phs[k] = cm.phase(zxy[k]) * 180.0 / np.pi
        
    return rho, phs, zxy
