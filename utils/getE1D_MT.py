'''
modified to python form Ralph-Uwe Boerner(2009)
2020-3-9
'''
import numpy as np

def getE1D(f,d,rho,zi): 
    '''
    Input:
    f: frequency,  float (only single value once a time)
    rho: (NL,) array of layer resistivities, numpy.array,
    d  : thickness of each layer, shape = (NL-1,), numpy.array;
    zi : (NZ,)coordination of output, numpy.array

    Output: 
    E at zi
    Note that E is normalized such that magnetic field H = 1 A/m at z = 0.

    '''
    if not isinstance(f, float): 
        raise ValueError("freq must be a float number, not array")

    pi     = np.pi 
    sigair = 1e-9
    mu0    = 4e-7*pi 
    # rho    = rho.T  # 变为列矩阵
    # d      = d.T
    II     = complex(0,1)
    nl     = len(rho)
    rho    = rho.flatten()[:,None]
    d      = d.flatten()[:,None]
    h      = np.hstack([0, np.cumsum(d)])
    iwm    = II*2*pi*f*mu0
    alpha  = np.zeros((nl,1),dtype=complex)
    b      = np.zeros((nl,1),dtype=complex)
    aa     = np.zeros((nl-1,1),dtype=complex)
    nz     = len(zi)
    E      = np.zeros((nz,1),dtype=complex)

    alpha = np.sqrt(iwm/rho)
    if nl == 1: 
        cl = iwm / alpha
    else: 
        alphad = alpha[0:nl-1]*d
        talphad = np.tanh(alphad)

        b[nl-1] = alpha[nl-1]
        for nn in range(nl-2,-1,-1): 
            b[nn] = alpha[nn] * (b[nn+1]+alpha[nn]*talphad[nn]) / (alpha[nn]+b[nn+1]*talphad[nn])
        
        cl = iwm/b[0]
        # Continuation from boundary to boundary
        for nn in range(nl-1): 
            aa[nn] = (b[nn]+alpha[nn]) / (b[nn+1]+alpha[nn]) * np.exp(-alpha[nn]*d[nn])
        
    for ii in range(nz):
        z = zi[ii]
        if z>=0: 
            if nl == 1: 
                a = np.exp(-alpha[nl-1]*z)
            else: 
                ind = np.argwhere(z>=h)[-1].item()
                if ind < nl-1: # not last layer
                    a = np.prod(aa[0:ind-1]) * 0.5 * (1+b[ind]/alpha[ind]) * \
                        (np.exp(-alpha[ind]*(z - h[ind])) - (b[ind+1] - alpha[ind]) / (b[ind+1] + alpha[ind]) * np.exp(-alpha[ind] * (d[ind] + h[ind+1] -z)))
                else: 
                    a = np.prod(aa) * np.exp(-alpha[ind] * (z - h[ind]))
        else: 
            k0 = np.sqrt(iwm * sigair)
            pr = (cl - iwm/k0) / (cl + iwm/k0)
            ar = k0 * z 
            a = np.exp(-ar) * (1 + pr * np.exp(2 * ar)) * 1 / (1 + pr)
        
        E[ii] = a * cl 
    
    return E