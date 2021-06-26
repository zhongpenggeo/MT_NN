'''
recursive calculation of E and H in bc
modifid from ren zhengyong in C++
2021-5-2

'''
import numpy as np 


II = np.array([1j],dtype=np.complex) 
RR = np.array([1 ],dtype=np.complex) 
PI = np.pi
mu0 = 4*np.pi*1e-7
epsilon0 = 0.0

class BC_u(object):
    def __init__(self,ep,freq,mode):
        self.dtype = "float64"
        self.np_dtype = np.float64
        self.np_complex = np.complex128

        self.inner_parameters(ep,freq)
        self.get_T(mode)
        self.mode = mode
        #self.grad_u = grad_u

    def compute_E_H(self,p):
        mode = self.mode
        if mode =="TE":
            E,H = self.TE_E_H(p)
            return E,H
        elif mode =="TM":
            E,H,grad_Ey = self.TM_E_H(p)
            return E,H#,grad_Ey
        else: 
            raise KeyError("wrong mode, plese set TE or TM")
        

    # def get_background(self):

    def TE_E_H(self,p):
        '''
        #    x  (Ex)
        #    /
        #   / air  
        #   o----------y         TE-mode
        #   | earth
        #   |
        #   |
        #   z
        '''
        A = self.A 
        B = self.B
        ky = self.ky 
        kz = self.kz
        Y = self.Y
        N = self.N
        depth = self.depth
        z_hat = self.z_hat
        y = p[:,0]
        z = p[:,1]
        num_p = len(p)
        E = np.zeros((num_p,3),dtype=self.np_complex) # E=(Ex, 0, 0)
        H = np.zeros((num_p,3),dtype=self.np_complex) # H=(0, Hy, Hz)
        # H = np.zeros((num_p,1),dtype=self.np_complex) # H=(0, Hy, Hz)

        located_n = self.which_layer(p)
        for i in range(num_p):
            if located_n[i]==N:  
                E[i,0]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*z[i]) +\
                    B[located_n[i]]*np.exp( II*kz[located_n[i]]*z[i]))*\
                        np.exp(II*ky*y[i]) # Ex
                # temp = np.exp(II*ky*y[i])
                E[i,1]= 0.# Ey
                E[i,2]= 0.# Ez
                H[i,0]= 0.# Hx
                H[i,1]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*z[i]) -\
                    B[located_n[i]]*np.exp( II*kz[located_n[i]]*z[i]))*\
                        np.exp(II*ky*y[i])*Y[located_n[i]]# Hy
                H[i,2]= E[i,0]*(II*ky)/z_hat[0]     # Hz
            else:
                E[i,0]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*(z[i]-depth[located_n[i]])) +\
                    B[located_n[i]]*np.exp( II*kz[located_n[i]]*(z[i]-depth[located_n[i]])))*\
                        np.exp(II*ky*y[i]) #Ex
                # temp = np.exp(II*ky*y[i])
                E[i,1]= 0.
                E[i,2]= 0.
                H[i,0]= 0.
                H[i,1]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*(z[i]-depth[located_n[i]]))-
                    B[located_n[i]]*np.exp( II*kz[located_n[i]]*(z[i]-depth[located_n[i]])))*\
                        np.exp(II*ky*y[i])*Y[located_n[i]] # Hy
                H[i,2]= E[i,0]*(II*ky)/z_hat[0]     # Hz

        # E and H are calculated.
            # grad_Ex_y = z_hat[located_n[i]]*H[i,2]
            # grad_Ex_z = z_hat[located_n[i]]*H[i,1]*(-1.0)
            # grad_u[i] = grad_Ex_y + II*grad_Ex_z#?
        return E, H#, grad_u
  
    def TM_E_H(self,p):
        '''
        #    x  (Ex)
        #    /
        #   / air  
        #   o----------y         TM-mode
        #   | earth
        #   |
        #   |
        #   z
        '''
        A = self.A 
        B = self.B
        ky = self.ky 
        kz = self.kz
        Z = self.Z
        N = self.N
        depth = self.depth
        y_hat = self.y_hat
        y = p[:,0]
        z = p[:,1]
        num_p = len(p)
        E = np.zeros((num_p,3),dtype=self.np_complex) # E=(Ex, 0, 0)
        H = np.zeros((num_p,3),dtype=self.np_complex) # H=(0, Hy, Hz)
        grad_Ey = np.zeros((num_p,1),dtype=self.np_complex)

        located_n = self.which_layer(p)
        for i in range(num_p):
            if located_n[i]!=N:  
                H[i,0]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*(z[i]-depth[located_n[i]])) + 
                         B[located_n[i]]*np.exp( II*kz[located_n[i]]*(z[i]-depth[located_n[i]])))\
                      *np.exp(II*ky*y[i]) #Ex
                H[i,1]= 0.# Ey
                H[i,2]= 0.# Ez
                E[i,0]= 0.# Hx
                E[i,1]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*(z[i]-depth[located_n[i]])) - 
                         B[located_n[i]]*np.exp( II*kz[located_n[i]]*(z[i]-depth[located_n[i]])))\
                      *np.exp(II*ky*y[i])*Z[located_n[i]]
                E[i,2]= H[i,0]*(-II*ky)/y_hat[located_n[i]] 
                grad_Ey[i] = -II*kz[located_n[i]]*Z[located_n[i]]*H[i,0]
            else:
                H[i,0]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*z[i]) + 
                         B[located_n[i]]*np.exp( II*kz[located_n[i]]*z[i]))\
                      *np.exp(II*ky*y[i]) 
                H[i,1]= 0.
                H[i,2]= 0.
                E[i,0]= 0.
                E[i,1]= (A[located_n[i]]*np.exp(-II*kz[located_n[i]]*z[i]) - 
                         B[located_n[i]]*np.exp( II*kz[located_n[i]]*z[i]))\
                      *np.exp(II*ky*y[i])*Z[located_n[i]] 
                E[i,2]= H[i,0]*(-II*ky)/y_hat[located_n[i]]  
                grad_Ey[i] = -II*kz[located_n[i]]*Z[located_n[i]]*H[i,0]

        # E and H are calculated.
        return E, H, grad_Ey

    def which_layer(self,p):
        # p 暂时为[y,z]
        depth = self.depth
        N     = self.N
        # located_n = np.ones(len(p),dtype=int)*(-1)
        z = p[:,1]
        h_up   = np.zeros(N+1)
        h_down = np.zeros(N+1)
        h_up[0] = -1e50
        h_up[1:] = depth[:-1]
        h_down = depth.copy()
        # for piecewise
        h_down[-1] = h_down[-1]+1e50
        # located_n = -1
        # may np.piecewise is better
        # idx = np.arange(0,N+1,1)
        located_n = np.piecewise(z,[(z>=h_up[i])&(z<h_down[i]) for i in range(N+1)],[i for i in range(N+1)])

        # np.set_printoptions(threshold=np.inf)
        # print(located_n)

        # for j in range(N+1):
        #     if z[j]>h_up[j] and z[j<h_down[j]:
        #         located_n[j] = i
        #         break
        #     elif np.abs(z[j]-h_up[i]<1e-10):
        #         located_n[j] = i
        #         break
        #     elif np.abs(z[j]-h_down[i]<1e-10):
        #         located_n[j] = i
        #         break
            # assert(located_n[j]!=-1)
            # assert(located_n[j]>=0 and located_n[j]<N+1)

        return located_n.astype(int)

    def get_T(self,mode):
        B0 = 1e0 # ampltitute of Hx-TM, Ex-TE at z=0
        N  = self.N
        h  = self.h 
        ky = self.ky
        kz = self.kz
        Y  = self.Y 
        Z  = self.Z 
        depth = self.depth
        A = np.zeros(N+1, dtype=self.np_complex)
        B = np.zeros(N+1, dtype=self.np_complex)
        T = np.zeros((2*(N+1),2), dtype=self.np_complex)
        for i in range(1,N):
            if mode ==  "TM":
                T[2*i,0]   = np.exp( II*kz[i]*h[i])*( Z[i]/Z[i-1]+1.0)*0.5
                T[2*i,1]   = np.exp(-II*kz[i]*h[i])*(-Z[i]/Z[i-1]+1.0)*0.5
                T[2*i+1,0] = np.exp( II*kz[i]*h[i])*(-Z[i]/Z[i-1]+1.0)*0.5
                T[2*i+1,1] = np.exp(-II*kz[i]*h[i])*( Z[i]/Z[i-1]+1.0)*0.5
            elif mode == "TE":
                T[2*i,0]   = np.exp( II*kz[i]*h[i])*( Y[i]/Y[i-1]+1.0)*0.5
                T[2*i,1]   = np.exp(-II*kz[i]*h[i])*(-Y[i]/Y[i-1]+1.0)*0.5
                T[2*i+1,0] = np.exp( II*kz[i]*h[i])*(-Y[i]/Y[i-1]+1.0)*0.5
                T[2*i+1,1] = np.exp(-II*kz[i]*h[i])*( Y[i]/Y[i-1]+1.0)*0.5
            else: 
                raise KeyError("TE or TM mode")
        # N 
        if mode =="TM":
            T[2*N,0]   = np.exp(-II*kz[N]*depth[N-1])*( Z[N]/Z[N-1]+1.0)*0.5
            T[2*N,1]   = np.exp( II*kz[N]*depth[N-1])*(-Z[N]/Z[N-1]+1.0)*0.5
            T[2*N+1,0] = np.exp(-II*kz[N]*depth[N-1])*(-Z[N]/Z[N-1]+1.0)*0.5
            T[2*N+1,1] = np.exp( II*kz[N]*depth[N-1])*( Z[N]/Z[N-1]+1.0)*0.5
        elif mode =="TE":
            T[2*N,0]   = np.exp(-II*kz[N]*depth[N-1])*( Y[N]/Y[N-1]+1.0)*0.5
            T[2*N,1]   = np.exp( II*kz[N]*depth[N-1])*(-Y[N]/Y[N-1]+1.0)*0.5
            T[2*N+1,0] = np.exp(-II*kz[N]*depth[N-1])*(-Y[N]/Y[N-1]+1.0)*0.5
            T[2*N+1,1] = np.exp( II*kz[N]*depth[N-1])*( Y[N]/Y[N-1]+1.0)*0.5
        else: 
            raise KeyError("TE or TM mode")
        S = T[2:4,:]
        for i in range(2,N+1):
            S = np.dot(S,T[2*i:2*(i+1),:])

        S22 = S[1,1]
        S12 = S[0,1]
        assert(np.abs(S22)>1e-14)
        BN = RR*B0/S22
        A0 = S12*B0/S22

        B[0] = B0
        A[0] = A0
        A[N] = 0.0
        B[N] = BN 

        for j in range(N-1,0,-1):
            A_B_j_1 = np.zeros((2,1),dtype=self.np_complex)
            A_B_j_1[0] = A[j+1]
            A_B_j_1[1] = B[j+1]
            A_B_j = np.dot(T[2*(j+1):2*(j+2),:],A_B_j_1)
            A[j] = A_B_j[0]
            B[j] = A_B_j[1]
        
        self.A = A
        self.B = B



    def inner_parameters(self,ep, freq):
        '''
        k: k=a+b*i,a>0,b>0, k^2 = -z_hat*y_hat
        z_hat = -i*omega*\mu
        y_hat = sigma-i*omega*epsilon
        kz: vertical wavenumber in each layer
        depth: depth to bottom  of each layer
        ep: [sigma,relative epsilon, relative mu,depth] ?
        '''
        theta = 0 # rad
        N = len(ep)-1
        # k = np.zeros(N+1,dtype=self.np_complex)
        # z_hat = np.zeros(N+1,dtype=self.np_complex)
        # y_hat = np.zeros(N+1,dtype=self.np_complex)
        # depth = np.zeros(N+1)

        cond = ep[:,0] # sigma
        # epsilon = ep[:,1:2]*epsilon0
        epsilon = epsilon0*np.ones_like(cond)
        # mu = ep[:,2:3]*mu0
        mu = mu0*np.ones_like(cond)
        omega = 2*PI*freq
        z_hat = -II*omega*mu
        y_hat = cond-II*omega*epsilon
        k     = np.sqrt(-z_hat*y_hat)
        # assert(k.real>-1e-50)
        # assert(k.imag>-1e-50)
        # depth = ep[:,3:4]
        depth = ep[:,1]
        assert(depth[0]<1e-12)
        
        k_air = k[0]
        ky = k_air*np.sin(theta)
        kz = np.sqrt(k*k-ky*ky)

        # Z = np.zeros(N+1,dtype=self.np_complex)
        # Y = np.zeros(N+1,dtype=self.np_complex)
        # for n in range(N+1):
            # Z[n] = -II*kz[n]/y_hat[n]
            # Y[n] =  II*kz[n]/z_hat[n]
        
        Z = -II*kz/y_hat
        Y =  II*kz/z_hat
        # h
        h = np.zeros(N+1)
        h[0] = 1e50
        h[N] = 1e50
        h[1:N] = depth[1:N]-depth[0:N-1]

        # for i in range(N):
        #     assert(depth[i]>depth[i-1])
        #     h[i] = depth[i]-depth[i-1]
        self.h = h
        self.ky = ky
        self.kz = kz
        self.Y = Y
        self.Z = Z
        self.depth = depth
        self.N = N
        self.z_hat = z_hat
        self.y_hat = y_hat

    # def get_X(self,p):
    #     E,H = self.compute_E_H(p,self.mode)
    #     return E,H