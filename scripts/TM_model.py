from pyDOE import lhs
from scipy.linalg import lstsq
import numpy as np
import time
import copy
from datetime import datetime
from NeuralNet import FNN
import sys
sys.path.append('../../utils')
from mtH1D import mt1dtm

# xavier_uniform initialization
def np_xavier_uniform(value,shape):
    fan_in=shape[0]*shape[1]
    fan_out=shape[0]*shape[1]
    limit=np.sqrt(6/(fan_in + fan_out))*value
    return np.random.uniform(-limit,limit,size=shape)
# initialization method
init_dict = {
    "default": None, # uniform
    "uniform": np.random.uniform,
    "normal": np.random.normal,
    "xavier_uniform": np_xavier_uniform,
}
class TM_class(object):
    def __init__(self,hp,Z_u,Y_test,Z_test,Sigma,freq,BCy,BCz):
        '''
        Z_u: elements in  boundaries, structural data;
        Y_test: all points in y direction(where there is test data); array
        Z_test: all points in Z direction(where there is test data); array
        Beta: media model parameters; array
        u_test: all E, tuple(real part, imag part)
        Bcy: list including postion of interface in y axis
        Bcz: list including postion of interface in z axis
        '''

        np.random.seed(1)
        self.mu_0 = 4*np.pi*1e-7    
        self.omega = 2*np.pi*freq
        self.y_nets = len(BCy)-1
        self.z_nets = len(BCz)-1
        self.num_nodes = hp["layers"][-2]
        self.y_points = hp["N_y"]
        self.z_points = hp["N_z"]
        self.y_inter  = hp["N_yi"]
        self.z_inter  = hp["N_zi"]
        self.dtype = "float64"
        self.np_dtype = np.float64
        self.np_complex = np.complex128

        self.model0_0 = FNN(hp["layers"],hp["activation"],hp["net_type"])
        self.weight_init(self.model0_0,hp["init"], hp["Rm1"],hp["Rm2"],hp["net_type"])
        # use same parameters for all neural networks
        for ii in range(self.z_nets):
            for jj in range(self.y_nets): 
                if ii !=0 or jj !=0:
                    setattr(self,'model{}_{}'.format(ii,jj),copy.deepcopy(self.model0_0))

        # initiaiza all parameters of the class
        self.to_model(Z_u,Y_test,Z_test,Sigma,freq,BCy,BCz) 
        self.Beta = np.sqrt(self.mu_0*self.Sigma*self.omega)

    def train(self, cond,driver,sample_method):
        start_time = time.time()
        self.TM_train(cond,driver,sample_method)
        running_time = time.time()-start_time
        print("all time: {:}".format(self.time_fd(running_time)))

    def TM_train(self,cond,driver,sample_method): 
        BCy        = self.BCy
        BCz        = self.BCz
        y_nets   = self.y_nets
        z_nets   = self.z_nets
        y_points = self.y_points
        z_points = self.z_points
        y_inter  = self.y_inter 
        z_inter  = self.z_inter 
        num_nodes  = self.num_nodes
        # for each sub-domain, we need calculate 4 boundaries, top and bottom(dy), left and right(dz) BC
        MM = (y_nets*z_nets)*(y_points*z_points+2*y_inter+2*z_inter)
        NN = (y_nets*z_nets)*num_nodes
        A = np.zeros((MM,NN),dtype=self.np_complex)
        b = np.zeros((MM,1), dtype=self.np_complex)
        II  = np.array([1j],dtype=self.np_complex)
        ########################   governing equation ############################################
        # build part of governing equation in A
        num_points = y_points*z_points # number of elements in a sub-domain
        for ii in range(z_nets): 
            for jj in range(y_nets): 
                model = getattr(self,'model{}_{}'.format(ii,jj))
                col_pos = (ii*y_nets+jj)
                if sample_method == "uniform":
                    YY = self.get_uniform_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                    ZZ = self.get_uniform_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                    YZ = np.concatenate((YY,ZZ),1)
                elif sample_method == "gaussian":
                    YY = self.get_gaussian_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                    ZZ = self.get_gaussian_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                    YZ = np.concatenate((YY,ZZ),1)
                elif sample_method == "lhs":
                    YZ = self.get_lhs_data(y_points,z_points,lb,ub)
                else: 
                    raise KeyError("bad sample method, please use gaussian or uniform ")
                # YY = self.get_uniform_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                # ZZ = self.get_uniform_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                # YZ = np.concatenate((YY,ZZ),1)
                # noticed that beta are identical in a domain
                Beta_ = self.get_uniform_data(self.Beta[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]],y_points,z_points,"Z")
                # print(np.unique(Beta_))
                lb = YZ.min(0)
                ub = YZ.max(0)
                setattr(self,'lb{}_{}'.format(ii,jj),lb)
                setattr(self,'ub{}_{}'.format(ii,jj),ub)
                u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
                u_yy = u_yy0 + u_zz0
                uu   = uu0*(Beta_**2)
                w_m = np.linalg.norm(np.concatenate((u_yy,uu),1), None,1,True)
                A[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
                del u_yy,uu
        ################################## END governing equation ##########################################

        ################################ top and bottom ############################################
        num_base = y_nets*z_nets*num_points # number of rows in governing equation
        for jj in range(y_nets): 
            for ii in range(z_nets):
                row_pos = 2*(z_nets*jj+ii)
                col_pos = (ii*y_nets+jj)
                model = getattr(self,'model{}_{}'.format(ii,jj))
                lb = getattr(self,'lb{}_{}'.format(ii,jj))
                ub = getattr(self,'ub{}_{}'.format(ii,jj))
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"tb")
                Beta_ = self.Beta[BCz[ii],BCy[jj]]
                if ii >0:
                    u_pred0, u_y00 = self.net_interface(model,Y_inter0,lb,ub,"Z")
                    u_y0 = u_y00/(Beta_**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(col_pos-y_nets)*num_nodes:(col_pos-y_nets+1)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                    del u_y0
                else:
                    u_pred,_ = model(self.Z_t[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                    A[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,:] = self.u_t[jj*y_inter:(jj+1)*y_inter,:]
                
                if ii <z_nets-1:
                    u_pred1, u_y11 = self.net_interface(model,Y_inter1,lb,ub,"Z")
                    u_y1 = u_y11/(Beta_**2)

                    A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1

                else:
                # Dirichlet conditons
                    u_pred,_ = model(self.Z_b[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                    A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,:] = self.u_b[jj*y_inter:(jj+1)*y_inter,:]
        ############################# END top and bottom  ################################################

        ################################ left and right ############################################
        num_base = y_nets*z_nets*num_points+(y_nets*z_nets*2)*y_inter # number of rows in governing equation
        for ii in range(z_nets): 
            for jj in range(y_nets):
                row_pos = (y_nets*ii+jj)*2
                col_pos = ((ii)*y_nets+jj)
                model = getattr(self,'model{}_{}'.format(ii,jj))
                lb = getattr(self,'lb{}_{}'.format(ii,jj))
                ub = getattr(self,'ub{}_{}'.format(ii,jj))
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"lr")
                Beta_ = self.Beta[BCz[ii],BCy[jj]]
                
                if jj>0 :# left
                    u_pred0, u_y00 = self.net_interface(model,Y_inter0,lb,ub,"Y")
                    u_y0 = u_y00/(Beta_**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*z_inter:num_base+(row_pos+1)*z_inter,(col_pos-1)*num_nodes:(col_pos)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*z_inter:num_base+(row_pos  )*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                    del u_y0
                else: # jj==0
                    u_pred,_ = model(self.Z_l[ii*z_inter :(ii+1)*z_inter ,0:2],lb,ub)
                    A[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,:] = self.u_l[ii*z_inter :(ii+1)*z_inter ,:]

                if jj< y_nets-1: # right
                    u_pred1, u_y11 = self.net_interface(model,Y_inter1,lb,ub,"Y")
                    u_y1 = u_y11/(Beta_**2)
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1


                else: # jj == y_nets-1:
                    u_pred,_ = model(self.Z_r[ii*z_inter:(ii+1)*z_inter ,0:2],lb,ub)
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,:] = self.u_r[ii*z_inter :(ii+1)*z_inter,:]

        ############################################ END left and right  ################################################

        start_time = time.time()
        YY,res,_,_ = lstsq(A,b,cond=cond, lapack_driver=driver)
        running_time = time.time()-start_time
        print("solve time: {:}".format(self.time_fd(running_time)))
        # print("residual= {res:.2e}")
        # print(res)
        # YY = np.from_numpy(YY.astype(self.np_dtype))

        # print(YY)
        for ii in range(self.z_nets):
            for jj in range(self.y_nets): 
                col_pos = ((ii)*y_nets+jj)
                model = getattr(self,'model{}_{}'.format(ii,jj))
                model.w1 =YY[(col_pos)*num_nodes:(col_pos+1)*num_nodes,:].T
                
    def net_f(self,model,YZ,beta,lb,ub): 
        u_f,_ = model(YZ,lb,ub)
        w0_0 = model.w0[:,0:1].T
        w0_1 = model.w0[:,1:2].T
        u_YY = (2*(u_f**3) - 2*(u_f))*((2/(ub[0]-lb[0])*w0_0)**2)
        u_ZZ = (2*(u_f**3) - 2*(u_f))*((2/(ub[1]-lb[1])*w0_1)**2)
        u_YY0 = u_YY
        u_ZZ0 = u_ZZ
        u0 = u_f
        return u_YY0,u_ZZ0,u0

    # return interface conditions
    def net_interface(self,model,y,lb,ub,flag):
        u_f,_ = model(y,lb,ub)
        w0_0 = model.w0[:,0:1].T
        w0_1 = model.w0[:,1:2].T
        if flag == "Y":
            u_Y = (1-u_f**2)*(2/(ub[0]-lb[0])*w0_0)
        elif flag =="Z":
            u_Y = (1-u_f**2)*(2/(ub[1]-lb[1])*w0_1)
        return u_f,u_Y

    def get_inter_data(self,y_data,z_data,y_points,z_points,flag):
        if flag =="tb": 
            y = np.linspace(y_data[0,0].item(),y_data[0,-1].item(),y_points)
            z = np.linspace(z_data[0,0].item(),z_data[0,-1].item(),y_points)
            y0 = np.concatenate((y.flatten()[:,None],z.flatten()[:,None]),1)
            y = np.linspace(y_data[-1,0].item(),y_data[-1,-1].item(),y_points)
            z = np.linspace(z_data[-1,0].item(),z_data[-1,-1].item(),y_points)
            y1 = np.concatenate((y.flatten()[:,None],z.flatten()[:,None]),1)
        if flag =="lr":
            y = np.linspace(y_data[0,0].item(),y_data[-1,0].item(),z_points)
            z = np.linspace(z_data[0,0].item(),z_data[-1,0].item(),z_points)
            y0 = np.concatenate((y.flatten()[:,None],z.flatten()[:,None]),1)
            y = np.linspace(y_data[0,-1].item(),y_data[-1,-1].item(),z_points)
            z = np.linspace(z_data[0,-1].item(),z_data[-1,-1].item(),z_points)
            y1 = np.concatenate((y.flatten()[:,None],z.flatten()[:,None]),1)
        return y0,y1

    def get_uniform_data(self,data, y_points,z_points,flag): 
        # include first and last element
        # data if format of numpy
        y = np.linspace(data[0,0].item(),data[0,-1].item(),y_points)
        z = np.linspace(data[0,0].item(),data[-1,0].item(),z_points)
        # different with np.meshgrid
        Y,Z = np.meshgrid(y,z)
        # return np.concatenate((Y.flatten()[:,None],Z.flatten()[:,None]),1)
        if flag =="Y":
            return Y.T.flatten()[:,None]
        if flag =="Z":
            return Z.T.flatten()[:,None]

    def get_gaussian_data(self,data,num_points): 
        # 1D gaussian quadrature
        lb = data[0,0].item()
        ub = data[-1,0].item()
        quad_y, _ = np.polznomial.legendre.leggauss(num_points-2)
        quad_y = quad_y.reshape(-1,1)
        y_grid = ((quad_y)*(ub-lb) + ub +lb)/2.0
        sig0 = data[0,1]
        sig = np.ones(num_points)*sig0
        Y = np.concatenate((data[0:1,0:1],np.from_numpy(y_grid.astype(self.np_dtype)),data[-1:,0:1]),0)
        Y_f = np.concatenate((Y,sig.flatten()[:,None]),1)
        return Y_f

    # generate points from Latin hypercube sampling
    def get_lhs_data(self, y_points,z_points,lb,ub): 
        YZ = lhs(2,y_points*z_points)*(ub-lb) + lb
        return YZ

    def weight_init(self, model, hp_init,value1,value2,net_type):
        II = np.array([0+1j],dtype=self.np_complex)
        if net_type == "real":
            w0_i = np.zeros(np.shape(model.w0))
            b0_i = np.zeros(np.shape(model.b0))
            if hp_init == "kaiming":
                s = 1./(np.shape(model.w0)[1])
                w0_r = np.random.rayleigh(s,size= np.shape(model.w0))
                b0_r = np.random.rayleigh(s,size= np.shape(model.b0))
                s = 1./(np.shape(model.w1)[1])
                w1_r = np.random.rayleigh(s,size= np.shape(model.w1))
                w1_i = np.random.rayleigh(s,size= np.shape(model.w1))
            elif hp_init == "xavier":
                s = 1./(np.sum(np.shape(model.w0)))
                w0_r = np.random.rayleigh(s,size= np.shape(model.w0))
                b0_r = np.random.rayleigh(s,size= np.shape(model.b0))
                s = 1./(np.sum(np.shape(model.w1)))
                w1_r = np.random.rayleigh(s,size= np.shape(model.w1))
                w1_i = np.random.rayleigh(s,size= np.shape(model.w1))
            else: 
                w0_r = init_dict[hp_init](value1,value2,size=np.shape(model.w0))
                b0_r = init_dict[hp_init](value1,value2,size=np.shape(model.b0))
                w1_r = init_dict[hp_init](value1,value2,size=np.shape(model.w1))
                w1_i = init_dict[hp_init](value1,value2,size=np.shape(model.w1))
        elif net_type == "complex":
            phase_w0 =np.random.uniform(-np.pi,np.pi,size=np.shape(model.w0))
            phase_b0 =np.random.uniform(-np.pi,np.pi,size=np.shape(model.b0))
            phase_w1 =np.random.uniform(-np.pi,np.pi,size=np.shape(model.w1))
            if hp_init == "kaiming":                        
                s = 1./(np.shape(model.w0)[1])
                modulus_w0 = np.random.rayleigh(s,size= np.shape(model.w0))
                modulus_b0 = np.random.rayleigh(s,size= np.shape(model.b0))
                s = 1./(np.shape(model.w1)[1])
                modulus_w1 = np.random.rayleigh(s,size= np.shape(model.w1))
            elif hp_init == "xavier":
                s = 1./(np.sum(np.shape(model.w0)))
                modulus_w0 = np.random.rayleigh(s,size= np.shape(model.w0))
                modulus_b0 = np.random.rayleigh(s,size= np.shape(model.b0))
                s = 1./(np.sum(np.shape(model.w1)))
                modulus_w1 = np.random.rayleigh(s,size= np.shape(model.w1))
            else: 
                modulus_w0 = init_dict[hp_init](value1,value2,size=np.shape(model.w0))
                modulus_b0 = init_dict[hp_init](value1,value2,size=np.shape(model.b0))
                modulus_w1 = init_dict[hp_init](value1,value2,size=np.shape(model.w1))
            
            w0_r = modulus_w0 * np.cos(phase_w0)
            w0_i = modulus_w0 * np.sin(phase_w0)
            b0_r = modulus_b0 * np.cos(phase_b0)
            b0_i = modulus_b0 * np.sin(phase_b0)
            w1_r = modulus_w1 * np.cos(phase_w1)
            w1_i = modulus_w1 * np.sin(phase_w1)
        else:
            raise KeyError('bad net types, just real or complex') 
        model.w0 = w0_r+w0_i*II
        model.b0 = b0_r+b0_i*II
        model.w1 = w1_r+w1_i*II

    def to_model(self,Z_u,Y_test,Z_test,Sigma,freq,BCy,BCz):
        self.Z_t = Z_u["Z_t"]
        self.Z_b = Z_u["Z_b"]
        self.Z_l = Z_u["Z_l"]
        self.Z_r = Z_u["Z_r"]
        self.u_t = Z_u["u_t"]
        self.u_b = Z_u["u_b"]
        self.u_l = Z_u["u_l"]
        self.u_r = Z_u["u_r"]
        self.Y_test = Y_test
        self.Z_test = Z_test
        self.Sigma = Sigma
        self.BCy = BCy
        self.BCz = BCz
        self.freq = freq

    def predict(self, Y_test,Z_test,BCy0,BCz0):
        BCy     = np.copy(BCy0)
        BCz     = np.copy(BCz0)
        BCy[-1] = BCy[-1]+1
        BCz[-1] = BCz[-1]+1
        u = np.empty(np.shape(Y_test),dtype=self.np_complex)
        for ii in range(self.z_nets):
            for jj in range(self.y_nets):
                model = getattr(self,'model{}_{}'.format(ii,jj))
                lb = getattr(self,'lb{}_{}'.format(ii,jj))
                ub = getattr(self,'ub{}_{}'.format(ii,jj))
                ny = BCz[ii+1]-BCz[ii]
                nz = BCy[jj+1]-BCy[jj]
                Y = Y_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                Z = Z_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                YZ = np.concatenate((Y,Z),1)
                _,u0 = model(YZ,lb,ub)
                u[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]] = u0.reshape(ny,nz)
        return u

    def predict_cpu(self, Y_test,Z_test,BCy,BCz):
        u = self.predict(Y_test,Z_test,BCy,BCz)
        return u

    def predict_derivative(self, Y_test,Z_test,ii,jj):
        '''
        Y_test: observation position(Y direction)
        Z_test: erath-air interface (Z direction)
        ii,jj: index of model

        output:
        u_f: [ur, ui]
        u_x: [ur_x,ui_x]
        '''
        Y_test = Y_test.flatten()[:,None]
        Z_test = Z_test.flatten()[:,None]
        model = getattr(self,'model{}_{}'.format(ii,jj))
        lb = getattr(self,'lb{}_{}'.format(ii,jj))
        ub = getattr(self,'ub{}_{}'.format(ii,jj))
        YZ = np.concatenate((Y_test,Z_test),1)
        sol = model.w1
        u0,u_x0 = self.net_interface(model,YZ,lb,ub,"Z")
        u = np.dot(u0,sol.T)
        u_x = np.dot(u_x0,sol.T)
        return u,u_x

    def cal_H(self,Y_obs,y,BCy):
        '''
        Y_obs: observation stations
        '''
        id_b = np.argwhere(y[BCy]<Y_obs[0])[-1][0]+1 # idx of model where obssevation begin(from 1) 
        id_e = len(BCy)-np.argwhere(y[BCy]>Y_obs[-1])[0][0]# idx of model where obssevation end 

        idx0 = np.where(Y_obs>y[BCy[id_b]])[0][0]
        Y_0 = Y_obs[0:idx0]
        Z_air = np.zeros(len(Y_0)) # air-earth interface
        u_pred, u_x = self.predict_derivative(Y_0,Z_air, 0,id_b-1)
        u_x = u_x/self.Sigma[0,BCy[id_b-1]]
        for ii in range(id_b+1,len(BCy)-id_e):
            idx = np.where(Y_obs>y[BCy[ii]])[0][0]
            Y_0 = Y_obs[idx0:idx]
            Z_air = np.zeros(len(Y_0))
            u_pred0, u_x0 = self.predict_derivative(Y_0,Z_air, 0,ii-1)
            u_x0 = u_x0/self.Sigma[0,BCy[ii-1]]
            u_pred = np.concatenate((u_pred,u_pred0),0)
            u_x    = np.concatenate((u_x,u_x0),0)
            idx0 = np.copy(idx)
        ii = len(BCy)-id_e
        Y_0 = Y_obs[idx:]
        Z_air = np.zeros(len(Y_0)) 
        u_pred0, u_x0 = self.predict_derivative(Y_0,Z_air, 0,ii-1)
        u_x0 = u_x0/self.Sigma[0,BCy[ii-1]]

        u_pred = np.concatenate((u_pred,u_pred0),0)
        u_x    = np.concatenate((u_x,u_x0),0)
        Hx   = u_pred
        Ey = u_x
        Zyx = Ey/Hx
        rho_a = np.abs(Zyx)**2/(self.omega*self.mu_0)
        phs_a = (np.arctan2(Zyx.imag,Zyx.real)* 180.0 / np.pi-180)*(-1.00)
        return Zyx, rho_a,phs_a

        
    def time_fd(self,time):
        return datetime.fromtimestamp(time).strftime("%M:%S")
    def pass_func(self,*args): 
        pass


# calculate the magnetic field on the boundaries
def BC_struct(Z,Y,nz,ny,BCz,BCy,freq,hL,sigL,hR,sigR):
    net_z = len(BCz)-1
    net_y = len(BCy)-1
    Z_t = np.ones((net_y*ny,2))
    Z_b = np.ones((net_y*ny,2))
    Z_l = np.ones((net_z*nz,2))
    Z_r = np.ones((net_z*nz,2))
    for jj in range(net_z):   # left and right    
        z0 = np.linspace(Z[BCz[jj],0],Z[BCz[jj+1]-1,0],nz)
        y0 = np.linspace(Y[BCz[jj],0],Y[BCz[jj+1]-1,0],nz)
        Z_l[jj*nz:(jj+1)*nz,:] = np.hstack((y0.flatten()[:,None],z0.flatten()[:,None]))
        z0 = np.linspace(Z[BCz[jj],-1],Z[BCz[jj+1]-1,-1],nz)
        y0 = np.linspace(Y[BCz[jj],-1],Y[BCz[jj+1]-1,-1],nz)
        Z_r[jj*nz:(jj+1)*nz,:] = np.hstack((y0.flatten()[:,None],z0.flatten()[:,None]))
    u0 = mt1dtm(freq,hL,sigL,Z_l[:,1])
    u_l= u0.flatten()[:,None]
    u0 = mt1dtm(freq,hR,sigR,Z_r[:,1])
    u_r= u0.flatten()[:,None]
    for jj in range(net_y): # top and bottom
        z0 = np.linspace(Z[0,BCy[jj]],Z[0,BCy[jj+1]-1],ny)
        y0 = np.linspace(Y[0,BCy[jj]],Y[0,BCy[jj+1]-1],ny)
        Z_t[jj*ny:(jj+1)*ny,:] = np.hstack((y0.flatten()[:,None],z0.flatten()[:,None]))
        z0 = np.linspace(Z[-1,BCy[jj]],Z[-1,BCy[jj+1]-1],ny)
        y0 = np.linspace(Y[-1,BCy[jj]],Y[-1,BCy[jj+1]-1],ny)
        Z_b[jj*ny:(jj+1)*ny,:] = np.hstack((y0.flatten()[:,None],z0.flatten()[:,None]))
    u1 = u0[0]*np.ones_like(Z_t[:,1])
    u_t = u1.flatten()[:,None]
    u1 = u0[-1]*np.ones_like(Z_b[:,1])
    u_b = u1.flatten()[:,None]
    Z_dict = {
        "Z_t": Z_t,
        "Z_b": Z_b,
        "Z_l": Z_l,
        "Z_r": Z_r,
        "u_t": u_t,
        "u_b": u_b,
        "u_r": u_r,
        "u_l": u_l

    }
    return Z_dict
