def setup_seed(seed):
    np.random.seed(seed)
    # random.seed(seed)
    # np.manual_seed(seed) #cpu
    # np.cuda.manual_seed_all(seed)  #并行gpu
    # np.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    # np.backends.cudnn.benchmark = True

#from np.utils.tensorboard import SummarzWriter
from pyDOE import lhs
from scipy.linalg import lstsq
import numpy as np
# np.manual_seed(1)
# np.random.seed(1)
import time
import sys
import copy
#from tensorboardY import SummarzWriter
from datetime import datetime
# setup_seed(1)

from NeuralNet import FNN, input_minmax
# from basic_model import basic_model
sys.path.append("../utils")
# from bc import BC_u
# from lbfgsnew import LBFGSNew

init_dict = {
    "default": None, # uniform
    "uniform": np.random.uniform,
    "normal": np.random.normal,
    # "xavier_uniform": np_xavier_uniform,
    # "xavier_normal": nn.init.xavier_normal_,
}

# def np_xavier_uniform(value,shape):
#     fan_in=shape[0]*shape[1]
#     fan_out=shape[0]*shape[1]
#     limit=np.sqrt(6/(fan_in + fan_out))*value
#     return np.random.uniform(-limit,limit,size=shape)


class second_model(object):
    def __init__(self,hp,Z_u,Y_test,Z_test,Beta,Beta0,u_test,E0_model,BCy,BCz):
        '''
        Z_u: elements in  boundaries, structural data;
        Y_test: all points in y direction(where there is test data); array
        Z_test: all points in Z direction(where there is test data); array
        Beta: media model parameters; array
        u_test: all E, tuple(real part, imag part)
        Bcy: list including postion of interface in y axis
        Bcz: list including postion of interface in z axis
        '''

        setup_seed(1)
        self.cluster = hp["cluster"]
        net_layers = hp["layers"]
        self.y_nets = len(BCy)-1
        self.z_nets = len(BCz)-1
        self.num_nodes = hp["layers"][-2]
        self.y_points = hp["N_y"]
        self.z_points = hp["N_z"]
        self.y_inter  = hp["N_yi"]
        self.z_inter  = hp["N_zi"]
        self.N_inter  = hp["N_inter"]
        self.freq = hp["freq"]
        # self.h0   = hp["h0"]
        # self.sig0 = hp["sig0"]
        self.dtype = "float64"
        self.np_dtype = np.float64
        self.np_complex = np.complex128

        self.error_dict = {
            # "relative": self.relative_error,
            "rms"     : self.rms_error,
            "max"     : self.max_error,
            "none"    : self.pass_func,
        }

        # primary field
        self.BC_p = E0_model
        
        self.model0_0 = FNN(hp["layers"],hp["activation"],hp["net_type"])
        # self.weight_init(self.model0_0,hp["init"], hp["Rm1"],hp["Rm2"],hp["net_type"])
        for ii in range(self.z_nets-1):
            if ii==0:
                for jj in range(1,4):
                    setattr(self,'model{}_{}'.format(ii,jj),copy.deepcopy(self.model0_0))
            else:
                for jj in range(1,self.y_nets-1): 
                    setattr(self,'model{}_{}'.format(ii,jj),copy.deepcopy(self.model0_0))
        # for ii in range(self.z_nets):
            # for jj in range(self.y_nets): 
                # setattr(self,'model{}_{}'.format(ii,jj),FNN(hp["layers"],hp["activation"],hp["net_type"]))
                # self.weight_init(getattr(self,'model{}_{}'.format(ii,jj)),hp["init"], hp["Rm1"],hp["Rm2"],hp["net_type"])

        # self.to_cuda(hp["cuda"],Z_u,Y_test,Z_test,Beta,u_test,BCy,BCz)
        self.to_model(Z_u,Y_test,Z_test,Beta,Beta0,u_test,BCy,BCz)

    def train(self, cond,driver,sample_method,er):
        start_time = time.time()
        self.ls_train(cond,driver,sample_method)
        running_time = time.time()-start_time
        print("all time: {:}".format(self.time_fd(running_time)))
        # error0 = self.error_dict[er](self.predict(self.Y_test,self.Z_test,self.BCy,self.BCz),self.u_test)
        # if er != "none":
        #     print("relative error of u is: %.3e " %(error0))
        #     print(" ")

    def ls_train(self,cond,driver,sample_method): 
        BCy        = self.BCy
        BCz        = self.BCz
        y_nets   = self.y_nets
        z_nets   = self.z_nets
        y_points = self.y_points
        z_points = self.z_points
        y_inter  = self.y_inter 
        z_inter  = self.z_inter 
        N_inter  = self.N_inter
        num_nodes  = self.num_nodes
        
        # *2 是因为实虚部。因为内部边界有C1连续性需要两个条件，
        # 而外部边界只有Neumann或Direchlet条件，头尾加起来也刚好两个，
        # 就等于2*num_nets
        MM = (y_nets*z_nets)*(y_points*z_points+2*y_inter+2*z_inter) #+ (2*(y_nets-1))*(z_inter*N_inter)
        NN = ((y_nets-2)*(z_nets-2)+4)*num_nodes # 4 models in BC
        A = np.zeros((MM,NN),dtype=self.np_complex)
        b = np.zeros((MM,1), dtype=self.np_complex)
        II  = np.array([1j],dtype=self.np_complex)
        ########################   governing equation ############################################
        # 先求所有model里的governing equation，因为需要lb和ub
        # print("start governing equation")
        # start_time = time.time()
        num_points = y_points*z_points # number of elements in a domain
        for ii in range(1,z_nets-1): 

            for jj in range(1,y_nets-1): 
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    col_pos = ((ii-1)*(y_nets-2)+jj-1)+4
                    lb = np.array([self.Y_test[BCz[ii],BCy[jj]],self.Z_test[BCz[ii],BCy[jj]]])
                    ub = np.array([self.Y_test[BCz[ii+1],BCy[jj+1]],self.Z_test[BCz[ii+1],BCy[jj+1]]])
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
                    # noticed that beta are identical in a domain
                    Beta_ = self.Beta[BCz[ii],BCy[jj]]
                    Beta0 = self.Beta0[BCz[ii],BCy[jj]]
                    # print(np.unique(Beta_))
                    # lb = YZ.min(0)
                    # ub = YZ.max(0)
                    setattr(self,'lb{}_{}'.format(ii,jj),lb)
                    setattr(self,'ub{}_{}'.format(ii,jj),ub)
                    u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
                    u_yy = u_yy0 + u_zz0
                    uu   = uu0*(Beta_**2)

                    E,_ = self.BC_p.compute_E_H(YZ)
                    E0 = E[:,0:1]
                    # Hy = H[:,1:2]
                    u_y_all = np.concatenate((u_yy,uu),1)
                    w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, axis=1,keepdims=True)
                    A[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
                    b[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,:] = -E0*(Beta0**2)*II/w_m
                    del u_yy,uu
        for jj in [0,y_nets-1]:
            if jj==0:
                bc_lr=2
            else:
                bc_lr=3
            model = getattr(self,'model{}_{}'.format(0,bc_lr))
            col_pos = bc_lr
            lb = np.array([self.Y_test[BCz[1],BCy[jj]],self.Z_test[BCz[1],BCy[jj]]])
            ub = np.array([self.Y_test[BCz[z_nets-1],BCy[jj+1]],self.Z_test[BCz[z_nets-1],BCy[jj+1]]])
            if sample_method == "uniform":
                YY = self.get_uniform_data(self.Y_test[BCz[1]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                ZZ = self.get_uniform_data(self.Z_test[BCz[1]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                YZ = np.concatenate((YY,ZZ),1)
            elif sample_method == "gaussian":
                YY = self.get_gaussian_data(self.Y_test[BCz[1]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                ZZ = self.get_gaussian_data(self.Z_test[BCz[1]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                YZ = np.concatenate((YY,ZZ),1)
            elif sample_method == "lhs":
                YZ = self.get_lhs_data(y_points,z_points,lb,ub)
            else: 
                raise KeyError("bad sample method, please use gaussian or uniform ")
            # noticed that beta are identical in a domain
            Beta_ = self.Beta[BCz[1],BCy[jj]]
            Beta0 = self.Beta0[BCz[1],BCy[jj]]
            # print(np.unique(Beta_))
            # lb = YZ.min(0)
            # ub = YZ.max(0)
            setattr(self,'lb{}_{}'.format(0,bc_lr),lb)
            setattr(self,'ub{}_{}'.format(0,bc_lr),ub)
            u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
            u_yy = u_yy0 + u_zz0
            uu   = uu0*(Beta_**2)

            E,_ = self.BC_p.compute_E_H(YZ)
            E0 = E[:,0:1]
            # Hy = H[:,1:2]
            u_y_all = np.concatenate((u_yy,uu),1)
            w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, axis=1,keepdims=True)
            A[((y_nets+jj))*num_points:((y_nets+jj)+1)*num_points,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
            b[((y_nets+jj))*num_points:((y_nets+jj)+1)*num_points,:] = -E0*(Beta0**2)*II/w_m
            del u_yy,uu

        for ii in [0,z_nets-1]:
            if ii==0:
                bc_tb=0
            else:
                bc_tb=1
            jj = y_nets-1
            model = getattr(self,'model{}_{}'.format(0,bc_tb))
            col_pos = bc_tb
            lb = np.array([self.Y_test[BCz[ii],BCy[0]],self.Z_test[BCz[ii],BCy[0]]])
            ub = np.array([self.Y_test[BCz[ii+1],BCy[jj+1]],self.Z_test[BCz[ii+1],BCy[jj+1]]])
            if sample_method == "uniform":
                YY = self.get_uniform_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[0]:BCy[jj+1]+1],y_points,z_points,"Y")
                ZZ = self.get_uniform_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[0]:BCy[jj+1]+1],y_points,z_points,"Z")
                YZ = np.concatenate((YY,ZZ),1)
            elif sample_method == "gaussian":
                YY = self.get_gaussian_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[0]:BCy[jj+1]+1],y_points,z_points,"Y")
                ZZ = self.get_gaussian_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[0]:BCy[jj+1]+1],y_points,z_points,"Z")
                YZ = np.concatenate((YY,ZZ),1)
            elif sample_method == "lhs":
                YZ = self.get_lhs_data(y_points,z_points,lb,ub)
            else: 
                raise KeyError("bad sample method, please use gaussian or uniform ")
            # noticed that beta are identical in a domain
            Beta_ = self.Beta[BCz[ii],BCy[jj]]
            Beta0 = self.Beta0[BCz[ii],BCy[jj]]
            # print(np.unique(Beta_))
            # lb = YZ.min(0)
            # ub = YZ.max(0)
            setattr(self,'lb{}_{}'.format(0,bc_tb),lb)
            setattr(self,'ub{}_{}'.format(0,bc_tb),ub)
            E,_ = self.BC_p.compute_E_H(YZ)
            E0 = E[:,0:1]
            u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
            u_yy = u_yy0 + u_zz0
            uu   = uu0*(Beta_**2)
            u_y_all = np.concatenate((u_yy,uu),1)
            w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, axis=1,keepdims=True)
            A[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
            b[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,:] = -E0*(Beta0**2)*II/w_m
      

        # running_time = time.time()-start_time
        # print("governing equation computation time: {:}".format(self.time_fd(running_time)))
        ################################## END governing equation ##########################################


        ################################ top and bottom ############################################
        # print(" ")
        # print("start top and bottom")
        # start_time = time.time()
        num_base = y_nets*z_nets*num_points # number of rows in governing equation
        for jj in range(y_nets): 
            if jj==0 or jj==y_nets-1:
                if jj==0:
                    bc_lr=2
                else:
                    bc_lr=3
                ii = 0
                row_pos = 2*(z_nets*jj+ii)
                model = getattr(self,'model{}_{}'.format(0,0))
                lb = getattr(self,'lb{}_{}'.format(0,0))
                ub = getattr(self,'ub{}_{}'.format(0,0))
                u_pred,_ = model(self.Z_t[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                A[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,(0)*num_nodes:(1)*num_nodes] = u_pred
                b[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,:] = self.u_t[jj*y_inter:(jj+1)*y_inter,:]
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"tb")
                u_pred1, u_y1 = self.net_interface(model,Y_inter1,lb,ub,"Z")

                ii = 1
                row_pos = 2*(z_nets*jj+ii)
                model = getattr(self,'model{}_{}'.format(0,bc_lr))
                lb = getattr(self,'lb{}_{}'.format(0,bc_lr))
                ub = getattr(self,'ub{}_{}'.format(0,bc_lr))
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"tb")
                u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Z")
                u_y_all = np.concatenate((u_y0,u_y1),1)
                w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(0)*num_nodes:(1)*num_nodes] = u_y1/w_i
                A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = -1.0*u_y0/w_i
                A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(0)*num_nodes:(1)*num_nodes] = u_pred1
                A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = -1.0*u_pred0
                u_pred1, u_y1 = self.net_interface(model,Y_inter1,lb,ub,"Z")

                ii = z_nets-1
                model = getattr(self,'model{}_{}'.format(0,1))
                lb = getattr(self,'lb{}_{}'.format(0,1))
                ub = getattr(self,'ub{}_{}'.format(0,1))
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"tb")
                u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Z")
                u_y_all = np.concatenate((u_y0,u_y1),1)
                w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                A[num_base+(row_pos+2)*y_inter:num_base+(row_pos+3)*y_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = u_y1/w_i
                A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = u_pred1

                A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(1)*num_nodes:(2)*num_nodes] = -1.0*u_pred0
                A[num_base+(row_pos+2)*y_inter:num_base+(row_pos+3)*y_inter,(1)*num_nodes:(2)*num_nodes] = -1.0*u_y0/w_i

                row_pos = 2*(z_nets*jj+ii)
                u_pred,_ = model(self.Z_b[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(1)*num_nodes:(2)*num_nodes] = u_pred
                b[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,:] = self.u_b[jj*y_inter:(jj+1)*y_inter,:]


            else:
                for ii in range(z_nets):
                    row_pos = 2*(z_nets*jj+ii)
                    col_pos = ((ii-1)*(y_nets-2)+jj-1)+4
                    Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                        self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"tb")
                    if ii > 1 and ii<z_nets-1:
                        model = getattr(self,'model{}_{}'.format(ii,jj))
                        lb = getattr(self,'lb{}_{}'.format(ii,jj))
                        ub = getattr(self,'ub{}_{}'.format(ii,jj))
                        u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Z")

                        u_y_all = np.concatenate((u_y0,u_y1),1)
                        w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                        A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(col_pos-(y_nets-2))*num_nodes:(col_pos-y_nets+3)*num_nodes] = u_y1/w_i
                        A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                        A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                        del u_y0
                    elif ii ==1:
                        model = getattr(self,'model{}_{}'.format(ii,jj))
                        lb = getattr(self,'lb{}_{}'.format(ii,jj))
                        ub = getattr(self,'ub{}_{}'.format(ii,jj))
                        u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Z")

                        u_y_all = np.concatenate((u_y0,u_y1),1)
                        w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                        A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(0)*num_nodes:(1)*num_nodes] = u_y1/w_i
                        A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                        A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                        del u_y0
                    elif ii== z_nets-1:
                        model = getattr(self,'model{}_{}'.format(0,1))
                        lb = getattr(self,'lb{}_{}'.format(0,1))
                        ub = getattr(self,'ub{}_{}'.format(0,1))
                        u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Z")
                        u_y_all = np.concatenate((u_y0,u_y1),1)
                        w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                        A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(col_pos-(y_nets-2))*num_nodes:(col_pos-y_nets+3)*num_nodes] = u_y1/w_i
                        A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(1)*num_nodes:(2)*num_nodes] = -1.0*u_pred0
                        A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(1)*num_nodes:(2)*num_nodes] = -1.0*u_y0/w_i
                    else: # ii==0
                        model = getattr(self,'model{}_{}'.format(0,0))
                        lb = getattr(self,'lb{}_{}'.format(0,0))
                        ub = getattr(self,'ub{}_{}'.format(0,0))
                        u_pred,_ = model(self.Z_t[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                        A[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,(0)*num_nodes:(1)*num_nodes] = u_pred
                        b[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,:] = self.u_t[jj*y_inter:(jj+1)*y_inter,:]
                
                    if ii>0 and ii <z_nets-1:
                        model = getattr(self,'model{}_{}'.format(ii,jj))
                        lb = getattr(self,'lb{}_{}'.format(ii,jj))
                        ub = getattr(self,'ub{}_{}'.format(ii,jj))
                        u_pred1, u_y1 = self.net_interface(model,Y_inter1,lb,ub,"Z")
                        A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1
                    elif ii==0:
                        model = getattr(self,'model{}_{}'.format(0,0))
                        lb = getattr(self,'lb{}_{}'.format(0,0))
                        ub = getattr(self,'ub{}_{}'.format(0,0))
                        u_pred1, u_y1 = self.net_interface(model,Y_inter1,lb,ub,"Z")
                        A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(0)*num_nodes:(1)*num_nodes] = u_pred1

                    else:#ii==z_nets-1
                        model = getattr(self,'model{}_{}'.format(0,1))
                        lb = getattr(self,'lb{}_{}'.format(0,1))
                        ub = getattr(self,'ub{}_{}'.format(0,1))
                    # Dirichlet conditons
                        u_pred,_ = model(self.Z_b[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                        A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(1)*num_nodes:(2)*num_nodes] = u_pred
                        b[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,:] = self.u_b[jj*y_inter:(jj+1)*y_inter,:]
        # running_time = time.time()-start_time
        # print("top and bottom BC computation time: {:}".format(self.time_fd(running_time)))
        ############################# END top and bottom  ################################################

        ################################ left and right ############################################
        # print(" ")
        # print("start left and right")
        # start_time = time.time()
        num_base = y_nets*z_nets*num_points+(y_nets*z_nets*2)*y_inter # number of rows in governing equation
        for ii in range(1,z_nets-1): 
            for jj in range(y_nets):
                row_pos = (y_nets*ii+jj)*2
                col_pos = ((ii-1)*(y_nets-2)+jj-1)+4
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"lr")
                
                if jj>1 and jj<y_nets-1:# left
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Y")
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*z_inter:num_base+(row_pos+1)*z_inter,(col_pos-1)*num_nodes:(col_pos)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*z_inter:num_base+(row_pos  )*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                    del u_y0
                elif jj ==1:
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Y")
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*z_inter:num_base+(row_pos+1)*z_inter,(2)*num_nodes:(3)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*z_inter:num_base+(row_pos  )*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i

                elif jj==y_nets-1:
                    model = getattr(self,'model{}_{}'.format(0,3))
                    lb = getattr(self,'lb{}_{}'.format(0,3))
                    ub = getattr(self,'ub{}_{}'.format(0,3))
                    u_pred0, u_y0 = self.net_interface(model,Y_inter0,lb,ub,"Y")
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*z_inter:num_base+(row_pos+1)*z_inter,(col_pos-1)*num_nodes:(col_pos)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*z_inter:num_base+(row_pos  )*z_inter,(3)*num_nodes:(4)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *z_inter:num_base+(row_pos+1)*z_inter,(3)*num_nodes:(4)*num_nodes] = -1.0*u_y0/w_i

                else: # jj==0
                    model = getattr(self,'model{}_{}'.format(0,2))
                    lb = getattr(self,'lb{}_{}'.format(0,2))
                    ub = getattr(self,'ub{}_{}'.format(0,2))
                    u_pred,_ = model(self.Z_l[ii*z_inter :(ii+1)*z_inter ,0:2],lb,ub)
                    A[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,(2)*num_nodes:(3)*num_nodes] = u_pred
                    b[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,:] = self.u_l[ii*z_inter :(ii+1)*z_inter ,:]

                    # if Neumann conditions
                    # _, u_y = self.net_interface(model,self.Z_l[ii*z_inter:(ii+1)*z_inter,0:2],lb,ub,"Y")
                    # # wi = np.sum(np.abs(u_y),1,True)
                    # wi_r = np.linalg.norm(u_y.real,axis=1,keepdims=True)
                    # wi_i = np.linalg.norm(u_y.imag,axis=1,keepdims=True)
                    # wi = w_i_r + II*w_i_i
                    # wi = np.linalg.norm(np.abs(u_y_all),axis=1,keepdims=True)
                    # # wi = np.linalg.norm(u_y,axis=1,keepdims=True)
                    # A[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,col_pos*num_nodes:(col_pos+1)*num_nodes] = u_y/wi

                if jj>0 and jj< y_nets-1: # right
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    # Y_inter0: top of domain
                    # Y_inter1: bottom of domain
                    u_pred1, u_y1 = self.net_interface(model,Y_inter1,lb,ub,"Y")
                    # if jj ==0:
                    #     u_pred_1, u_y_1 = self.net_interface(model,Y_inter1,lb,ub,"Y")
                    #     u_pred1 = np.copy(u_pred_1)
                    #     u_y1    = np.copy(u_y_1)
                    #     bc_inter1 = 2.0/(ub-lb)
                    # else: 
                    #     u_y1    = u_y_1/bc_inter1[0]*bc1[0]
                   
                    # w_i = np.sum(np.abs(u_y1),1,keepdims=True)
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1
                    # del u_y1

                elif jj==0:
                    model = getattr(self,'model{}_{}'.format(0,2))
                    lb = getattr(self,'lb{}_{}'.format(0,2))
                    ub = getattr(self,'ub{}_{}'.format(0,2))
                    u_pred1, u_y1 = self.net_interface(model,Y_inter1,lb,ub,"Y")
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(2)*num_nodes:(3)*num_nodes] = u_pred1


                else: # jj == y_nets-1:
                    model = getattr(self,'model{}_{}'.format(0,3))
                    lb = getattr(self,'lb{}_{}'.format(0,3))
                    ub = getattr(self,'ub{}_{}'.format(0,3))
                    u_pred,_ = model(self.Z_r[ii*z_inter:(ii+1)*z_inter ,0:2],lb,ub)
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(3)*num_nodes:(4)*num_nodes] = u_pred
                    b[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,:] = self.u_r[ii*z_inter :(ii+1)*z_inter,:]

                    # if Neumann conditions
                    # _, u_y = self.net_interface(model,self.Z_r[ii*z_inter:(ii+1)*z_inter,0:2],lb,ub,"Y")
                    # wi_r = np.linalg.norm(u_y.real,axis=1,keepdims=True)
                    # wi_i = np.linalg.norm(u_y.imag,axis=1,keepdims=True)
                    # wi = w_i_r + II*w_i_i
                    # wi = np.linalg.norm(np.abs(u_y_all),axis=1,keepdims=True)
                    # # wi = np.linalg.norm(u_y,axis=1,keepdims=True)
                    # # wi = np.sum(np.abs(u_y),1,True)
                    # A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)  *num_nodes:(col_pos+1)*num_nodes] = u_y/wi

        for ii in [0,z_nets-1]:
            if ii==0:
                bc_tb=0
            else: 
                bc_tb=1
            model = getattr(self,'model{}_{}'.format(0,bc_tb))
            lb = getattr(self,'lb{}_{}'.format(0,bc_tb))
            ub = getattr(self,'ub{}_{}'.format(0,bc_tb))
            jj = y_nets-1
            row_pos = (y_nets*ii+jj)*2
            col_pos = bc_tb
            u_pred,_ = model(self.Z_l[ii*z_inter :(ii+1)*z_inter ,0:2],lb,ub)
            A[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
            b[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,:] = self.u_l[ii*z_inter :(ii+1)*z_inter ,:]

            u_pred,_ = model(self.Z_r[ii*z_inter:(ii+1)*z_inter ,0:2],lb,ub)
            A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
            b[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,:] = self.u_r[ii*z_inter :(ii+1)*z_inter,:]
 

        # running_time = time.time()-start_time
        # print("left and right BC computation time: {:}".format(self.time_fd(running_time)))
        ############################################ END left and right  ################################################
        if self.cluster == True:
            num_base = (y_nets*z_nets)*(y_points*z_points+2*y_inter+2*z_inter)
            num_points2 = z_inter * N_inter
            ii = 1
            for jj in range(y_nets-1): 
                row_pos = jj*2
                if jj == 0:
                    col_pos = ((ii)*y_nets+jj)
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    Y_test = self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1]
                    Z_test = self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1]
                    m,n = np.shape(Y_test)
                    Beta_ = self.Beta[BCz[ii],BCy[jj]]
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    # YY = self.get_uniform_data(Y_test[:,int(0.9*n):],N_inter,z_points,"Y")
                    # ZZ = self.get_uniform_data(Z_test[:,int(0.9*n):],N_inter,z_points,"Z")
                    # YZ = np.concatenate((YY,ZZ),1)
                    lb0 = np.array([Y_test[0,int(0.9*n)],Z_test[0,int(0.9*n)]])
                    ub0 = np.array([Y_test[-1,-1],Z_test[-1,-1]])
                    YZ = self.get_lhs_data(N_inter,z_inter,lb0,ub0)
                    YZ_put = np.copy(YZ)
                    u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
                    u_yy = u_yy0 + u_zz0
                    uu   = uu0*(Beta_**2)
                    w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, 1,True)
                    A[num_base+row_pos*num_points2:num_base+(row_pos+1)*num_points2,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m

                    jj0 = y_nets-1
                    col_pos = ((ii)*y_nets+jj0)
                    model = getattr(self,'model{}_{}'.format(ii,jj0))
                    Y_test = self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj0]:BCy[jj0+1]+1]
                    Z_test = self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj0]:BCy[jj0+1]+1]
                    m,n = np.shape(Y_test)
                    Beta_ = self.Beta[BCz[ii],BCy[jj0]]
                    lb = getattr(self,'lb{}_{}'.format(ii,jj0))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj0))
                    # YY = self.get_uniform_data(Y_test[:,:int(0.1*n)],N_inter,z_points,"Y")
                    # ZZ = self.get_uniform_data(Z_test[:,:int(0.1*n)],N_inter,z_points,"Z")
                    # YZ = np.concatenate((YY,ZZ),1)
                    lb0 = np.array([Y_test[0,0],Z_test[0,0]])
                    ub0 = np.array([Y_test[-1,int(0.1*n)],Z_test[-1,int(0.1*n)]])
                    YZ = self.get_lhs_data(N_inter,z_inter,lb0,ub0)
                    YZ_put = np.concatenate((YZ_put,YZ),0)
                    u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
                    u_yy = u_yy0 + u_zz0
                    uu   = uu0*(Beta_**2)
                    w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, 1,True)
                    A[num_base+(row_pos+1)*num_points2:num_base+(row_pos+2)*num_points2,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m

                else: 
                    col_pos = ((ii)*y_nets+jj)
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    Y_test = self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1]
                    Z_test = self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1]
                    m,n = np.shape(Y_test)
                    Beta_ = self.Beta[BCz[ii],BCy[jj]]
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    # YY = self.get_uniform_data(Y_test[:,:int(0.2*n)],N_inter,z_points,"Y")
                    # ZZ = self.get_uniform_data(Z_test[:,:int(0.2*n)],N_inter,z_points,"Z")
                    # YZ = np.concatenate((YY,ZZ),1)
                    lb0 = np.array([Y_test[0,0],Z_test[0,0]])
                    ub0 = np.array([Y_test[-1,int(0.2*n)],Z_test[-1,int(0.2*n)]])
                    YZ = self.get_lhs_data(N_inter,z_inter,lb0,ub0)
                    YZ_put = np.concatenate((YZ_put,YZ),0)
                    u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
                    u_yy = u_yy0 + u_zz0
                    uu   = uu0*(Beta_**2)
                    w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, 1,True)
                    A[num_base+row_pos*num_points2:num_base+(row_pos+1)*num_points2,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
                    # YY = self.get_uniform_data(Y_test[:,int(0.8*n):],N_inter,z_points,"Y")
                    # ZZ = self.get_uniform_data(Z_test[:,int(0.8*n):],N_inter,z_points,"Z")
                    # YZ = np.concatenate((YY,ZZ),1)
                    lb0 = np.array([Y_test[0,int(0.8*n)],Z_test[0,int(0.8*n)]])
                    ub0 = np.array([Y_test[-1,-1],Z_test[-1,-1]])
                    YZ = self.get_lhs_data(N_inter,z_inter,lb0,ub0)
                    YZ_put = np.concatenate((YZ_put,YZ),0)
                    u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta_,lb,ub)
                    u_yy = u_yy0 + u_zz0
                    uu   = uu0*(Beta_**2)
                    w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, 1,True)
                    A[num_base+(row_pos+1)*num_points2:num_base+(row_pos+2)*num_points2,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
            self.YZ_put = YZ_put

###################################################################################################################################################

        # self.A = A.clone()

        # the first n terms of YY is solution
        # YY,_ = np.lstsq(b,A)

        # print(" ")
        # print("star solving equation")
        start_time = time.time()
        # delete 0 in row
        idx = np.argwhere(np.all(np.abs(A[:,...]) == 0, axis=1))
        A2 = np.delete(A, idx, axis=0)
        b2 = np.delete(b, idx, axis=0)
        # print(MM-len(A2))
        YY,res,_,_ = lstsq(A2,b2,cond=cond,lapack_driver=driver)

        # running_time = time.time()-start_time
        # print("solve time: {:}".format(self.time_fd(running_time)))
        # print("residual= {res:.2e}")
        # print(res)
        # YY = np.from_numpy(YY.astype(self.np_dtype))

        # print(YY)
        for ii in range(4):
            model = getattr(self,'model{}_{}'.format(0,ii))
            model.w1 =YY[(ii)*num_nodes:(ii+1)*num_nodes,:].T

        for ii in range(1,z_nets-1): 
            for jj in range(1,y_nets-1):
                col_pos = ((ii-1)*(y_nets-2)+jj-1)+4
                model = getattr(self,'model{}_{}'.format(ii,jj))
                model.w1 =YY[(col_pos)*num_nodes:(col_pos+1)*num_nodes,:].T

                
    def net_f(self,model,YZ,beta,lb,ub): 
        # YY.requires_grad_()
        # ZZ.requires_grad_()
        # YZ = np.concatenate([YY,ZZ],1)
        u_f,_ = model(YZ,lb,ub)
        w0_0 = model.w0[:,0:1].T
        w0_1 = model.w0[:,1:2].T
        b0   = model.b0.T
        w0   = model.w0.T
        x0 = input_minmax(YZ,lb,ub)
        X = np.dot(x0,w0)+b0
        k0 = 2/(ub[0]-lb[0])*w0_0
        k1 = 2/(ub[1]-lb[1])*w0_1
        u_YY = u_f*(4*k0**2*(X)**2-2*k0**2)
        u_ZZ = u_f*(4*k1**2*(X)**2-2*k1**2)
        u_YY0 = u_YY
        u_ZZ0 = u_ZZ
        u0 = u_f
        # del u_f,u_u,u_Y,u_Z,u_YY,u_ZZ
        return u_YY0,u_ZZ0,u0

    # return interface conditions
    def net_interface(self,model,YZ,lb,ub,flag):
        u_f,_ = model(YZ,lb,ub)
        w0_0 = model.w0[:,0:1].T
        w0_1 = model.w0[:,1:2].T
        b0   = model.b0.T
        w0   = model.w0.T
        k0 = 2/(ub[0]-lb[0])*w0_0
        k1 = 2/(ub[1]-lb[1])*w0_1
        x0 = input_minmax(YZ,lb,ub)
        X = np.dot(x0,w0)+b0
        # u_u  = np.concatenate([self.gradients(u_f[:,ii],y)[0] for ii in range(u_f.size(1))],1)
        if flag == "Y":
            u_Y = -2*k0*X*u_f
        #     u_Y  = np.concatenate([u_u[:,2*ii:2*ii+1]   for ii in range(int(u_u.size(1)/2))],1)
        elif flag =="Z":
            u_Y = -2*k1*X*u_f
        #     u_Y  = np.concatenate([u_u[:,2*ii+1:2*ii+2] for ii in range(int(u_u.size(1)/2))],1)
        # del u_u
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
        # y0.requires_grad_()
        # y1.requires_grad_()
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

    def get_gaussian_data(self,data,y_points,z_points,flag): 
        '''
        data: Y or Z
        y_points: number of samples in y
        z_points: number of samples in z
        '''
        # 2D gaussian quadrature
        y_lb = data[0,0].item()
        y_ub = data[0,-1].item()        
        z_lb = data[0,0].item()
        z_ub = data[-1,0].item()        
        # to add top and bottom points
        quad_y, _ = np.polynomial.legendre.leggauss(y_points-2)
        quad_y = quad_y.reshape(-1,1)
        y_grid0 = ((quad_y)*(y_ub-y_lb) + y_ub +y_lb)/2.0
        y_grid = np.concatenate((data[0:1,0:1],y_grid0,data[0:1,-1:]),0)
        quad_z, _ = np.polynomial.legendre.leggauss(z_points-2)
        quad_z = quad_z.reshape(-1,1)
        z_grid0 = ((quad_z)*(z_ub-z_lb) + z_ub +z_lb)/2.0
        z_grid = np.concatenate((data[0:1,0:1],z_grid0,data[-1:,0:1]),0)
        Y,Z = np.meshgrid(y_grid.flatten(),z_grid.flatten()) 
        if flag =="Y":
            return Y.T.flatten()[:,None]
        if flag =="Z":
            return Z.T.flatten()[:,None]

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

    # def weight_init2(self, model, hp_init,value1,value2,net_type):
    #     II = np.array([0+1j],dtype=self.np_complex)
    #     w0_r = np.random.uniform(size=np.shape(model.w0))
    #     w0_i = np.random.uniform(size=np.shape(model.w0))
    #     b0_r = np.random.uniform(size=np.shape(model.b0))
    #     b0_i = np.random.uniform(size=np.shape(model.b0))
    #     w0_z = w0_r + w0_i*II 
    #     b0_z = b0_r + b0_i*II 
    #     w0_u,_w0_v = np.linalg.svd(w0_z)
    #     b0_u,_b0_v = np.linalg.svd(b0_z)

    def to_model(self,Z_u,Y_test,Z_test,Beta,Beta0,u_test,BCy,BCz):
        self.Z_t = Z_u["Z_t"]
        self.Z_b = Z_u["Z_b"]
        self.Z_l = Z_u["Z_l"]
        self.Z_r = Z_u["Z_r"]
        self.u_t = Z_u["u_t"]
        self.u_b = Z_u["u_b"]
        self.u_l = Z_u["u_l"]
        self.u_r = Z_u["u_r"]
        self.u_test = u_test
        self.Y_test = Y_test
        self.Z_test = Z_test
        self.Beta = Beta
        self.Beta0 = Beta0
        self.BCy = BCy
        self.BCz = BCz

    # def to_cuda(self,use_cuda,Z_u,Y_test,Z_test,Beta,u_test,BCy,BCz):
    #     self.Z_t = np.from_numpy(Z_u["Z_t"].astype(self.np_dtype))
    #     self.Z_b = np.from_numpy(Z_u["Z_b"].astype(self.np_dtype))
    #     self.Z_l = np.from_numpy(Z_u["Z_l"].astype(self.np_dtype))
    #     self.Z_r = np.from_numpy(Z_u["Z_r"].astype(self.np_dtype))
    #     self.u_t = np.from_numpy(Z_u["u_t"].astype(self.np_dtype))
    #     self.u_b = np.from_numpy(Z_u["u_b"].astype(self.np_dtype))
    #     self.u_l = np.from_numpy(Z_u["u_l"].astype(self.np_dtype))
    #     self.u_r = np.from_numpy(Z_u["u_r"].astype(self.np_dtype))
    #     self.u_test = (np.from_numpy(u_test[0].astype(self.np_dtype)),np.from_numpy(u_test[1].astype(self.np_dtype)))
    #     self.Y_test = np.from_numpy(Y_test.astype(self.np_dtype))
    #     self.Z_test = np.from_numpy(Z_test.astype(self.np_dtype))
    #     self.Beta = np.from_numpy(Beta.astype(self.np_dtype))
    #     self.BCy = np.from_numpy(BCy)
    #     self.BCz = np.from_numpy(BCz)

    def predict(self, Y_test,Z_test,BCy0,BCz0):
        BCy     = np.copy(BCy0)
        BCz     = np.copy(BCz0)
        BCy[-1] = BCy[-1]+1
        BCz[-1] = BCz[-1]+1
        u = np.empty(np.shape(Y_test),dtype=self.np_complex)
        for ii in range(1,self.z_nets-1):
            for jj in range(self.y_nets):
                if jj==0 or jj==self.y_nets-1:
                    if jj==0:
                        bc_lr=2
                    else: 
                        bc_lr=3
                    model = getattr(self,'model{}_{}'.format(0, bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(0,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(0,bc_lr))
                    nz = BCz[ii+1]-BCz[ii]
                    ny = BCy[jj+1]-BCy[jj]
                    Y = Y_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                    Z = Z_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                    YZ = np.concatenate((Y,Z),1)
                    _,u0 = model(YZ,lb,ub)
                    u[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]] = u0.reshape(nz,ny)

                else:
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    nz = BCz[ii+1]-BCz[ii]
                    ny = BCy[jj+1]-BCy[jj]
                    Y = Y_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                    Z = Z_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                    YZ = np.concatenate((Y,Z),1)
                    _,u0 = model(YZ,lb,ub)
                    u[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]] = u0.reshape(nz,ny)
        for ii in [0,self.z_nets-1]:
            if ii==0:
                bc_tb=0
            else: 
                bc_tb=1
            model = getattr(self,'model{}_{}'.format(0,bc_tb))
            lb = getattr(self,'lb{}_{}'.format(0,bc_tb))
            ub = getattr(self,'ub{}_{}'.format(0,bc_tb))
            for jj in range(self.y_nets):
                nz = BCz[ii+1]-BCz[ii]
                ny = BCy[jj+1]-BCy[jj]
                Y = Y_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                Z = Z_test[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]].flatten()[:,None]
                YZ = np.concatenate((Y,Z),1)
                _,u0 = model(YZ,lb,ub)
                u[BCz[ii]:BCz[ii+1],BCy[jj]:BCy[jj+1]] = u0.reshape(nz,ny)
        return u

    def predict_cpu(self, Y_test,Z_test,BCy,BCz):
        # Y_test = np.from_numpy(Y_test.astype(self.np_dtype))
        # Z_test = np.from_numpy(Z_test.astype(self.np_dtype))
        u = self.predict(Y_test,Z_test,BCy,BCz)
        return u
        # return u_real().numpy(),u_imag().numpy()

    def predict_H(self, Y_test,Z_test,ii,jj):
        '''
        Y_test: observation position(Y direction)
        Z_test: erath-air interface (Z direction)
        ii,jj: index of model

        output:
        u_f: [ur, ui]
        u_x: [ur_x,ui_x]
        '''
        # BCy = self.BCy.clone()
        # BCz = self.BCz.clone()
        # Y_test = np.from_numpy(Y_test.astype(self.np_dtype)).flatten()[:,None]
        # Z_test = np.from_numpy(Z_test.astype(self.np_dtype)).flatten()[:,None]
        # Y_test.requires_grad=True
        # Z_test.requires_grad=True
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

    # def relative_error(self, u_pred,u_star):
    #     error_u = np.linalg.norm(u_star-u_pred)/np.linalg.norm(u_star)
    #     return error_u#().cpu().numpy()

    def rms_error(self,u_pred,u_star): 
        return np.sqrt(np.sum(np.abs(u_pred-u_star)**2)/len(u_pred))

    def max_error(self,u_pred,u_star):
        return np.max(np.abs(u_pred-u_star))

    def save_param(self, path):
        np.save(self.model.state_dict(),path)

    def load_param(self, path):
        self.model.load_state_dict(np.load(path))

    def print_start(self):
        print("the training start")
        print("---------------------------------------")

    def print_lf_start(self):
        print("---------------------------------------")
        print("the LBFGS training start")

    def summarz(self,model):
        print(model)

    def print_param(self,model): 
        for param in model.parameters(): 
            print(param)

    def time_fd(self,time):
        return datetime.fromtimestamp(time).strftime("%M:%S")

    def freeze_model(self,model): 
        for param in model.parameters():
            param.requires_grad = False

    def get_A(self): 
        return(self.A())

    def pass_func(self,*args): 
        pass 