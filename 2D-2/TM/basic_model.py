"""
"""
def setup_seed(seed):
    np.random.seed(seed)
    # random.seed(seed)
    # np.manual_seed(seed) #cpu
    # np.cuda.manual_seed_all(seed)  #并行gpu
    # np.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    # np.backends.cudnn.benchmark = True

#from np.utils.tensorboard import SummarzWriter
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

from NeuralNet import FNN,input_minmax
# szs.path.append("../utils")
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


class basic_model(object):
    def __init__(self,hp,Z_u,Y_test,Z_test,Beta,u_test,BCy,BCz):
        '''
        Z_u: elements in  boundaries, structural data;
        Y_test: all points in y direction(where there is test data); array
        Z_test: all points in Z direction(where there is test data); array
        Beta: media model parameters; array
        u_test: all E, tuple(real part, imag part)
        Bcy: list including postion of interface in y axis
        Bcz: list including postion of interface in z axis
        '''

        # self.tau_bc = 1.0
        setup_seed(1)
        # print(np.initial_seed())
        net_layers = hp["layers"]
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

        self.error_dict = {
            # "relative": self.relative_error,
            "rms"     : self.rms_error,
            "max"     : self.max_error,
            "none"    : self.pass_func,
        }

        self.model0_0 = FNN(hp["layers"],hp["Rm1"],hp["Rm2"],hp["net_type"])
        # self.weight_init(self.model0_0,hp["init"], hp["Rm1"],hp["Rm2"],hp["net_type"])
        # for ii in range(self.z_nets):
        #     for jj in range(self.y_nets): 
        #         if ii !=0 or jj !=0:
                    # setattr(self,'model{}_{}'.format(ii,jj),copy.deepcopy(self.model0_0))
                    # setattr(self,'model{}_{}'.format(ii,jj),FNN(hp["layers"],hp["Rm1"],hp["Rm2"],hp["net_type"]))

        for ii in range(self.z_nets):
            if ii == self.z_nets-1:
                for jj in range(3):
                    # setattr(self,'model{}_{}'.format(ii,jj),FNN(hp["layers"],hp["Rm1"],hp["Rm2"],hp["net_type"]))
                    setattr(self,'model{}_{}'.format(ii,jj),copy.deepcopy(self.model0_0))

            else:
                for jj in range(1,self.y_nets-1): 
                    # setattr(self,'model{}_{}'.format(ii,jj),FNN(hp["layers"],hp["Rm1"],hp["Rm2"],hp["net_type"]))
                    setattr(self,'model{}_{}'.format(ii,jj),copy.deepcopy(self.model0_0))
                    # self.weight_init(self.model0_0,hp["init"], hp["Rm1"],hp["Rm2"],hp["net_type"])
        # self.to_cuda(hp["cuda"],Z_u,Y_test,Z_test,Beta,u_test,BCy,BCz)
        self.to_model(Z_u,Y_test,Z_test,Beta,u_test,BCy,BCz)

    def train(self, cond,driver,sample_method,er):
        start_time = time.time()
        # self.tr_train(num1)
        # self.lf_train(num2)
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
        num_nodes  = self.num_nodes
        # *2 是因为实虚部。因为内部边界有C1连续性需要两个条件，
        # 而外部边界只有Neumann或Direchlet条件，头尾加起来也刚好两个，
        # 就等于2*num_nets
        MM = (y_nets*z_nets)*(y_points*z_points+2*y_inter+2*z_inter)
        NN = ((y_nets-2)*(z_nets-1)+3)*num_nodes
        A = np.zeros((MM,NN),dtype=self.np_complex)
        b = np.zeros((MM,1), dtype=self.np_complex)
        II  = np.array([1j],dtype=self.np_complex)
        ########################   governing equation ############################################
        # 先求所有model里的governing equation，因为需要lb和ub
        # print("start governing equation")
        # start_time = time.time()
        num_points = y_points*z_points # number of elements in a domain
        for ii in range(z_nets-1): 
            for jj in range(1,y_nets-1): 
                model = getattr(self,'model{}_{}'.format(ii,jj))
                col_pos = ((ii)*(y_nets-2)+jj-1)+3
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
                # YY = self.get_uniform_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                # ZZ = self.get_uniform_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                # YZ = np.concatenate((YY,ZZ),1)
                # noticed that beta are identical in a domain
                Beta = self.Beta[BCz[ii],BCy[jj]]
                # print(np.unique(Beta))
                setattr(self,'lb{}_{}'.format(ii,jj),lb)
                setattr(self,'ub{}_{}'.format(ii,jj),ub)
                u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta,lb,ub)
                u_yy = u_yy0 + u_zz0
                uu   = uu0*(Beta**2)
                # print(u_yy)
                # print(uu)
                # w_m = np.sum(np.abs(u_yy)+np.abs(uu),1,keepdims=True)
                # w_m = np.linalg.norm(np.concatenate((u_yy,uu),1), None,1,True)
                w_m = np.linalg.norm(np.concatenate((u_yy,uu),1), np.inf,1,True)
                A[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
                del u_yy,uu
        
        for jj in [0,y_nets-1]:
            if jj==0:
                bc_lr = 1
            elif jj == y_nets-1:
                bc_lr = 2
            ii = 0
            col_pos = (ii*y_nets+bc_lr)
            lb = np.array([self.Y_test[BCz[ii],BCy[jj]],self.Z_test[BCz[ii],BCy[jj]]])
            ub = np.array([self.Y_test[BCz[z_nets-1],BCy[jj+1]],self.Z_test[BCz[z_nets-1],BCy[jj+1]]])
            if sample_method == "uniform":
                YY = self.get_uniform_data(self.Y_test[BCz[ii]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                ZZ = self.get_uniform_data(self.Z_test[BCz[ii]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                YZ = np.concatenate((YY,ZZ),1)
            elif sample_method == "gaussian":
                YY = self.get_gaussian_data(self.Y_test[BCz[ii]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Y")
                ZZ = self.get_gaussian_data(self.Z_test[BCz[ii]:BCz[z_nets-1]+1,BCy[jj]:BCy[jj+1]+1],y_points,z_points,"Z")
                YZ = np.concatenate((YY,ZZ),1)
            elif sample_method == "lhs":
                YZ = self.get_lhs_data(y_points,z_points,lb,ub)
            Beta = self.Beta[BCz[ii],BCy[jj]]
            model = getattr(self,'model{}_{}'.format(z_nets-1,bc_lr))
            setattr(self,'lb{}_{}'.format(z_nets-1,bc_lr),lb)
            setattr(self,'ub{}_{}'.format(z_nets-1,bc_lr),ub)
            u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta,lb,ub)
            u_yy = u_yy0 + u_zz0
            uu   = uu0*(Beta**2)
            # w_m = np.linalg.norm(np.concatenate((u_yy,uu),1), None,1,True)
            w_m = np.linalg.norm(np.concatenate((u_yy,uu),1), np.inf,1,True)
            A[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
            del u_yy,uu

        
        ii = z_nets-1
        jj = 0
        model = getattr(self,'model{}_{}'.format(ii,jj))
        col_pos = 0
        lb = np.array([self.Y_test[BCz[ii],BCy[jj]],self.Z_test[BCz[ii],BCy[jj]]])
        ub = np.array([self.Y_test[BCz[ii+1],BCy[y_nets]],self.Z_test[BCz[ii+1],BCy[y_nets]]])
        if sample_method == "uniform":
            YY = self.get_uniform_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[y_nets]+1],y_points,z_points,"Y")
            ZZ = self.get_uniform_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[y_nets]+1],y_points,z_points,"Z")
            YZ = np.concatenate((YY,ZZ),1)
        elif sample_method == "gaussian":
            YY = self.get_gaussian_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[y_nets]+1],y_points,z_points,"Y")
            ZZ = self.get_gaussian_data(self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[y_nets]+1],y_points,z_points,"Z")
            YZ = np.concatenate((YY,ZZ),1)
        elif sample_method == "lhs":
            YZ = self.get_lhs_data(y_points,z_points,lb,ub)
        else: 
            raise KeyError("bad sample method, please use gaussian or uniform ")
        # noticed that beta are identical in a domain
        Beta = self.Beta[BCz[ii],BCy[jj]]
        # print(np.unique(Beta))
        # lb = YZ.min(0)
        # ub = YZ.max(0)
        setattr(self,'lb{}_{}'.format(ii,jj),lb)
        setattr(self,'ub{}_{}'.format(ii,jj),ub)
        u_yy0,u_zz0,uu0 = self.net_f(model,YZ,Beta,lb,ub)
        u_yy = u_yy0 + u_zz0
        uu   = uu0*(Beta**2)
        u_y_all = np.concatenate((u_yy,uu),1)
        # w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),None, axis=1,keepdims=True)
        w_m = np.linalg.norm(np.concatenate((u_yy,uu),1),np.inf, axis=1,keepdims=True)
        A[((ii*y_nets+jj))*num_points:((ii*y_nets+jj)+1)*num_points,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = (u_yy+II*uu)/w_m
        del u_yy,uu
        # running_time = time.time()-start_time
        # print("governing equation computation time: {:}".format(self.time_fd(running_time)))
        ################################## END governing equation ##########################################


        ################################ top and bottom ############################################
        # print(" ")
        # print("start top and bottom")
        # start_time = time.time()
        num_base = y_nets*z_nets*num_points # number of rows in governing equation
        for jj in range(1,y_nets-1): 
            for ii in range(z_nets):
                row_pos = 2*(z_nets*jj+ii)
                col_pos = ((ii)*(y_nets-2)+jj-1)+3
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"tb")
                Beta = self.Beta[BCz[ii],BCy[jj]]
                if ii >0 and ii<z_nets-1:
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    u_pred0, u_y00 = self.net_interface(model,Y_inter0,lb,ub,"Z")
                    u_y0 = u_y00/(Beta**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(col_pos-(y_nets-2))*num_nodes:(col_pos-y_nets+3)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                    del u_y0
                elif ii ==z_nets-1:
                    col_pos0 = 0
                    model = getattr(self,'model{}_{}'.format(ii,0))
                    lb = getattr(self,'lb{}_{}'.format(ii,0))
                    ub = getattr(self,'ub{}_{}'.format(ii,0))
                    u_pred0, u_y00 = self.net_interface(model,Y_inter0,lb,ub,"Z")
                    u_y0 = u_y00/(Beta**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(col_pos-(y_nets-2))*num_nodes:(col_pos-y_nets+3)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(col_pos0)*num_nodes:(col_pos0+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(col_pos0)*num_nodes:(col_pos0+1)*num_nodes] = -1.0*u_y0/w_i
                else:
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    u_pred,_ = model(self.Z_t[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                    A[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,:] = self.u_t[jj*y_inter:(jj+1)*y_inter,:]
                
                if ii <z_nets-1:
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    # u_pred1, u_y1 = self.net_interface(model,Y_inter1,lb,ub,"Z")
                    # Y_inter0: top of domain
                    # Y_inter1: bottom of domain
                    u_pred1, u_y11 = self.net_interface(model,Y_inter1,lb,ub,"Z")
                    u_y1 = u_y11/(Beta**2)

                    # w_i = np.sum(np.abs(u_y1),1,keepdims=True)
                    # w_i = np.linalg.norm(np.abs(u_y1),axis=1,keepdims=True)
                    A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1
                    # A[num_base+(row_pos+2)*y_inter:num_base+(row_pos+3)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_y1/w_i
                    # del u_y1

                # Robin  conditions 
                else: # ii=z_nets-1
                    # X_end = self.Z_b[jj*y_inter:(jj+1)*y_inter,0:2]
                    # X_end.requires_grad_()
                    # # 最后一行的beta值
                    # Betaend = self.get_uniform_data(self.Beta[-2:-1,:],1,X_end.size(0),"Z")
                    # u_pred, u_y = self.net_interface(model,X_end,lb,ub,"Z")
                    # A[num_base+(row_pos+2)*y_inter:num_base+(row_pos+3)*y_inter,(col_pos)  *num_nodes:(col_pos+1)*num_nodes] = u_y+np.sqrt(2)/2*Betaend*u_pred
                    # A[num_base+(row_pos+2)*y_inter:num_base+(row_pos+3)*y_inter,(col_pos+1)*num_nodes:(col_pos+2)*num_nodes] = np.sqrt(2)/2*Betaend*u_pred
                    # A[num_base+(row_pos+3)*y_inter:num_base+(row_pos+4)*y_inter,(col_pos)  *num_nodes:(col_pos+1)*num_nodes] = -1.0*(np.sqrt(2)/2*Betaend*u_pred)
                    # A[num_base+(row_pos+3)*y_inter:num_base+(row_pos+4)*y_inter,(col_pos+1)*num_nodes:(col_pos+2)*num_nodes] = u_y+np.sqrt(2)/2*Betaend*u_pred

                # Dirichlet conditons
                    col_pos = 0
                    model = getattr(self,'model{}_{}'.format(ii,0))
                    lb = getattr(self,'lb{}_{}'.format(ii,0))
                    ub = getattr(self,'ub{}_{}'.format(ii,0))
                    u_pred,_ = model(self.Z_b[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                    A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,:] = self.u_b[jj*y_inter:(jj+1)*y_inter,:]
        
        for jj in [0, y_nets-1]:
            if jj==0:
                bc_lr = 1
            elif jj==y_nets-1:
                bc_lr = 2
            for ii in [0,z_nets-2,z_nets-1]:
                row_pos = 2*(z_nets*jj+ii)
                if ii ==0:
                    col_pos = bc_lr
                    model = getattr(self,'model{}_{}'.format(z_nets-1,bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(z_nets-1,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(z_nets-1,bc_lr))
                    u_pred,_ = model(self.Z_t[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                    A[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos)*y_inter:num_base+(row_pos+1)*y_inter,:] = self.u_t[jj*y_inter:(jj+1)*y_inter,:]

                elif ii == z_nets-2:
                    col_pos = bc_lr
                    _,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                        self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"tb")
                    Beta = self.Beta[BCz[ii],BCy[jj]]
                    model = getattr(self,'model{}_{}'.format(z_nets-1,bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(z_nets-1,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(z_nets-1,bc_lr))
                    u_pred1, u_y11 = self.net_interface(model,Y_inter1,lb,ub,"Z")
                    u_y1 = u_y11/(Beta**2)
                    A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1

                elif ii ==z_nets-1:
                    col_pos0 = 0
                    Beta = self.Beta[BCz[ii],BCy[jj]]
                    model = getattr(self,'model{}_{}'.format(ii,0))
                    lb = getattr(self,'lb{}_{}'.format(ii,0))
                    ub = getattr(self,'ub{}_{}'.format(ii,0))
                    u_pred0, u_y00 = self.net_interface(model,Y_inter1,lb,ub,"Z")
                    u_y0 = u_y00/(Beta**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*y_inter:num_base+(row_pos+1)*y_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*y_inter:num_base+(row_pos  )*y_inter,(col_pos0)*num_nodes:(col_pos0+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *y_inter:num_base+(row_pos+1)*y_inter,(col_pos0)*num_nodes:(col_pos0+1)*num_nodes] = -1.0*u_y0/w_i

                    u_pred,_ = model(self.Z_b[jj*y_inter:(jj+1)*y_inter,0:2],lb,ub)
                    A[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,(col_pos0)*num_nodes:(col_pos0+1)*num_nodes] = u_pred
                    b[num_base+(row_pos+1)*y_inter:num_base+(row_pos+2)*y_inter,:] = self.u_b[jj*y_inter:(jj+1)*y_inter,:]

        # running_time = time.time()-start_time
        # print("top and bottom BC computation time: {:}".format(self.time_fd(running_time)))
        ############################# END top and bottom  ################################################

        ################################ left and right ############################################
        # print(" ")
        # print("start left and right")
        # start_time = time.time()
        num_base = y_nets*z_nets*num_points+(y_nets*z_nets*2)*y_inter # number of rows in governing equation
        for ii in range(z_nets-1): 
            for jj in range(y_nets):
                row_pos = (y_nets*ii+jj)*2
                col_pos = ((ii)*(y_nets-2)+jj-1)+3
                # model = getattr(self,'model{}_{}'.format(ii,jj))
                # lb = getattr(self,'lb{}_{}'.format(ii,jj))
                # ub = getattr(self,'ub{}_{}'.format(ii,jj))
                Y_inter0,Y_inter1 = self.get_inter_data(self.Y_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],\
                    self.Z_test[BCz[ii]:BCz[ii+1]+1,BCy[jj]:BCy[jj+1]+1],y_inter,z_inter,"lr")
                Beta = self.Beta[BCz[ii],BCy[jj]]
                
                if jj>1 and jj<y_nets-1:# left
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    u_pred0, u_y00 = self.net_interface(model,Y_inter0,lb,ub,"Y")
                    u_y0 = u_y00/(Beta**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*z_inter:num_base+(row_pos+1)*z_inter,(col_pos-1)*num_nodes:(col_pos)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*z_inter:num_base+(row_pos  )*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                    del u_y0
                elif jj==1:# left
                    bc_lr = 1
                    model = getattr(self,'model{}_{}'.format(ii,jj))
                    lb = getattr(self,'lb{}_{}'.format(ii,jj))
                    ub = getattr(self,'ub{}_{}'.format(ii,jj))
                    u_pred0, u_y00 = self.net_interface(model,Y_inter0,lb,ub,"Y")
                    u_y0 = u_y00/(Beta**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*z_inter:num_base+(row_pos+1)*z_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*z_inter:num_base+(row_pos  )*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = -1.0*u_y0/w_i
                    del u_y0
 
                elif jj==y_nets-1:
                    bc_lr = 2
                    # col_pos = bc_lr
                    model = getattr(self,'model{}_{}'.format(z_nets-1,bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(z_nets-1,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(z_nets-1,bc_lr))
                    u_pred0, u_y00 = self.net_interface(model,Y_inter0,lb,ub,"Y")
                    u_y0 = u_y00/(Beta**2)
                    u_y_all = np.concatenate((u_y0,u_y1),1)
                    w_i = np.linalg.norm(np.abs(u_y_all),np.inf,axis=1,keepdims=True)
                    A[num_base+(row_pos  )*z_inter:num_base+(row_pos+1)*z_inter,(col_pos-1)*num_nodes:(col_pos)*num_nodes] = u_y1/w_i
                    A[num_base+(row_pos-1)*z_inter:num_base+(row_pos  )*z_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = -1.0*u_pred0
                    A[num_base+(row_pos)  *z_inter:num_base+(row_pos+1)*z_inter,(bc_lr)*num_nodes:(bc_lr+1)*num_nodes] = -1.0*u_y0/w_i
                    del u_y0

                else: # jj==0
                    bc_lr = 1
                    col_pos = bc_lr
                    model = getattr(self,'model{}_{}'.format(z_nets-1,bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(z_nets-1,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(z_nets-1,bc_lr))
                    u_pred,_ = model(self.Z_l[ii*z_inter :(ii+1)*z_inter ,0:2],lb,ub)
                    A[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,:] = self.u_l[ii*z_inter :(ii+1)*z_inter ,:]

                    # if Neumann conditions
                    # _, u_y = self.net_interface(model,self.Z_l[ii*z_inter:(ii+1)*z_inter,0:2],lb,ub,"Y")
                    # wi = np.sum(np.abs(u_y),1,True)
                    # A[num_base+(row_pos)*z_inter  :num_base+(row_pos+1)*z_inter,(col_pos)  *num_nodes:(col_pos+1)*num_nodes] = u_y/wi
                    # A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos+1)*num_nodes:(col_pos+2)*num_nodes] = u_y/wi

                if jj>0 and jj< y_nets-1: # right
                    # Y_inter0: top of domain
                    # Y_inter1: bottom of domain
                    u_pred1, u_y11 = self.net_interface(model,Y_inter1,lb,ub,"Y")
                    u_y1 = u_y11/(Beta**2)
                    # w_i = np.sum(np.abs(u_y1),1,keepdims=True)
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1
                    # del u_y1

                elif jj == 0:
                    bc_lr = 1
                    col_pos = bc_lr
                    model = getattr(self,'model{}_{}'.format(z_nets-1,bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(z_nets-1,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(z_nets-1,bc_lr))
                    u_pred1, u_y11 = self.net_interface(model,Y_inter1,lb,ub,"Y")
                    u_y1 = u_y11/(Beta**2)
                    # w_i = np.sum(np.abs(u_y1),1,keepdims=True)
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred1

                else: # jj == y_nets-1:
                    bc_lr = 2
                    col_pos = bc_lr
                    model = getattr(self,'model{}_{}'.format(z_nets-1,bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(z_nets-1,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(z_nets-1,bc_lr))
                    u_pred,_ = model(self.Z_r[ii*z_inter:(ii+1)*z_inter ,0:2],lb,ub)
                    A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
                    b[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,:] = self.u_r[ii*z_inter :(ii+1)*z_inter,:]

        ii = z_nets-1
        jj = 0
        row_pos = (y_nets*ii+jj)*2
        col_pos = 0
        model = getattr(self,'model{}_{}'.format(ii,jj))
        lb = getattr(self,'lb{}_{}'.format(ii,jj))
        ub = getattr(self,'ub{}_{}'.format(ii,jj))
        Beta = self.Beta[BCz[ii],BCy[jj]]
        u_pred,_ = model(self.Z_l[ii*z_inter :(ii+1)*z_inter ,0:2],lb,ub)
        A[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
        b[num_base+(row_pos)*z_inter:num_base+(row_pos+1)*z_inter,:] = self.u_l[ii*z_inter :(ii+1)*z_inter ,:]
        u_pred,_ = model(self.Z_r[ii*z_inter:(ii+1)*z_inter ,0:2],lb,ub)
        A[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,(col_pos)*num_nodes:(col_pos+1)*num_nodes] = u_pred
        b[num_base+(row_pos+1)*z_inter:num_base+(row_pos+2)*z_inter,:] = self.u_r[ii*z_inter :(ii+1)*z_inter,:]
                    # if Neumann conditions
                    # _, u_y = self.net_interface(model,self.Z_r[ii*z_inter:(ii+1)*z_inter,0:2],lb,ub,"Y")
                    # wi = np.sum(np.abs(u_y),1,True)
                    # A[num_base+(row_pos+2)*z_inter:num_base+(row_pos+3)*z_inter,(col_pos)  *num_nodes:(col_pos+1)*num_nodes] = u_y/wi
                    # A[num_base+(row_pos+3)*z_inter:num_base+(row_pos+4)*z_inter,(col_pos+1)*num_nodes:(col_pos+2)*num_nodes] = u_y/wi
        # running_time = time.time()-start_time
        # print("left and right BC computation time: {:}".format(self.time_fd(running_time)))
        ############################################ END left and right  ################################################

        # self.A = A.clone()

        # the first n terms of YY is solution
        # YY,_ = np.lstsq(b,A)

        # print(" ")
        # print("star solving equation")
        start_time = time.time()
        idx = np.argwhere(np.all(np.abs(A[:,...]) == 0, axis=1))
        A2 = np.delete(A, idx, axis=0)
        b2 = np.delete(b, idx, axis=0)
        YY,res,_,_ = lstsq(A2,b2,cond=cond,lapack_driver=driver)
        # YY,res,_,_ = lstsq(A,b,cond=cond, lapack_driver=driver)
        # running_time = time.time()-start_time
        # print("solve time: {:}".format(self.time_fd(running_time)))
        # print("residual= {res:.2e}")
        # print(res)
        # YY = np.from_numpy(YY.astype(self.np_dtype))

        # print(YY)
        # for ii in range(self.z_nets-1):
        #     for jj in range(self.y_nets): 
        #         col_pos = ((ii)*y_nets+jj)
        #         model = getattr(self,'model{}_{}'.format(ii,jj))
        #         model.w1 =YY[(col_pos)*num_nodes:(col_pos+1)*num_nodes,:].T
        # ii = z_nets-1
        # jj = 0
        # col_pos = ((ii)*y_nets+jj)
        # model = getattr(self,'model{}_{}'.format(ii,jj))
        # model.w1 =YY[(col_pos)*num_nodes:(col_pos+1)*num_nodes,:].T

        for ii in range(3):
            model = getattr(self,'model{}_{}'.format(z_nets-1,ii))
            model.w1 =YY[(ii)*num_nodes:(ii+1)*num_nodes,:].T

        for ii in range(z_nets-1): 
            for jj in range(1,y_nets-1):
                col_pos = ((ii)*(y_nets-2)+jj-1)+3
                model = getattr(self,'model{}_{}'.format(ii,jj))
                model.w1 =YY[(col_pos)*num_nodes:(col_pos+1)*num_nodes,:].T

 
    def net_f(self,model,YZ,beta,lb,ub): 
        # YY.requires_grad_()
        # ZZ.requires_grad_()
        # YZ = np.concatenate([YY,ZZ],1)
        u_f,_ = model(YZ,lb,ub)
        w0_0 = model.w0[:,0:1].T
        w0_1 = model.w0[:,1:2].T

        u_YY = (2*(u_f**3) - 2*(u_f))*((2/(ub[0]-lb[0])*w0_0)**2)
        u_ZZ = (2*(u_f**3) - 2*(u_f))*((2/(ub[1]-lb[1])*w0_1)**2)
        u_YY0 = u_YY
        u_ZZ0 = u_ZZ
        u0 = u_f
        # del u_f,u_u,u_Y,u_Z,u_YY,u_ZZ
        return u_YY0,u_ZZ0,u0

    def net_interface(self,model,y,lb,ub,flag):
        u_f,_ = model(y,lb,ub)
        w0_0 = model.w0[:,0:1].T
        w0_1 = model.w0[:,1:2].T
        # u_u  = np.concatenate([self.gradients(u_f[:,ii],y)[0] for ii in range(u_f.size(1))],1)
        if flag == "Y":
            u_Y = (1-u_f**2)*(2/(ub[0]-lb[0])*w0_0)
        #     u_Y  = np.concatenate([u_u[:,2*ii:2*ii+1]   for ii in range(int(u_u.size(1)/2))],1)
        elif flag =="Z":
            u_Y = (1-u_f**2)*(2/(ub[1]-lb[1])*w0_1)
        #     u_Y  = np.concatenate([u_u[:,2*ii+1:2*ii+2] for ii in range(int(u_u.size(1)/2))],1)
        # del u_u
        return u_f,u_Y

    # def net_f(self,model,YZ,beta,lb,ub): 
    #     # YY.requires_grad_()
    #     # ZZ.requires_grad_()
    #     # YZ = np.concatenate([YY,ZZ],1)
    #     u_f,_ = model(YZ,lb,ub)
    #     w0_0 = model.w0[:,0:1].T
    #     w0_1 = model.w0[:,1:2].T
    #     b0   = model.b0.T
    #     w0   = model.w0.T
    #     x0 = input_minmax(YZ,lb,ub)
    #     X = np.dot(x0,w0)+b0
    #     k0 = 2/(ub[0]-lb[0])*w0_0
    #     k1 = 2/(ub[1]-lb[1])*w0_1
    #     u_YY = u_f*(4*k0**2*(X)**2-2*k0**2)
    #     u_ZZ = u_f*(4*k1**2*(X)**2-2*k1**2)
    #     u_YY0 = u_YY
    #     u_ZZ0 = u_ZZ
    #     u0 = u_f
    #     # del u_f,u_u,u_Y,u_Z,u_YY,u_ZZ
    #     return u_YY0,u_ZZ0,u0

    # # return interface conditions
    # def net_interface(self,model,YZ,lb,ub,flag):
    #     u_f,_ = model(YZ,lb,ub)
    #     w0_0 = model.w0[:,0:1].T
    #     w0_1 = model.w0[:,1:2].T
    #     b0   = model.b0.T
    #     w0   = model.w0.T
    #     k0 = 2/(ub[0]-lb[0])*w0_0
    #     k1 = 2/(ub[1]-lb[1])*w0_1
    #     x0 = input_minmax(YZ,lb,ub)
    #     X = np.dot(x0,w0)+b0
    #     # u_u  = np.concatenate([self.gradients(u_f[:,ii],y)[0] for ii in range(u_f.size(1))],1)
    #     if flag == "Y":
    #         u_Y = -2*k0*X*u_f
    #     #     u_Y  = np.concatenate([u_u[:,2*ii:2*ii+1]   for ii in range(int(u_u.size(1)/2))],1)
    #     elif flag =="Z":
    #         u_Y = -2*k1*X*u_f
    #     #     u_Y  = np.concatenate([u_u[:,2*ii+1:2*ii+2] for ii in range(int(u_u.size(1)/2))],1)
    #     # del u_u
    #     return u_f,u_Y

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

    def to_model(self,Z_u,Y_test,Z_test,Beta,u_test,BCy,BCz):
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
        for ii in range(self.z_nets-1):
            for jj in range(self.y_nets):
                if jj==0 or jj==self.y_nets-1:
                    if jj==0:
                        bc_lr=1
                    else: 
                        bc_lr=2
                    model = getattr(self,'model{}_{}'.format(self.z_nets-1, bc_lr))
                    lb = getattr(self,'lb{}_{}'.format(self.z_nets-1,bc_lr))
                    ub = getattr(self,'ub{}_{}'.format(self.z_nets-1,bc_lr))
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
        ii = self.z_nets-1
        bc_tb = 0
        model = getattr(self,'model{}_{}'.format(self.z_nets-1,bc_tb))
        lb = getattr(self,'lb{}_{}'.format(self.z_nets-1,bc_tb))
        ub = getattr(self,'ub{}_{}'.format(self.z_nets-1,bc_tb))
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