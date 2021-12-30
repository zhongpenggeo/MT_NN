'''
use COMMEMI 2D-0 model to test the code
'''
from TM_model import *
import matplotlib.pyplot as plt
import scipy.io as scio

# build media
def generate_media(plot_flag="True"):
    '''
    generate the media, like COMMEMI 2D-2 model
    '''
    ny = 100
    nz = 100
    y_0 = np.array([-2e5,-22e3,-4e3,10e3,22e3,2e5])
    z_0 = np.array([-2e5,0,7e3,9e3,75e3,2e5])
    y1 = np.linspace(y_0[0],y_0[1],ny+1)[:-1]
    y2 = np.linspace(y_0[1],y_0[2],ny+1)[:-1]
    y3 = np.linspace(y_0[2],y_0[3],ny+1)[:-1]
    y4 = np.linspace(y_0[3],y_0[4],ny+1)[:-1]
    y5 = np.linspace(y_0[4],y_0[5],ny)
    z1 = np.linspace(z_0[0],z_0[1],nz+1)[:-1]
    z2 = np.linspace(z_0[1],z_0[2],nz+1)[:-1]
    z3 = np.linspace(z_0[2],z_0[3],nz+1)[:-1]
    z4 = np.linspace(z_0[3],z_0[4],nz+1)[:-1]
    z5 = np.linspace(z_0[4],z_0[5],nz)
    y  = np.concatenate((y1,y2,y3,y4,y5))
    z  = np.concatenate((z2,z3,z4,z5))
    Y,Z= np.meshgrid(y,z)
    BCy = np.array([0,ny,2*ny,3*ny,4*ny,5*ny])
    BCz = np.array([0,nz,2*nz,3*nz,4*nz])
    sigma = np.array([[1e-2,1e-2,1e-2,1e-2,1e-2],
                      [1e-2,1e1, 1e-2,1e1, 1e-2],
                      [1e-2,1e-2,1e-2,1e-2,1e-2],
                      [1e-1,1e-1,1e-1,1e-1,1e-1]])
    Sigma = np.ones_like(Y)
    for ii in range(len(BCy)-1):
        for jj in range(len(BCz)-1):
            Sigma[BCz[jj]:BCz[jj+1],BCy[ii]:BCy[ii+1]]=sigma[jj,ii]
    BCy0 = np.array([0,ny,2*ny,3*ny,4*ny,5*ny-1])
    BCz0 = np.array([0,nz,2*nz,3*nz,4*nz-1])
    sigL = sigma[:,0]
    hL   = np.array([z[BCz0[i+1]]-z[BCz0[i]] for i in range(len(BCz0)-1)])
    sigR = sigma[:,-1]
    hR   = hL
    if plot_flag=="True":
        fig = plt.figure(figsize=(10,4))
        ax = plt.subplot(1,2,1)
        h = ax.pcolormesh(Y,Z, np.log10(Sigma), 
                          cmap='jet',shading='auto')
        ax.invert_yaxis()
        fig.colorbar(h)
        plt.show()
    return Y,Z,ny,nz,Sigma,sigL,hL,sigR,hR

# parameters
hp={}
hp["net_type"] = "complex" # networkd type: complex or real
hp["layers"] = [2, 100,1]
hp["N_y"]  = 10#hp["layers"][-2]
hp["N_z"]  = 10#hp["layers"][-2]
hp["N_yi"] = 10#hp["N_y"] # number of points on the interface.
hp["N_zi"] = 10#hp["N_z"]
hp["L2"] = 0
hp["activation"] = ["Tanh"]*1
# initialzation method: uniform, norml 
hp["init"] = "uniform"
hp["Rm1"] = -1.0 # the range of the uniform distribution
hp["Rm2"] = 1.0 #  the range of the uniform distribution

cond = None # Cutoff for ‘small’ singular values; used to determine effective rank of A
lapack_driver = "gelsd" # Which LAPACK driver is used. Options are 'gelsd', 'gelsy', 'gelss'
sample_method = "uniform" # sampling method: uniform, normal
freq = 1.0/300
# import results from FEM
data_FEM = scio.loadmat("../Data/CM2D2_FEM.mat")
obs_FEM = data_FEM["obs"].flatten()/1e3
rho_FEM_TM= data_FEM["rho_TM"].flatten()
phs_FEM_TM= data_FEM["phs_TM"].flatten()
H_c = np.loadtxt("../Data/CM2D2_10s.dat")
obs_c = H_c[:,2]
rho_TM = H_c[:,3]

Y,Z,ny,nz,Sigma,sigL,hL,sigR,hR = generate_media(plot_flag="True")
# calculate TE mode
BCy = np.array([0,int(0.85*ny),ny,int(1.5*nz),2*ny,3*ny,4*ny,int(4.15*ny),5*ny])
BCz = np.array([0,int(0.5*nz),nz,2*nz,3*nz,4*nz])
Z_u = BC_struct(Z,Y,hp["N_zi"],hp["N_yi"],BCz,BCy,freq,hL,sigL,hR,sigR)
BCy[-1] = BCy[-1]-1
BCz[-1] = BCz[-1]-1
pinn = TM_class(hp, Z_u,Y,Z,Sigma,freq,BCy,BCz)
# 'gelsd', 'gelsy', 'gelss'
pinn.train(cond,lapack_driver,sample_method)
u_pred = pinn.predict_cpu(Y,Z,BCy,BCz)

Y_obs = np.linspace(-4e4,4e4,321)
Zxy,rho_a,phs_a = pinn.cal_H(Y_obs,Y[0,:],BCy)

fig = plt.figure(figsize=(14,10))
ax = plt.subplot(2,1,1)
ax.plot(obs_FEM, rho_FEM_TM,'bp-',label="FEM")
ax.plot(Y_obs/1e3, rho_a,'rp-',label="prediction")
ax.plot(obs_c, rho_TM,'gD', markersize=10,label="COMMEMI")
ax.set_yscale("log")
ax.set_ylabel(r'apparent resistivity($\Omega\cdot$m)')
ax.legend()

ax = plt.subplot(2,1,2)
ax.plot(obs_FEM, phs_FEM_TM,'bp-',label="FEM")
ax.plot(Y_obs/1e3, phs_a,'rp-',label="prediction")
ax.set_ylabel('phase(degree)')
ax.set_xlabel('distance(km)')

# plt.savefig("./imag/tanh.jpg",dpi=300,bbox_inches='tight',pad_inches=0.05)
plt.show()

