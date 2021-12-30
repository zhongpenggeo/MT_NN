'''
use COMMEMI 2D-0 model to test the code
'''
from TE_model import *
import matplotlib.pyplot as plt
import scipy.io as scio

# build media
def generate_media(plot_flag="True"):
    '''
    generate the media, like COMMEMI 2D-0 model

    plot_flag: plot the media model or not.
    '''
    ny = 100
    nz = 100
    y1 = np.linspace(-2e5,-1e4,ny+1)[:-1]
    y2 = np.linspace(-1e4,1e4,ny+1)[:-1]
    y3 = np.linspace(1e4,2e5,ny)
    z1 = np.linspace(-2e5,0,nz+1)[:-1]
    z2 = np.linspace(0, 5e4,nz+1)[:-1]
    z3 = np.linspace(5e4,2e5,nz)
    y  = np.concatenate((y1,y2,y3))
    z  = np.concatenate((z1,z2,z3))
    Y,Z= np.meshgrid(y,z)
    BCy = [0,ny,2*ny,3*ny]
    BCz = [0,nz,2*nz,3*nz]
    sigma = np.array([[1e-9,1e-9,1e-9],[1e-1,1,5e-1],[1e1,1e1,1e1]])
    Sigma = np.ones_like(Y)
    for ii in range(len(BCy)-1):
        for jj in range(len(BCz)-1):
            Sigma[BCz[jj]:BCz[jj+1],BCy[ii]:BCy[ii+1]]=sigma[jj,ii]
    BCy0 = [0,ny,2*ny,3*ny-1]
    BCz0 = [0,nz,2*nz,3*nz-1]
    # conductivity of the left
    sigL = sigma[:,0]
    # thickness of the left
    hL   = np.array([z[BCz0[i+1]]-z[BCz0[i]] for i in range(len(BCz0)-1)])
    # conductivity of the right
    sigR = sigma[:,-1]
    # conductivity of the right
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
hp["layers"] = [2, 100,1] # neurons in each layer of the network
hp["N_y"]  = 10 # number of points for governing equation of y direction.
hp["N_z"]  = 10 # number of points for governing equation of z direction.
hp["N_yi"] = 10 # number of points on the interfaceu in y direction.
hp["N_zi"] = 10 # number of points on the interfaceu in y direction.
hp["activation"] = ["Tanh"] # activation function
# initialzation method: uniform, norml 
hp["init"] = "uniform"
hp["Rm1"] = -1.0 # the range of the uniform distribution
hp["Rm2"] = 1.0 #  the range of the uniform distribution

cond = None # Cutoff for ‘small’ singular values; used to determine effective rank of A
lapack_driver = "gelsd" # Which LAPACK driver is used. Options are 'gelsd', 'gelsy', 'gelss'
sample_method = "uniform" # sampling method: uniform, normal
freq = 1.0/300

# import results from FEM
data_FEM = scio.loadmat("../Data/CM2D0_FEM.mat")
obs_FEM = data_FEM["obs"].flatten()/1e3
rho_FEM_TE= data_FEM["rho_TE"].flatten()
phs_FEM_TE= data_FEM["phs_TE"].flatten()
H_c = np.loadtxt("../Data/CM2D0_300s.dat")
obs_c = H_c[:,0]
rho_TE = H_c[:,1]

# generate media
Y,Z,ny,nz,Sigma,sigL,hL,sigR,hR = generate_media(plot_flag="True")
# partition the media in y and z direction
BCy = np.array([0,int(0.8*ny),ny,2*ny,int(2.1*ny),3*ny])
BCz = np.array([0,nz,int(1.2*nz),2*nz,3*nz])

Z_u = BC_struct(Z,Y,hp["N_zi"],hp["N_yi"],BCz,BCy,freq,hL,sigL,hR,sigR)
BCy[-1] = BCy[-1]-1
BCz[-1] = BCz[-1]-1
pinn = TE_class(hp, Z_u,Y,Z,Sigma,freq,BCy,BCz)
# 'gelsd', 'gelsy', 'gelss'
pinn.train(cond,lapack_driver,sample_method)
u_pred = pinn.predict_cpu(Y,Z,BCy,BCz)

Y_obs = np.linspace(-4e4,4e4,321)
Zxy,rho_a,phs_a = pinn.cal_H(Y_obs,Y[0,BCy])

fig = plt.figure(figsize=(14,10))
ax = plt.subplot(2,1,1)
ax.plot(obs_FEM, rho_FEM_TE,'bp-',label="FEM")
ax.plot(Y_obs/1e3, rho_a,'rp-',label="prediction")
ax.plot(obs_c, rho_TE,'gD', markersize=10,label="COMMEMI")
ax.set_yscale("log")
ax.set_ylabel(r'apparent resistivity($\Omega\cdot$m)')
ax.legend()

ax = plt.subplot(2,1,2)
ax.plot(obs_FEM, phs_FEM_TE,'bp-',label="FEM")
ax.plot(Y_obs/1e3, phs_a,'rp-',label="prediction")
ax.set_ylabel('phase(degree)')
ax.set_xlabel('distance(km)')

# plt.savefig("./imag/tanh.jpg",dpi=300,bbox_inches='tight',pad_inches=0.05)
plt.show()

