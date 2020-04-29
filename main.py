from __future__ import division  
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Inverse Greens function obtained from R-transforms

def Ginv(w,lm,s):
    return w*(lm**2+s/np.sqrt(1+w**2))

#------------------------------------------------------------------------------
# Function finding minima of the cost function
    
def find_minima(Rew,Imw,C):
    # first and second derivatives
    Dx  = 0.5*(np.roll(C,+1,axis=0)-np.roll(C,-1,axis=0))
    Dy  = 0.5*(np.roll(C,+1,axis=1)-np.roll(C,-1,axis=1))
    Dxx = 0.5*(np.roll(Dx,+1,axis=0)-np.roll(Dx,-1,axis=0))
    Dyy = 0.5*(np.roll(Dy,+1,axis=1)-np.roll(Dy,-1,axis=1))
    Dxy = 0.5*(np.roll(Dx,+1,axis=1)-np.roll(Dx,-1,axis=1))
    Dyx = 0.5*(np.roll(Dy,+1,axis=0)-np.roll(Dy,-1,axis=0))
    D = Dxx*Dyy-Dxy*Dyx
    # find minima
    point_is_root = (Dx*np.roll(Dx,1,axis=0)<0)*(Dy*np.roll(Dy,-1,axis=1)<0)*(D>0)*(Dxx>0)
    # cut edges
    point_is_root[:2],point_is_root.T[:2],point_is_root[-2:],point_is_root.T[-2:]=False,False,False,False
    # reordering: separating real and imaginary parts
    re,im = Rew[point_is_root],Imw[point_is_root]
    sol = np.vstack((re,im)).T
    return sol

#------------------------------------------------------------------------------
# Finding roots unsing polynomials (as in the paper)
    
def find_roots_poly(E,lm):
    # setting z at real axis
    z = E
    # polynomial coefficients from Eq.(S.16)
    p = [lm**4,-2*z*lm**2,lm**4+z**2-1,-2*z*lm**2,z**2]
    # root finding
    rts = np.roots(p)
    # reordering: separating real and imaginary parts
    sol = np.vstack((rts.real,rts.imag)).T
    return sol

#------------------------------------------------------------------------------
# Finding roots unsing minimization of cost function
    
def find_roots_optm(E,lm,rmax,imax,ifplot=False,rres=500,ires=500):
    # setting complex plane
    Rew,Imw = np.meshgrid(np.linspace(-rmax,rmax,rres),np.linspace(-imax,imax,ires))
    w = Rew+1j*Imw
    # cost function
    C = np.abs(E-Ginv(w,lm,+1))*np.abs(E-Ginv(w,lm,-1))
    # finding solution
    sol = find_minima(Rew,Imw,C)
    # plotting cost function (optional)
    if ifplot:
        plt.pcolor(Rew,Imw,C,vmin=0,vmax = 0.5)
        plt.colorbar()
        plt.scatter(sol.T[0],sol.T[1])
        plt.title('Cost function, E='+str(E)+", $\lambda$="+str(lm))
        plt.xlabel('Re w')
        plt.ylabel('Im w')
    return sol

#------------------------------------------------------------------------------
# Plotting optimization function (subplot 1)

plt.subplot(1,2,1)
find_roots_optm(E=1.0,lm=0.1,rmax=5,imax=5,ifplot=True,rres=100,ires=100)

#------------------------------------------------------------------------------
# Plotting solutions (subplot 2)

lm=0.1
plt.subplot(1,2,2)
plt.title('Solutions, $\lambda$='+str(lm))
plt.xlabel('E')
plt.ylabel('Im w')

# plotting solutions of polynomial
En = np.linspace(-2,2,1000)
lines = np.zeros([4,len(En)])
for ei in range(len(En)):
    E = En[ei]
    sol = find_roots_poly(E=E,lm=0.1)
    for i in range(4):
        lines[i,ei] = np.sort(sol.T[1])[i]

plt.plot(En,lines[3],zorder=-1,ls='-',c='r',label = 'physical (paper)')
plt.plot(En,lines[0],zorder=-1,ls='--',c='k',label = 'other (paper)')
plt.plot(En,lines[1],zorder=-1,ls='--',c='k')
plt.plot(En,lines[2],zorder=-1,ls='--',c='k')

# plotting solutions of function minimization
En = np.hstack((np.linspace(-2,-1.3,10),np.linspace(-1.3,-0.9,50),np.linspace(-0.9,0.9,10),np.linspace(0.9,1.3,50),np.linspace(1.3,2,10)))
for ei in range(len(En)):
    E = En[ei]
    sol = find_roots_optm(E=E,lm=lm,rmax=10,imax =4)
    plt.scatter(np.repeat(E,len(sol)),sol.T[1],c='k',s=4,zorder=1)
    if ei==0:
        plt.scatter(np.repeat(E,len(sol)),sol.T[1],c='k',s=4,zorder=1,label = "minimization")
    
plt.legend(loc='upper_center')
