#/usr/bin/env/python
#importing libraries
from scipy import *
from cmath import *
from matplotlib.pylab import *
from scipy.sparse import *
from scipy.linalg import *
from scipy.weave import *
from scipy.weave.converters import blitz
import time as t

close('all')

def c_TDMA_Solver(a, b, c, d):
    x = zeros(len(d),dtype='complex')
    gamma = zeros(len(d),dtype='complex')
    n = len(d)
    beta = b[0]
    code = """
        x(0) = d(0)/beta;
        int i;
        for (i=1; i<n; i++){
            gamma(i) = c(i-1)/beta;
            beta = b(i)-a(i)*gamma(i);
            x(i) = (d(i)-a(i)*x(i-1))/beta;
        }
        int k;
        for (i=1; i<n; i++){
            k = n-i;
            x(k-1) = x(k-1)-gamma(k)*x(k);
        }
        """
    inline(code, ['a', 'b', 'c', 'd', 'x', 'gamma', 'n', 'beta'], type_converters=blitz, compiler='gcc', verbose=1)
    return x

ncore   = 1.5                         # refractive index of core
nclad   = 1.48                        # refractive index of cladding
nref    = 1.48
a       = 2                           # half length of core in um
lam     = 1.55                        # wavelength in microns
k0      = 2*pi/lam
beta    = k0*nref
eta0    = 120*pi

dz  = 0.55                         # in microns
L   = 100
Nz  = int(L/dz)                    # no. of points in z-direction
Nx  = 2048                         # no. of points to keep in x-direction
Nskip   = 1
Npts    = int(Nz/Nskip)
z   = linspace(0, L, Npts)
Lx  = 20

x   = linspace(-Lx, Lx, Nx+1)
x   = x[0:-1]
dx  = x[1]-x[0]
kx  = 2*pi*linspace(-0.5/dx, 0.5/dx, Nx+1)
kx  = kx[0:-1]
kz  = 2*pi*linspace(-0.5/dz, 0.5/dz, Npts)

#refractive index profile
n = ncore*ones(len(x))
n[where(x<-a)[0]]  = nclad
n[where(x>a)[0]]   = nclad

epsr = n*n

EX = zeros((Npts,Nx),dtype='complex')
Ex = zeros(Nx,dtype='complex')
power = zeros(Npts)
p = zeros(Npts,dtype='complex')
mxm = zeros(Npts,dtype='complex')

# source signal
x0  = 0
sig = 4
Ex  = 1*exp(-(x-x0)**2/sig**2).astype(complex)#*exp(1j*k0*x*sin(5.7*pi/180))#1*exp(-1j*k0*nclad*sin(5.7*pi/180)*x)#
EX[0,:] = Ex
mxm[0] = (abs(Ex).max())
power[0] = trapz(EX[0,:]**2,x)
p[0] = trapz(conjugate(EX[0,:])*EX[0,:],x)

figure(1)
plot(x, abs(EX[0,:]))
xlabel('$x$')
ylabel('Electric Field Magnitude')
title('Source Electric Field')

alphaw = zeros(Nx-1)
alphae = zeros(Nx-1)
alphax = zeros(Nx-2)

# TE coefficients
alphaw = 1/dx**2*ones(Nx-1)                               # for p=0 to N-2 : 1/(dx*dx)
alphae = 1/dx**2*ones(Nx-1)                               # for p=1 to N-1 : 1/(dx*dx)
alphax = -4/dx**2+alphaw[1:]+alphae[:-1]                  # for p=1 to N-2 : -2/(dx*dx)

B = zeros(Nx-2, dtype='complex')
D = zeros(Nx-2, dtype='complex')
d = zeros(Nx-2, dtype='complex')

gamr = zeros(Npts, dtype='complex')
gaml = zeros(Npts, dtype='complex')
kxbr = zeros(Npts, dtype='complex')
kxbl = zeros(Npts, dtype='complex')

gamr[0]=Ex[-2]/Ex[-3]   # last but 2/ last but 3
gaml[0]=Ex[1]/Ex[2]     # 2nd/ 3rd

B  = -alphax + 4*1j*beta/dz - k0**2*(epsr[1:-1]-nref**2)
b0 = B[0]
b1 = B[-1]
d  = alphax + 4*1j*beta/dz + k0**2*(epsr[1:-1]-nref**2)
# d refers to the rhs

t1 = t.time()
k  = 0
for i in range(1, Npts):
    B[0]  = b0 - gaml[i-1]*alphaw[0]
    B[-1] = b1 - gamr[i-1]*alphae[-1]
    D[1:-1] = alphaw[1:-2]*Ex[1:-3] + d[1:-1]*Ex[2:-2] + alphae[2:-1]*Ex[3:-1]
    D[0] = (d[0] + alphaw[0]*gaml[i-1])*Ex[1] + alphae[1]*Ex[2]
    D[-1] = alphaw[-2]*Ex[-3] + (d[-1]+gamr[i-1]*alphae[-1])*Ex[-2]
    
    Ex[1:-1] = c_TDMA_Solver(-alphaw[:-1], B, -alphae[1:], D)
    
    # Simple TBC    
    gamr[i]=Ex[-2]/Ex[-3]
    kxbr[i]=-1/(1j*dx)*log(gamr[i])
    if(real(kxbr[i])<0):
        kxbr[i]=0+1j*imag(kxbr[i])
        gamr[i]=exp(-1j*kxbr[i]*dx)
    gaml[i]=Ex[1]/Ex[2]
    kxbl[i]=-1/(1j*dx)*log(gaml[i])
    if(real(kxbl[i])<0):
        kxbl[i]=0+1j*imag(kxbl[i])
        gaml[i]=exp(-1j*kxbl[i]*dx)
        
    Ex[0]  = Ex[1]*gaml[i]
    Ex[-1] = Ex[-2]*gamr[i]
            
    if i%Nskip==0:
        k = k+1
        EX[k,:] = Ex
        mxm[k] = (abs(Ex).max())
        power[k] = trapz(EX[k,:]*conjugate(EX[k,:]),x)
        p[k] = trapz(conjugate(EX[0,:])*EX[k,:],x)
        
t2 = t.time()

print 'Time taken = %f' % (t2-t1)

P = 1.0/Npts*fftshift(fft(p))
neff_sim = nref-kz[where(abs(P)==abs(P).max())]/k0

print abs(P.max())
print 'The simulated effective index is %f' % neff_sim

figure(2)
contourf(x, linspace(0, L, Npts),((EX)), cmap=cm.Blues)
xlabel('$x$')
ylabel('$z$')
title('Contour Plot of Electric Field')
colorbar()

figure(3)
plot(x, abs(EX[-1,:]))
xlabel('$x$')
title('Field Amplitude at z = %d [mm]'%(L/1000))

figure(4)
plot(linspace(0, L, Npts)/1000, power)
xlabel('z [mm]')
ylabel('Power')

figure(5)
semilogy(kz, abs(P),'r')
xlabel('$k_z$')

show()