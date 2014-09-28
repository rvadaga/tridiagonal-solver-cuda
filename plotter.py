# #setting our fonts
# from matplotlib import rc
# rc('font', **{'family':'serif', 'size':'14', 'serif':'serif'})
# rc('axes', **{'grid':'False', 'titlesize':'16', 'labelsize':'16'})
# rc('text', usetex=True)

from matplotlib.pylab import *
close('all')

config = loadtxt('configFile')
size = config[0]
s = config[2]
steps = config[3]
b_dim = config[4]
stride = config[5]

print 'size = %d, steps = %d\n' % (size, steps)
x = loadtxt('output')
figure(0)
plot(x)
xlim(0, size)
title (('Electric Field after %d steps' % steps)) 
L = 20

figure(1)
E = loadtxt('outputField')
E = E[0:, :]
index = where(E[1, :]>0)[0][-1]
E = E[:, :index]
print 'Es shape = %d, %d' % (E.shape[0], E.shape[1])
contourf(linspace(0, 40, index), linspace(0, L, steps), ((E)), cmap='Blues')
xlabel('$x$')
ylabel('$z$')
title('Contour Plot of Electric Field')
colorbar()

show()