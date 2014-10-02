# #setting our fonts
from matplotlib import rc
rc('font', **{'family':'serif', 'size':'14', 'serif':'serif'})
rc('axes', **{'grid':'False', 'titlesize':'16', 'labelsize':'16'})
rc('text', usetex=True)

from matplotlib.pylab import *
close('all')

config = loadtxt('configFile')
size = config[0]
s = config[2]
steps = config[3]
b_dim = config[4]
stride = config[5]

print 'size = %d, steps = %d\n' % (size, steps)
E = loadtxt('outputField')
E = E[0:, :]
index = where(E[1, :]>0)[0][-1]
E = E[:, :index]
print 'Es shape = %d, %d' % (E.shape[0], E.shape[1])

figure(0)
plot(E[0, :])
plot(E[-1, :])
xlim(0, index)
title (('Electric Field' % steps))
legend(('Source Electric Field', 'Final Electric Field'))
L = 20

figure(1)
# contourf(linspace(0, 40, index), linspace(0, L, steps), ((E)))
contourf(linspace(0, 40, index), linspace(0, L, steps), ((E)), cmap=cm.Blues)
xlabel('$x$')
ylabel('$z$')
title('Contour Plot of Electric Field')
colorbar()

show()