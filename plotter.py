# #setting our fonts
# from matplotlib import rc
# rc('font', **{'family':'serif', 'size':'14', 'serif':'serif'})
# rc('axes', **{'grid':'False', 'titlesize':'16', 'labelsize':'16'})
# rc('text', usetex=True)

from matplotlib.pylab import *

close('all')
size=int(sys.argv[1])
steps=int(sys.argv[2])

print 'size = %d, steps = %d\n' % (size, steps)
x = loadtxt('output')
plot(x)
title (('Electric Field after %d steps' % steps)) 
show()