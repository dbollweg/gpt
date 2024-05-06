import gpt as g 
import os
import sys
import numpy as np
import math


U = g.qcd.gauge.unit(g.grid([64,64,64,16], g.double))
grid = U[0].grid
print('Here1')
g.mem_report(details=False)

srcD = g.mspincolor(grid)
g.mem_report(details=False)
print('Here2')
rng = g.random("test")

rng.cnormal(srcD)        
psrc = g.create.point(srcD, [0,0,0,0])
 
prop_bw_seq = [psrc]
 
prop_f = [psrc]
one = g.identity(g.complex(grid))
 
pp = [2 * np.pi * np.array([0,0,0,0]) / grid.fdimensions] # plist is the q

P = g.exp_ixp(pp, [0,0,0,0])
mom = [g.eval(pp*one) for pp in P]
 
g.message("Starting slice_trDA")
corr = g.slice_trDA(prop_bw_seq, prop_f, mom,3)
g.message("slice_trDA finished")




