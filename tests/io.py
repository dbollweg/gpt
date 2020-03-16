#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")

# Show metadata of field
g.message("Metadata", U[0].metadata)

def plaquette(U):
    # U[mu](x)*U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
    tr=0.0
    vol=float(U[0].grid.gsites)
    for mu in range(4):
        for nu in range(mu):
            tr += g.sum( g.trace(U[mu] * g.cshift( U[nu], mu, 1) * g.adj( g.cshift( U[mu], nu, 1 ) ) * g.adj( U[nu] )) )
    return 2.*tr.real/vol/4./3./3.

# Calculate Plaquette
g.message(g.qcd.gauge.plaquette(U))
g.message(plaquette(U))

# Precision change
Uf = g.convert(U, g.single)
g.message(g.qcd.gauge.plaquette(Uf))

Uf0 = g.convert(U[0], g.single)
g.message(g.norm2(Uf0))

del Uf0
g.meminfo()

# Slice
x=g.sum(Uf[0])

print(x)

sys.exit(0)

# Calculate U^\dag U
u = U[0][0,1,2,3]

v = g.vcolor([0,1,0])

g.message(g.adj(v))
g.message(g.adj(u) * u * v) 


gr=g.grid([2,2,2,2],g.single)
g.message(g.mspincolor(gr)[0,0,0,0] * g.vspincolor(gr)[0,0,0,0])

g.message(g.trace(g.mspincolor(gr)[0,0,0,0]))

# Expression including numpy array
r=g.eval( u*U[0] + U[1]*u )
g.message(g.norm2(r))

# test inner and outer products
v=g.vspincolor([[0,0,0],[0,0,2],[0,0,0],[0,0,0]])
w=g.vspincolor([[0,0,0],[0,0,0],[0,0,0],[1,0,0]])
xx=v * g.adj(w)
g.message(xx[1][3][2][0])
g.message(xx)
g.message(g.adj(v) * v)

g.message(g.transpose(v) * v)

u += g.adj(u)
g.message(u)


v=g.vspincolor([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
l=g.vspincolor(gr)
l[:]=0
l[0,0,0,0]=v

g.message(l)

for mu in [ 0,1,2,3,5]:
    for nu in [0,1,2,3,5]:
        g.message(mu,nu,g.norm2(g.gamma[mu] * g.gamma[nu] * l + g.gamma[nu] * g.gamma[mu] * l)/g.norm2(l))

g.message(l)

m=g.mspincolor(gr)
m[0,0,0,0]=xx
m @= g.gamma[5] * m * g.gamma[5]
g.message(m)
