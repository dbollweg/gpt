#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Grid, Lattice, Simple Manipulations
#
import gpt as g

grid=g.grid([ 4,4,4,4 ], g.single)

src=g.complex(grid)

src[:]=0
src[0,0,0,0]=1

print(src)


