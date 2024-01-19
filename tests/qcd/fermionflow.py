import gpt as g


grid = g.grid([8,8,8,8], g.double)

rng = g.random('test')

U = g.qcd.gauge.random(grid,rng)
chi = g.vspincolor(grid)
chi_flowed = g.vspincolor(grid)
g.message(type(chi_flowed))
rng.cnormal(chi)

plaq = g.qcd.gauge.stencil.plaquette(U)

g.message("Initial plaquette: ", plaq)
g.message(f"Initial chi norm: {g.norm2(chi)}")
epsilon = 0.1
Nsteps = 20
g.message("Starting zeuthen flow with fixed stepsize eps = ", epsilon, " and Nsteps = ", Nsteps)

U_flowed,chi_flowed = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U,chi, epsilon, Nsteps)

plaq = g.qcd.gauge.stencil.plaquette(U_flowed)

g.message(f"Plaquette after {Nsteps} steps: {plaq}")
chi_color = g.vcolor(grid)
g.message(f"Chi_flowed norm after {Nsteps} steps: {g.norm2(chi_flowed)}")
g.message(type(chi_flowed))
assert(abs(plaq-1.0) < 1e-5)
g.message("Test passed!")