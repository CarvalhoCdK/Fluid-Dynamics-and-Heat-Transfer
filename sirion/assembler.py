import numpy as np
from time import perf_counter
from numba import njit, jit

from mesh import Mesh
from interp import NS_x, NS_y
from prime import Prime


# Re = 100, 400 e 1000
# Parâmtros da malha
nx = 30
ny = 30

# Reynolds
Re = 1#00

# Geometria
L = 3
H = 3

# Propriedades
rho = 1
gamma = 1

U = Re*gamma / (rho*L)

TIMER = np.zeros(10)
j = 0

TIMER[j] = perf_counter()
j += 1

umesh = Mesh(nx + 1, ny, L, H)
vmesh = Mesh(nx, ny + 1, L, H)
pmesh = Mesh(nx, ny, L, H)

TIMER[j] = perf_counter()
j += 1
print(f'Time: Meshing : {TIMER[1] - TIMER[0]}')

# Initial Guess
u = np.ones(umesh.elements['number'])*U
v = np.ones(vmesh.elements['number'])*0#U/2
p = np.ones(pmesh.elements['number'])

model = {
    'rho' : 1.0,  # Densidade
    'gamma' : 1.0,
    'a' : L,
    'b' : H,
    'nx' : nx,
    'ny' : ny,
    'deltax' : L/nx,  # Número de elementos em x
    'deltay' : H/ny,   # Número de elementos em y
    'sp' : 0.0
    }


momentum_u = NS_x(model)

#id = 2
#w = umesh.neighbours['W'][id]
#e = umesh.neighbours['E'][id]
#au = momentum_u.internal(id)

momentum_u = NS_x(model)
momentum_v = NS_y(model)

## SET BOUNDARY CONDITIONS
uelements = []
velements = []



# for el in np.arange(vmesh.elements['number']):
#     print(f'element : {el}')
#     w = vmesh.neighbours['W'][el]
#     e = vmesh.neighbours['E'][el]
#     s = vmesh.neighbours['S'][el]
#     n = vmesh.neighbours['N'][el]

#     if s == -1:
#         print('South')
        
#     elif n == -1:
#         print('North')

#     elif w == -1:
#         print('West')

#     elif e == -1:
#         print('East')
            
#     else:
#         print('Internal')

#     print('\n')
        


# for id in np.arange(umesh.elements['number']):
#     print(id)
#     w = umesh.neighbours['W'][id]
#     e = umesh.neighbours['E'][id]
#     au = momentum_u.internal(id, w, e)
#     print(f'u : {au} \n')

#     s = vmesh.neighbours['S'][id]
#     n = vmesh.neighbours['N'][id]
#     av = momentum_v.v_internal(id, s, n)
#     print(f'v : {av} \n')

## Define elemente type (internal, north, west,...) by list
##PRIME
## Guess u,v and p
uequation = NS_x(model)
vequation = NS_y(model)

u = np.ones(umesh.elements['number'])*0
v = np.ones(vmesh.elements['number'])*0
p = np.ones(pmesh.elements['number'])


model['umesh'] = umesh
model['vmesh'] = vmesh
model['pmesh'] = pmesh
model['U'] = U

## Get Ap, Aw, Ae, As, An and B ##################################################
## Solve algebric uh and vh
## u


pv_coupling = Prime(model)

Uh, Apu, Vh, Apv = pv_coupling._get_velocity(u, uequation, v, vequation, p)
west, east, south, north, internal = pv_coupling._map_pressure()

deltax = model['deltax']
deltay = model['deltay']
nx = model['nx']
pv_coupling.solve(u, uequation, v, vequation, p)
#print(c)

#uequation = NS_x(model)
#vequation = NS_y(model)

#pv_coupling.solve(u, uequation, v, vequation)

## Solve system for P

## Correct u and v
