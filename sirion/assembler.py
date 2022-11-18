import numpy as np
from time import perf_counter
from numba import njit, jit

from mesh import Mesh
from interp import NS_x, NS_y
from prime import Prime


# Re = 100, 400 e 1000
# Parâmtros da malha
nx = 3
ny = 3

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

u = np.ones(umesh.elements['number'])*U
v = np.arange(vmesh.elements['number'])*0
p = np.ones(pmesh.elements['number'])


model['umesh'] = umesh
model['vmesh'] = vmesh
model['pmesh'] = pmesh
model['U'] = U

## Get Ap, Aw, Ae, As, An and B ##################################################
## Solve algebric uh and vh
## u
start_solveu = perf_counter()

apu = np.zeros(umesh.elements['number'])
uh = np.zeros(umesh.elements['number'])

for el in np.arange(umesh.elements['number']):
    #print(f'element : {el}')
    w = int(umesh.neighbours['W'][el])
    e = int(umesh.neighbours['E'][el])
    s = int(umesh.neighbours['S'][el])
    n = int(umesh.neighbours['N'][el])

    if w == -1:
        #print('West')
        a = momentum_u.boundary(el, 'W', 0, u, v, p)        

    elif e == -1:
        #print('East')
        a = momentum_u.boundary(el, 'E', 0, u, v, p)        

    elif s == -1:
        #print('South')
        a = momentum_u.boundary(el, 'S', 0, u, v, p)        

    elif n == -1:
        #print('North')
        a = momentum_u.boundary(el, 'N', U, u, v, p)
    
    else:
        #print('Internal')
        a = momentum_u.internal(el, u, v, p)
        
        
    vnb = np.array([u[w], u[e], u[s], u[n]])
    uh[el] = (np.dot(a[1:5], vnb) + a[-1]) / a[0]
    apu[el] = a[0]

        # print(f'A : {a}')
        # print(f'Ass : {a[1:5]}')
        # print(f'vnb : {vnb}')
        # print(f'uh : {uh}')

end_solveu = perf_counter()
print(f'Time to solve uh : {end_solveu-start_solveu}')
## v
apv = np.zeros(vmesh.elements['number'])
vh = np.zeros(vmesh.elements['number'])

for el in np.arange(vmesh.elements['number']):
    #print(f'element : {el}')
    w = int(vmesh.neighbours['W'][el])
    e = int(vmesh.neighbours['E'][el])
    s = int(vmesh.neighbours['S'][el])
    n = int(vmesh.neighbours['N'][el])

    if s == -1:
        #print('South')
        a = momentum_v.boundary(el, 'S', 0, u, v, p)        

    elif n == -1:
        #print('North')
        a = momentum_v.boundary(el, 'N', 0, u, v, p)

    elif w == -1:
        #print('West')
        a = momentum_v.boundary(el, 'W', 0, u, v, p)        

    elif e == -1:
        #print('East')
        a = momentum_v.boundary(el, 'E', 0, u, v, p)        
    
    else:
        #print('Internal')
        a = momentum_v.internal(el, u, v, p)
        
    vnb = np.array([v[w], v[e], v[s], v[n]])
    vh[el] = (np.dot(a[1:5], vnb) + a[-1]) / a[0]
    apv[el] = a[0]




pv_coupling = Prime(model)

Uh, Apu, Vh, Apv = pv_coupling._get_velocity(u, uequation, v, vequation, p)
west, east, south, north, internal = pv_coupling._map_pressure()

deltax = model['deltax']
deltay = model['deltay']
nx = model['nx']
c = pv_coupling.solve_pressure(uh, Apu, vh, Apv, p, nx, deltax, deltay, west, east, south, north, internal)
print(c)

#uequation = NS_x(model)
#vequation = NS_y(model)

#pv_coupling.solve(u, uequation, v, vequation)

## Solve system for P

## Correct u and v
