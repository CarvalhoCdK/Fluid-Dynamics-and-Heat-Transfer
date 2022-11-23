import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import perf_counter
from numba import njit, jit

from mesh import Mesh
from interp import NS_x, NS_y
from prime import Prime


# Re = 100, 400 e 1000


# Parâmtros da malha
nx = 80
ny = 80

tol = 1e-5

# Reynolds
Re = 400

# Geometria
L = 1
H = 1

# Propriedades
rho = 1
gamma = 1
U = Re*gamma / (rho*L)

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


op = perf_counter()
print('Meshing \n ...')

umesh = Mesh(nx + 1, ny+2, L, H)
vmesh = Mesh(nx+2, ny + 1, L, H)
pmesh = Mesh(nx, ny, L, H)

model['U'] = U    # Boundary condition fo u at north border
model['umesh'] = umesh
model['unx'] = nx + 1

model['vmesh'] = vmesh
model['vnx'] = nx + 2

model['pmesh'] = pmesh

ed = perf_counter()
print(f'    time[s] : {ed - op}')

# Initial Guess
u = np.ones(umesh.elements['number'])*0#U
v = np.ones(vmesh.elements['number'])*0#U/2
p = np.ones(pmesh.elements['number'])

# Build discretization for u and v
uequation = NS_x(model)
vequation = NS_y(model)

# Solve p-v
pv_coupling = Prime(model)

pressure, u, v, uunknow, vunknown = pv_coupling.solve(u, uequation, v, vequation, p)

## SCALE
u = u / Re
v = v / Re
# umesh.plot()
# vmesh.plot()

## Pick velocity at cell center
n = pmesh.elements['number']
um = np.zeros(n)
vm = np.zeros(n)

for el in np.arange(n):
    
    line = el // nx

    w = el + line + nx + 1
    e = w + 1
    s = el + (2*line + 1)
    n = s + nx + 2

    um[el] = (u[w] + u[e]) / 2
    vm[el] = (v[n] + v[s]) / 2






## PLOT PRESSURE
x = pmesh.elements['x']
y = pmesh.elements['y']

x = x.reshape((nx, ny))
y = y.reshape((nx, ny))
pressure = pressure.reshape((nx, ny))

p_min,p_max = np.abs(pressure).min(), np.abs(pressure).max()

fig, ax = plt.subplots()

color = ax.pcolormesh(x, y, pressure, cmap='coolwarm',  vmin=p_min, vmax=p_max)
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax, label='P')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')

## PLOT u
u = u[uunknow]
x = umesh.elements['x'][uunknow]
y = umesh.elements['y'][uunknow]

x = x.reshape((nx, ny-1))#((nx+2, ny+1))
y = y.reshape((nx, ny-1))#((nx+2, ny+1))
u = u.reshape((nx, ny-1))#((nx+2, ny+1))

mid = nx // 2
yyU = y[:, mid]

p_min,p_max = u.min(), u.max()

fig, ax = plt.subplots(1, 2, figsize=(12,4))
color = ax[0].pcolormesh(x, y, u, cmap='coolwarm',  vmin=p_min, vmax=p_max,# shading='gouraud')#,
                         edgecolors='k', linewidths=0)
#ax[0].axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax[0], label='u')
ax[0].set_xlabel('x', fontsize=14)  
ax[0].set_ylabel('y', fontsize=14)
ax[0].set_aspect('equal')

#fig.show()
#plt.show()

## PLOT v
v = v[vunknown]
x = vmesh.elements['x'][vunknown]
y = vmesh.elements['y'][vunknown]

x = x.reshape((nx-1, ny))#((nx+1, ny+2))
y = y.reshape((nx-1, ny))#((nx+1, ny+2))
v = v.reshape((nx-1, ny))#((nx+1, ny+2))

xxV = x[mid, :]

p_min,p_max = v.min(), v.max()

color = ax[1].pcolormesh(x, y, v, cmap='coolwarm',  vmin=p_min, vmax=p_max,# shading='gouraud')
                         edgecolors='k', linewidths=0)
#ax[1].axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax[1], label='v')
ax[1].set_xlabel('x', fontsize=14)  
ax[1].set_ylabel('y', fontsize=14)
ax[1].set_aspect('equal')
#fig.show()


# ghiau = np.array([[0.00E+00,0.00E+00],
# [5.47E-02,-3.72E-02],
# [6.25E-02,-4.19E-02],
# [7.03E-02,-4.78E-02],
# [1.02E-01,-6.43E-02],
# [1.72E-01,-1.02E-01],
# [2.81E-01,-1.57E-01],
# [4.53E-01,-2.11E-01],
# [5.00E-01,-2.06E-01],
# [6.17E-01,-1.36E-01],
# [7.34E-01,3.32E-03],
# [8.52E-01,2.32E-01],
# [9.53E-01,6.87E-01],
# [9.61E-01,7.37E-01],
# [9.69E-01,7.89E-01],
# [9.77E-01,8.41E-01],
# [1.00E+00,1.00E+00]])

# ghiav = np.array([[0.00E+00,	0.00E+00],
# [6.25E-02,	9.23E-02],
# [7.03E-02,	1.01E-01],
# [7.81E-02,	1.09E-01],
# [9.38E-02,	1.23E-01],
# [1.56E-01,	1.61E-01],
# [2.27E-01,	1.75E-01],
# [2.34E-01,	1.75E-01],
# [5.00E-01,	5.45E-02],
# [8.05E-01,	-2.45E-01],
# [8.59E-01,	-2.24E-01],
# [9.06E-01,	-1.69E-01],
# [9.53E-01,	-1.03E-01],
# [9.53E-01,	-8.86E-02],
# [9.61E-01,	-7.39E-02],
# [9.69E-01,	-5.91E-02],
# [1.00E+00,	0.00E+00]])
#ghiau = np.flip(ghiau, axis=1)
ghia400 = np.genfromtxt('experimental_data/Re400.csv', delimiter=',')
ghiau = ghia400[:,:2]
ghiav = ghia400[:,2:]


## u and v at horizontal and vertical center lines respectively
x = pmesh.elements['x']
y = pmesh.elements['y']

x = x.reshape((nx, ny))
y = y.reshape((nx, ny))
pressure = pressure.reshape((nx, ny))
um = um.reshape((nx, ny))
vm = vm.reshape((nx, ny))

midx = nx // 2
midy = ny // 2

uu = uu = um[:, mid]
yyU = y[:, mid]

vv = vm[mid,:]
xxV = x[mid, :]

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(uu, yyU, color='k')
ax[0].plot(ghiau[:,0], ghiau[:,1], 'x', mew=2, markersize=7)
#ax[0].plot([-0.4, 1.0], [0.5, 0.5], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
ax[0].set_xlabel('u/U', fontsize=14)  
ax[0].set_ylabel('y', fontsize=14)
ax[0].grid()


ax[1].plot(xxV, vv, color='k', label='Numérico')
ax[1].plot(ghiav[:,0], ghiav[:,1], 'x', mew=2, markersize=7, label='Experimental')
#ax[1].plot([0.5, 0.5], [-0.5, 0.4], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
ax[1].set_xlabel('x', fontsize=14)  
ax[1].set_ylabel('v/U', fontsize=14)
ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[1].grid()

plt.show()


## Velocity field
step = 2
x = pmesh.elements['x']
y = pmesh.elements['y']

x = x.reshape((nx, ny))[::step,::step]
y = y.reshape((nx, ny))[::step,::step]
um_vf = um.reshape((nx, ny))[::step,::step]
vm_vf = vm.reshape((nx, ny))[::step,::step]

fig, ax = plt.subplots()


plt.quiver(x, y, um_vf, vm_vf)

plt.plot([0.0, 1.0], [0.5, 0.5], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
plt.plot([0.5, 0.5], [0.0, 1.0], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
plt.show()


## Streamlines
x = pmesh.elements['x']
y = pmesh.elements['y']

x = x.reshape((nx, ny))
y = y.reshape((nx, ny))
um = um.reshape((nx, ny))
vm = vm.reshape((nx, ny))

fig, ax = plt.subplots()

plt.streamplot(x, y, um, vm, color='k')
plt.plot([0.0, 1.0], [0.5, 0.5], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
plt.plot([0.5, 0.5], [0.0, 1.0], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
plt.show()





## SAVE RESULTS
# import pickle

# output = {
#     'Re' : Re,
#     'pmesh' : pmesh,
#     'umesh' : umesh,
#     'vmesh' : vmesh,
#     'um' : um,
#     'vm' : vm,
#     'p' : pressure,
#     'tol' : tol
    
# }

# name = f'Reynolds_{Re}__Mesh_{nx}x{nx}.pickle'
# path = 'results/'

# with open(path + name, 'wb') as handle:
#     pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(path + name, 'rb') as handle:
#     b = pickle.load(handle)

