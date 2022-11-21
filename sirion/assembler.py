import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter
from numba import njit, jit

from mesh import Mesh
from interp import NS_x, NS_y
from prime import Prime


# Re = 100, 400 e 1000
# Parâmtros da malha
nx = 60
ny = 60

# Reynolds
Re = 100#00

# Geometria
L = 1
H = 1

# Propriedades
rho = 1
gamma = 1

U = Re*gamma / (rho*L)

TIMER = np.zeros(10)
j = 0

TIMER[j] = perf_counter()
j += 1

umesh = Mesh(nx + 1, ny+2, L, H)
vmesh = Mesh(nx+2, ny + 1, L, H)
pmesh = Mesh(nx, ny, L, H)

#umesh.plot()

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


## Define elemente type (internal, north, west,...) by list
##PRIME
## Guess u,v and p
uequation = NS_x(model)
vequation = NS_y(model)

u = np.ones(umesh.elements['number'])*0
# u[0] = 10
# u[5] = 20
# u[9] = 30
v = np.ones(vmesh.elements['number'])*0
p = np.ones(pmesh.elements['number'])


model['umesh'] = umesh
model['unx'] = nx + 1

model['vmesh'] = vmesh
model['vnx'] = nx + 2
model['pmesh'] = pmesh
model['U'] = U

## Get Ap, Aw, Ae, As, An and B ##################################################
## Solve algebric uh and vh
## u


pv_coupling = Prime(model)

Uh, Apu, Vh, Apv, uniternal, vinternal = pv_coupling._get_velocity(u, uequation, v, vequation, p)
#west, east, south, north = pv_coupling._map_pressure()

deltax = model['deltax']
deltay = model['deltay']
nx = model['nx']
pressure, u, v, uunknow, vunknown = pv_coupling.solve(u, uequation, v, vequation, p)

## SCALE
u = u / Re
v = v / Re
# umesh.plot()
# vmesh.plot()

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


#fig.show()
#plt.show()

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
                         edgecolors='k', linewidths=1)
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
                         edgecolors='k', linewidths=1)
#ax[1].axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax[1], label='v')
ax[1].set_xlabel('x', fontsize=14)  
ax[1].set_ylabel('y', fontsize=14)
ax[1].set_aspect('equal')
#fig.show()

fig, ax = plt.subplots(1,2, figsize=(12,4))



ghiau = np.array([[0.00E+00,0.00E+00],
[5.47E-02,-3.72E-02],
[6.25E-02,-4.19E-02],
[7.03E-02,-4.78E-02],
[1.02E-01,-6.43E-02],
[1.72E-01,-1.02E-01],
[2.81E-01,-1.57E-01],
[4.53E-01,-2.11E-01],
[5.00E-01,-2.06E-01],
[6.17E-01,-1.36E-01],
[7.34E-01,3.32E-03],
[8.52E-01,2.32E-01],
[9.53E-01,6.87E-01],
[9.61E-01,7.37E-01],
[9.69E-01,7.89E-01],
[9.77E-01,8.41E-01],
[1.00E+00,1.00E+00]])

ghiav = np.array([[0.00E+00,	0.00E+00],
[6.25E-02,	9.23E-02],
[7.03E-02,	1.01E-01],
[7.81E-02,	1.09E-01],
[9.38E-02,	1.23E-01],
[1.56E-01,	1.61E-01],
[2.27E-01,	1.75E-01],
[2.34E-01,	1.75E-01],
[5.00E-01,	5.45E-02],
[8.05E-01,	-2.45E-01],
[8.59E-01,	-2.24E-01],
[9.06E-01,	-1.69E-01],
[9.53E-01,	-1.03E-01],
[9.53E-01,	-8.86E-02],
[9.61E-01,	-7.39E-02],
[9.69E-01,	-5.91E-02],
[1.00E+00,	0.00E+00]])

#yyU = y[:, mid]
uu = u[:, mid]
ax[0].plot(uu, yyU)
ax[0].plot(ghiau[:,1], ghiau[:,0], '*')
ax[0].set_xlabel('u', fontsize=14)  
ax[0].set_ylabel('y', fontsize=14)
ax[0].grid()


vv = v[mid,:]
ax[1].plot(xxV, vv)
ax[1].plot(ghiav[:,0], ghiav[:,1], '*')
ax[1].set_xlabel('x', fontsize=14)  
ax[1].set_ylabel('v', fontsize=14)
plt.show()
ax[1].grid()

