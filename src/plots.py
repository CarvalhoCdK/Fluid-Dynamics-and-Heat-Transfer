import pickle
import matplotlib.pyplot as plt
import numpy as np

from assembler import solve_cavity

ghia100 = np.genfromtxt('experimental_data/Re100.csv', delimiter=',')
ghia400 = np.genfromtxt('experimental_data/Re400.csv', delimiter=',')
ghia1000 = np.genfromtxt('experimental_data/Re1000.csv', delimiter=',')

## Re 100; OFICIAL RESULTS
Re = 100
mesh = 80
tol = 1e-6

name = f'Reynolds_{Re}__Mesh_{mesh}_Tol_{tol}.pickle'
path = 'results/Re100_tolerance_test/'

with open(path + name, 'rb') as handle:
    output = pickle.load(handle)

pmesh = output['pmesh']
p = output['p']
um = output['um']#[::10]
vm = output['vm']#[::10]

p = p / np.abs(np.max(p))
nx = ny = mesh

x = pmesh.elements['x']
y = pmesh.elements['y']

x = x.reshape((nx, ny))
y = y.reshape((nx, ny))
p = p.reshape((nx, ny))

fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, p)
fig.colorbar(cp) # Add a colorbar to a plot
ax.axis([x.min(), x.max(), y.min(), y.max()])
#fig.colorbar(color, ax=ax, label='P')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')
ax.grid()

rate = 5
plt.quiver(x[::rate, ::rate], y[::rate, ::rate],
          um[::rate, ::rate], vm[::rate, ::rate])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
plt.show()

# Compare to experimental
ghiau = ghia100[:,:2]
ghiav = ghia100[:,2:]

mid = int(nx // 2)

uu = uu = um[:, mid]
yyU = y[:, mid]

vv = vm[mid,:]
xxV = x[mid, :]

fig, ax = plt.subplots(figsize=(6.4,6.4))
ax.plot(uu, yyU, color='b')
ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='(Ghia, 1982)')
ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
#ax.set_aspect('equal')
ax.legend()
ax.grid()

fig, ax = plt.subplots(figsize=(6.4,6.4))
ax.plot(xxV, vv, color='b')
ax.plot(ghiav[:,0], ghiav[:,1], 'x', color='k', mew=2, markersize=7, label='(Ghia, 1982)')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('v/U', fontsize=14)
#ax.set_aspect('equal')
ax.legend()
ax.grid()
plt.show()

V = np.sqrt(np.square(um) + np.square(vm))
fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, V, cmap='coolwarm')#levels=10, cmap='coolwarm')
fig.colorbar(cp) # Add a colorbar to a plot
ax.axis([x.min(), x.max(), y.min(), y.max()])
#fig.colorbar(color, ax=ax, label='P')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')
ax.grid()


fig,ax=plt.subplots(figsize=(6.4,6.4))
cp = ax.contourf(x, y, um, cmap='coolwarm', levels=6)#, cmap='coolwarm')
fig.colorbar(cp) # Add a colorbar to a plot
ax.axis([x.min(), x.max(), y.min(), y.max()])
#fig.colorbar(color, ax=ax, label='P')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')
ax.grid()
###
ax.tick_params(axis='both', which='major', labelsize=10)
###
fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, vm, cmap='coolwarm', levels=8)#levels=10, cmap='coolwarm')
fig.colorbar(cp) # Add a colorbar to a plot
ax.axis([x.min(), x.max(), y.min(), y.max()])
#fig.colorbar(color, ax=ax, label='P')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')
ax.grid()
plt.show()

##
p_min,p_max = um.min(), um.max()
fig, ax = plt.subplots(1, 2, figsize=(12,4))
color = ax[0].pcolormesh(x, y, um, cmap='coolwarm',  vmin=p_min, vmax=p_max,
                        shading='gouraud')
#ax[0].axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax[0], label='u')
ax[0].set_xlabel('x', fontsize=14)  
ax[0].set_ylabel('y', fontsize=14)
ax[0].set_aspect('equal')

p_min,p_max = V.min(), V.max()

fig,ax=plt.subplots(1,1)
color = ax.pcolormesh(x, y, V, cmap='coolwarm',  vmin=p_min, vmax=p_max)
#ax[1].axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax, label='v')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')
plt.show()

fig, ax = plt.subplots()

plt.streamplot(x, y, um, vm, color='k')
plt.plot([0.0, 1.0], [0.5, 0.5], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
plt.plot([0.5, 0.5], [0.0, 1.0], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
plt.show()


#################################################################
#################################################################
## Re 400; OFICIAL RESULTS
Re = 400
mesh = 100
tol = 1e-6

name = f'Reynolds_{Re}__Mesh_{mesh}_Tol_{tol}.pickle'
path = 'results/RE400_TOLERANCES/'

with open(path + name, 'rb') as handle:
    output = pickle.load(handle)

pmesh = output['pmesh']
p = output['p']
um = output['um']#[::10]
vm = output['vm']#[::10]

p = p / np.abs(np.max(p))
nx = ny = mesh

x = pmesh.elements['x']
y = pmesh.elements['y']

x = x.reshape((nx, ny))
y = y.reshape((nx, ny))
p = p.reshape((nx, ny))

fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, p)
#fig.colorbar(cp) # Add a colorbar to a plot
ax.axis([x.min(), x.max(), y.min(), y.max()])
#fig.colorbar(color, ax=ax, label='P')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')
ax.grid()

rate = 8
plt.quiver(x[::rate, ::rate], y[::rate, ::rate],
          um[::rate, ::rate], vm[::rate, ::rate])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
plt.show()


# Compare to experimental
ghiau = ghia400[:,:2]
ghiav = ghia400[:,2:]

mid = int(nx // 2)

uu = uu = um[:, mid]
yyU = y[:, mid]

vv = vm[mid,:]
xxV = x[mid, :]

fig, ax = plt.subplots()
ax.plot(uu, yyU, color='b')
ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='Ghia et al. (1982)')
ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
ax.grid()

fig, ax = plt.subplots()
ax.plot(xxV, vv, color='b')
ax.plot(ghiav[:,0], ghiav[:,1], 'x', color='k', mew=2, markersize=7, label='Ghia et al. (1982)')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('v/U', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
ax.grid()
plt.show()

fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, um, cmap='coolwarm')
fig.colorbar(cp) # Add a colorbar to a plot
ax.axis([x.min(), x.max(), y.min(), y.max()])
#fig.colorbar(color, ax=ax, label='P')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_aspect('equal')
ax.grid()
###

###
fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, vm, cmap='coolwarm', levels=10)
fig.colorbar(cp) # Add a colorbar to a plot
ax.axis([x.min(), x.max(), y.min(), y.max()])
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_aspect('equal')
ax.grid()
plt.show()


fig, ax = plt.subplots()
plt.streamplot(x, y, um, vm, color='k')
plt.plot([0.0, 1.0], [0.5, 0.5], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
plt.plot([0.5, 0.5], [0.0, 1.0], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
plt.show()


#################################################################
#################################################################
## Re 1000; OFICIAL RESULTS
Re = 1000
mesh = 120
tol = 1e-6

name = f'Reynolds_{Re}__Mesh_{mesh}_Tol_{tol}.pickle'
path = 'results/RE1000_MESHES/'

with open(path + name, 'rb') as handle:
    output = pickle.load(handle)

pmesh = output['pmesh']
p = output['p']
um = output['um']#[::10]
vm = output['vm']#[::10]

p = p / np.abs(np.max(p))
nx = ny = mesh

x = pmesh.elements['x']
y = pmesh.elements['y']

x = x.reshape((nx, ny))
y = y.reshape((nx, ny))
p = p.reshape((nx, ny))

fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, p)
ax.axis([x.min(), x.max(), y.min(), y.max()])
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_aspect('equal')
ax.grid()

rate = 7
plt.quiver(x[::rate, ::rate], y[::rate, ::rate],
          um[::rate, ::rate], vm[::rate, ::rate])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
plt.show()


# Compare to experimental
ghiau = ghia1000[:,:2]
ghiav = ghia1000[:,2:]

mid = int(nx // 2)

uu = uu = um[:, mid]
yyU = y[:, mid]

vv = vm[mid,:]
xxV = x[mid, :]

fig, ax = plt.subplots()
ax.plot(uu, yyU, color='b')
ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='Ghia et al. (1982)')
ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
ax.grid()

fig, ax = plt.subplots()
ax.plot(xxV, vv, color='b')
ax.plot(ghiav[:,0], ghiav[:,1], 'x', color='k', mew=2, markersize=7, label='Ghia et al. (1982)')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('v/U', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
ax.grid()
plt.show()



fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, um, cmap='coolwarm')
fig.colorbar(cp) 
ax.axis([x.min(), x.max(), y.min(), y.max()])
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_aspect('equal')
ax.grid()
###

###
fig,ax=plt.subplots(1,1)
cp = ax.contourf(x, y, vm, cmap='coolwarm', levels=10)
fig.colorbar(cp) 
ax.axis([x.min(), x.max(), y.min(), y.max()])
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_aspect('equal')
ax.grid()
plt.show()


fig,ax=plt.subplots(1,1)
plt.streamplot(x, y, um, vm, color='k')
plt.plot([0.0, 1.0], [0.5, 0.5], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
plt.plot([0.5, 0.5], [0.0, 1.0], color='k', linestyle='--', linewidth=1.0, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
plt.show()