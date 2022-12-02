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


##############################################################################
##############################################################################
##############################################################################
## Multiplots

ghiau = ghia100[:,:2]
ghiav = ghia100[:,2:]

uu = dict()
uNormal = dict()
yyU = dict()
yNormal = dict()

mesh = 80

for tol in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:

    name = f'Reynolds_{100}__Mesh_{80}_Tol_{tol}.pickle'
    path = 'results/Re100_tolerance_test/'

    with open(path + name, 'rb') as handle:
        output = pickle.load(handle)

    nx = ny = 80
    um = output['um']
    pmesh = output['pmesh']

    y = pmesh.elements['y']
    y = y.reshape((nx, ny))

    prime_it = output['prime_it']
    comp_time = output['comp_time']


    print(f'Tolerance: {tol}x{tol}')
    print(f'    Iterações: {prime_it}')
    print(f'    Tempo [s]: {comp_time}')

    um = um.reshape((nx, ny))

    mid = nx // 2

    
    uu[str(tol)] = um[:, mid]
    yyU[str(tol)] = y[:, mid]
   

fig, ax = plt.subplots()


ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.grid()

## FOR Re100
## By Color
ax.plot(uu['0.01'], yyU['0.01'], label = '1e-2')
ax.plot(uu['0.001'], yyU['0.001'], label = '1e-3')
ax.plot(uu['0.0001'], yyU['0.0001'], label = '1e-4')
ax.plot(uu['1e-05'], yyU['1e-05'], label = '1e-5')
ax.plot(uu['1e-06'], yyU['1e-06'], label = '1e-6')

ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='Ghia et al. (1982)')
ax.legend()
plt.show()

# Error
i = 1
meshes = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
for nx in meshes[:-1]:#]:
    error = np.sum(np.abs(uu[str(nx)] - uu[str(meshes[i])])) / 80
    print(f'{nx}x{nx} : {error*100}')
    i += 1




## COMPARE MESH

ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.grid()

## FOR Re100
## By Color
ax.plot(uu['0.001'], yyU['0.0001'], label = '1e-3')
ax.plot(uu['0.0001'], yyU['0.0001'], label = '1e-4')
ax.plot(uu['1e-05'], yyU['1e-05'], label = '1e-5')

ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='(Ghia et al.)')
ax.legend()
plt.show()

ghiau = ghia100[:,:2]
ghiav = ghia100[:,2:]

uu = dict()
uNormal = dict()
yyU = dict()
yNormal = dict()

for mesh in [10, 20, 40, 80, 100]:

    name = f'Reynolds_{100}__Mesh_{mesh}_Tol_{1e-5}.pickle'
    path = 'results/Re100 _mesh_test/'

    with open(path + name, 'rb') as handle:
        output = pickle.load(handle)

    nx = ny = mesh
    um = output['um']
    pmesh = output['pmesh']

    y = pmesh.elements['y']
    y = y.reshape((nx, ny))

    prime_it = output['prime_it']
    comp_time = output['comp_time']


    print(f'Mesh: {mesh}x{mesh}')
    print(f'    Iterações: {prime_it}')
    print(f'    Tempo [s]: {comp_time}')

    um = um.reshape((nx, ny))

    mid = nx // 2

    
    uu[str(mesh)] = um[:, mid]
    yyU[str(mesh)] = y[:, mid]

    uNormal[str(mesh)] = um[0:nx:int(mesh/10), mid]
    yNormal[str(mesh)] = y[0:nx:int(mesh/10), mid]

    

fig, ax = plt.subplots()


ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.grid()

## FOR Re100
## By Color
ax.plot(uu['10'], yyU['10'], label = '10x10')
ax.plot(uu['20'], yyU['20'], label = '20x20')
ax.plot(uu['40'], yyU['40'], label = '40x40')
ax.plot(uu['80'], yyU['80'], label = '80x80')
ax.plot(uu['100'], yyU['100'], label = '100x100')

ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='(Ghia, 1982)')
ax.legend()
plt.show()

## By Linestyle
# ax.plot(uu['10'], yyU['10'])#, color='k', marker='.', label = '10x10')
# ax.plot(uu['20'], yyU['20'], color='k', marker='*', label = '20x20')
# ax.plot(uu['40'], yyU['40'], color='k', label = '40x40')
# ax.plot(uu['80'], yyU['80'], color='k', linestyle='-.', label = '80x80')
# ax.plot(uu['160'], yyU['160'])#, color='k', linestyle='')
# ax.legend()
# plt.show()

## Mean variation## By Color
fig, ax = plt.subplots()
ax.plot(uNormal['10'], yNormal['10'], label = '10x10')
ax.plot(uNormal['20'], yNormal['20'], label = '20x20')
ax.plot(uNormal['40'], yNormal['40'], label = '40x40')
ax.plot(uNormal['80'], yNormal['80'], label = '80x80')
ax.plot(uNormal['160'], yNormal['160'], label = '160x160')
ax.legend()
plt.show()

i = 1
meshes = [100, 80, 40, 20, 10]
for nx in meshes[:-1]:#]:
    error = np.sum(np.abs(uNormal[str(nx)] - uNormal[str(meshes[i])])) / 10
    print(f'{nx}x{nx} : {error*100}')
    i += 1



# ## FOR Re400
# # ax.plot(uu['80'], yyU['80'], color='k', linestyle='--', label = '80x80')
# # ax.plot(uu['120'], yyU['120'], color='k', label = '120x120')
# # ax.legend()
# # plt.show()

#error = np.sum(np.abs(uNormal['120'] - uNormal['80'])) / 20

# ax.plot(uNormal['80'], yNormal['80'], color='k', marker = '*', label = '80x80')
# ax.plot(uNormal['120'], yNormal['120'], color='k', marker = '*', label = '120x120')
# ax.legend()
# plt.show()




############################################################################
############################################################################
## Re 400

ghiau = ghia400[:,:2]
ghiav = ghia400[:,2:]

uu = dict()
uNormal = dict()
yyU = dict()
yNormal = dict()

for mesh in [10, 20, 40, 80, 100]:

    name = f'Reynolds_{400}__Mesh_{mesh}_Tol_{1e-5}.pickle'
    path = 'results/RE400_MESHES/'

    with open(path + name, 'rb') as handle:
        output = pickle.load(handle)

    nx = ny = mesh
    um = output['um']
    pmesh = output['pmesh']

    y = pmesh.elements['y']
    y = y.reshape((nx, ny))

    prime_it = output['prime_it']
    comp_time = output['comp_time']


    print(f'Mesh: {mesh}x{mesh}')
    print(f'    Iterações: {prime_it}')
    print(f'    Tempo [s]: {comp_time}')

    um = um.reshape((nx, ny))

    mid = nx // 2

    
    uu[str(mesh)] = um[:, mid]
    yyU[str(mesh)] = y[:, mid]

    uNormal[str(mesh)] = um[0:nx:int(mesh/10), mid]
    yNormal[str(mesh)] = y[0:nx:int(mesh/10), mid]

    

fig, ax = plt.subplots()

ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='Ghia et al. (1982)')
ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.grid()

ax.plot(uu['10'], yyU['10'], label = '10x10')
ax.plot(uu['20'], yyU['20'], label = '20x20')
ax.plot(uu['40'], yyU['40'], label = '40x40')
ax.plot(uu['80'], yyU['80'], label = '80x80')
ax.plot(uu['100'], yyU['100'], label = '100x100')
#ax.plot(uu['160'], yyU['160'], label = '160x160')
ax.legend()
plt.show()


i = 1
meshes = [10, 20, 40, 80, 100]
for nx in meshes[:-1]:#]:
    error = np.sum(np.abs(uNormal[str(nx)] - uNormal[str(meshes[i])])) / 10
    print(f'{meshes[i]}x{meshes[i]} : {error*100}')
    i += 1




## TOLERANCE

ghiau = ghia400[:,:2]
ghiav = ghia400[:,2:]

uu = dict()
uNormal = dict()
yyU = dict()
yNormal = dict()

mesh = 80

for tol in [0.0001, 0.00001, 0.000001]:

    name = f'Reynolds_{400}__Mesh_{100}_Tol_{tol}.pickle'
    path = 'results/RE400_TOLERANCES/'

    with open(path + name, 'rb') as handle:
        output = pickle.load(handle)

    nx = ny = 100
    um = output['um']
    pmesh = output['pmesh']

    y = pmesh.elements['y']
    y = y.reshape((nx, ny))

    prime_it = output['prime_it']
    comp_time = output['comp_time']


    print(f'Tolerance: {tol}x{tol}')
    print(f'    Iterações: {prime_it}')
    print(f'    Tempo [s]: {comp_time}')

    um = um.reshape((nx, ny))

    mid = int(nx // 2)

    
    uu[str(tol)] = um[:, mid]
    yyU[str(tol)] = y[:, mid]
    

fig, ax = plt.subplots()


ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.grid()

## FOR Re100
## By Color
ax.plot(uu['0.001'], yyU['0.0001'], label = '1e-3')
ax.plot(uu['0.0001'], yyU['0.0001'], label = '1e-4')
ax.plot(uu['1e-05'], yyU['1e-05'], label = '1e-5')
ax.plot(uu['1e-06'], yyU['1e-06'], label = '1e-6')

ax.plot(ghiau[:,1], ghiau[:,0], 'x', color='k', mew=2, markersize=7, label='Ghia et al (1982)')
ax.legend()
plt.show()

# Error
i = 1
meshes = [0.0001, 0.00001, 0.000001]
for nx in meshes[:-1]:#]:
    error = np.sum(np.abs(uu[str(nx)] - uu[str(meshes[i])])) / 80
    print(f'{nx}x{nx} : {error*100}')
    i += 1

############################################################################
############################################################################
## Re 1000

ghiau = ghia1000[:,:2]
ghiav = ghia1000[:,2:]

uu = dict()
uNormal = dict()
yyU = dict()
yNormal = dict()

i = 0
#tol = [0.00001, 0.00001, 0.00001, 0.00001, 0.000001, 0.000001]
tol = 0.00001
for mesh in [10, 20, 40, 80, 100, 120]:

    name = f'Reynolds_{1000}__Mesh_{mesh}_Tol_{tol}.pickle'
    path = 'results/RE1000_MESHES/'

    i += 1

    with open(path + name, 'rb') as handle:
        output = pickle.load(handle)

    nx = ny = mesh
    um = output['um']
    pmesh = output['pmesh']

    y = pmesh.elements['y']
    y = y.reshape((nx, ny))

    prime_it = output['prime_it']
    comp_time = output['comp_time']


    print(f'Mesh: {mesh}x{mesh}')
    print(f'    Iterações: {prime_it}')
    print(f'    Tempo [s]: {comp_time}')

    um = um.reshape((nx, ny))

    mid = nx // 2

    
    uu[str(mesh)] = um[:, mid]
    yyU[str(mesh)] = y[:, mid]

    uNormal[str(mesh)] = um[0:nx:int(mesh/10), mid]
    yNormal[str(mesh)] = y[0:nx:int(mesh/10), mid]

    

fig, ax = plt.subplots()

ax.plot(ghiau[:,1], ghiau[:,0], 'x', mew=2, markersize=7, label='Ghia et al. (1982)', color='k')
ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.grid()

ax.plot(uu['10'], yyU['10'], label = '10x10')
ax.plot(uu['20'], yyU['20'], label = '20x20')
ax.plot(uu['40'], yyU['40'], label = '40x40')
ax.plot(uu['80'], yyU['80'], label = '80x80')
ax.plot(uu['100'], yyU['100'], label = '100x100')
ax.plot(uu['120'], yyU['120'], label = '120x120')
ax.legend()
plt.show()


i = 1
meshes = [10, 20, 40, 80, 100, 120]
for nx in meshes[:-1]:#]:
    error = np.sum(np.abs(uNormal[str(nx)] - uNormal[str(meshes[i])])) / 10
    print(f'{meshes[i]}x{meshes[i]} : {error*100}')
    i += 1


## Tolerance
ghiau = ghia1000[:,:2]
ghiav = ghia1000[:,2:]

uu = dict()
uNormal = dict()
yyU = dict()
yNormal = dict()

for mesh in [0.0001, 0.00001, 0.000001]:

    name = f'Reynolds_{1000}__Mesh_{120}_Tol_{mesh}.pickle'
    path = 'results/RE1000_MESHES/'

    #i += 1

    with open(path + name, 'rb') as handle:
        output = pickle.load(handle)

    nx = ny = 120
    um = output['um']
    pmesh = output['pmesh']

    y = pmesh.elements['y']
    y = y.reshape((nx, ny))

    prime_it = output['prime_it']
    comp_time = output['comp_time']


    print(f'Mesh: {mesh}x{mesh}')
    print(f'    Iterações: {prime_it}')
    print(f'    Tempo [s]: {comp_time}')

    um = um.reshape((nx, ny))

    mid = nx // 2

    
    uu[str(mesh)] = um[:, mid]
    yyU[str(mesh)] = y[:, mid]

    uNormal[str(mesh)] = um[0:nx:int(120/10), mid]
    yNormal[str(mesh)] = y[0:nx:int(120/10), mid]

    

fig, ax = plt.subplots()

ax.plot(ghiau[:,1], ghiau[:,0], 'x', mew=2, markersize=7, label='Ghia et al. (1982)', color='k')
ax.set_xlabel('u/U', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.grid()

ax.plot(uu['10'], yyU['10'], label = '10x10')
ax.plot(uu['20'], yyU['20'], label = '20x20')
ax.plot(uu['40'], yyU['40'], label = '40x40')
ax.plot(uu['80'], yyU['80'], label = '80x80')
ax.plot(uu['100'], yyU['100'], label = '100x100')
ax.plot(uu['120'], yyU['120'], label = '120x120')
ax.legend()
plt.show()


i = 1
meshes = [0.0001, 1e-5, 1e-6]
for nx in meshes[:-1]:#]:
    error = np.sum(np.abs(uNormal[str(nx)] - uNormal[str(meshes[i])])) / 10
    print(f'{meshes[i]}x{meshes[i]} : {error*100}')
    i += 1