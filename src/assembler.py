import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import perf_counter, ctime
from numba import njit, jit

from mesh import Mesh
from interp import NS_x, NS_y
from prime import Prime



def solve_cavity(nx, ny, reynolds, tolerance, experimental_results, save=True):
   
    tol = tolerance
    Re = reynolds

    # Geometria
    L = 1
    H = 1

    # Propriedades
    rho = 1
    gamma = 1
    U = Re*gamma / (rho*L)

    model = {
        'rho' : 1.0, 
        'gamma' : 1.0,
        'a' : L,
        'b' : H,
        'nx' : nx,
        'ny' : ny,
        'deltax' : L/nx, 
        'deltay' : H/ny,   
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
    op = perf_counter()
    pv_coupling = Prime(model)

    pressure, u, v, uunknow, vunknown, it = pv_coupling.solve(u, uequation, v, vequation, p, tolerance)
    ed = perf_counter()

    comp_time = ed - op


    ## SCALE
    u = u / Re
    v = v / Re

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


    ## PLOT RESULTS
    # Pressure
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

    # Plot u
    u = u[uunknow]
    x = umesh.elements['x'][uunknow]
    y = umesh.elements['y'][uunknow]

    x = x.reshape((nx, ny-1))
    y = y.reshape((nx, ny-1))
    u = u.reshape((nx, ny-1))

    mid = nx // 2
    yyU = y[:, mid]

    p_min,p_max = u.min(), u.max()

    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    color = ax[0].pcolormesh(x, y, u, cmap='coolwarm',  vmin=p_min, vmax=p_max,# shading='gouraud')#,
                            edgecolors='k', linewidths=0)
   
    fig.colorbar(color, ax=ax[0], label='u')
    ax[0].set_xlabel('x', fontsize=14)  
    ax[0].set_ylabel('y', fontsize=14)
    ax[0].set_aspect('equal')

    # Plot v
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
  
    fig.colorbar(color, ax=ax[1], label='v')
    ax[1].set_xlabel('x', fontsize=14)  
    ax[1].set_ylabel('y', fontsize=14)
    ax[1].set_aspect('equal')
    #fig.show()
        
    # Compare to experimental
    ghia400 = experimental_results 
    ghiau = ghia400[:,:2]
    ghiav = ghia400[:,2:]

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
    ax[0].plot(ghiau[:,1], ghiau[:,0], 'x', mew=2, markersize=7)
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

    # Velocity field
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


    # Streamlines
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
    run_date = ctime()

    output = {
        'Re' : Re,
        'pmesh' : pmesh,
        'umesh' : umesh,
        'vmesh' : vmesh,
        'um' : um,
        'vm' : vm,
        'p' : pressure,
        'tol' : tol,
        'comp_time' : comp_time,
        'prime_it' : it,
        'run_date' : run_date
    }

    if save:
        name = f'Reynolds_{Re}__Mesh_{nx}_Tol_{tol}.pickle'
        path = 'results/'

        with open(path + name, 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output

