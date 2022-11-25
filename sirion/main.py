import pickle
import numpy as np
import matplotlib.pyplot as plt

from assembler import solve_cavity




nx = 80#120
ny = 80#120
reynolds = 400
tolerance = 1e-3
experimental_results = np.genfromtxt('experimental_data/Re400.csv', delimiter=',')

ghia100 = np.genfromtxt('experimental_data/Re100.csv', delimiter=',')
ghia400 = np.genfromtxt('experimental_data/Re400.csv', delimiter=',')
ghia1000 = np.genfromtxt('experimental_data/Re1000.csv', delimiter=',')

output = solve_cavity(nx, ny, reynolds, tolerance, experimental_results, save=True)


# name = f'Reynolds_{Re}__Mesh_{nx}_Tol_{tol}.pickle'
# path = 'results/'

# with open(path + name, 'wb') as handle:
#     pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## RECOVER SOLUTIONS
# e = ''
# fig, ax = plt.subplots()

# ghiau = ghia400[:,:2]
# ghiav = ghia400[:,2:]

# x = pmesh.elements['x']
# y = pmesh.elements['y']

# x = x.reshape((nx, ny))
# y = y.reshape((nx, ny))

# ax.plot(ghiau[:,0], ghiau[:,1], 'x', mew=2, markersize=7)
# ax.set_xlabel('u/U', fontsize=14)  
# ax.set_ylabel('y', fontsize=14)
# ax.grid()

# for tol in [0.001, 0.0001]:

#     name = f'Reynolds_{400}__Mesh_{80}_Tol_{tol}.pickle'
#     path = 'results/'

#     with open(path + name, 'rb') as handle:
#         output = pickle.load(handle)

#     um = output['um']
#     pmesh = output['pmesh']

#     um = um.reshape((nx, ny))

#     mid = nx // 2

#     uu[str(tol)] = um[:, mid]
#     yyU = y[:, mid]


# ax.plot(uu['0.001'], yyU, color='k', linestyle='--', label = '0.001')
# ax.plot(uu['0.0001'], yyU, color='k', label = '0.0001')
# plt.show()
