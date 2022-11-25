import pickle
import numpy as np
import matplotlib.pyplot as plt

from assembler import solve_cavity




#nx = 80#120
#ny = 80#120


# Load ghia results for validation
ghia100 = np.genfromtxt('experimental_data/Re100.csv', delimiter=',')
ghia400 = np.genfromtxt('experimental_data/Re400.csv', delimiter=',')
ghia1000 = np.genfromtxt('experimental_data/Re1000.csv', delimiter=',')



#nx = 80
#ny = 80
reynolds = 400
tolerance = 1e-5

meshes = [80, 120, 180]
for n in meshes:

    try:
        output = solve_cavity(n, n, reynolds, tolerance, ghia100, save=True)

    except:
        print(f'No convergence for mesh {n}x{n}')


# COMPARE TOLERANCE
# e = ''
# fig, ax = plt.subplots()

# ghiau = ghia400[:,:2]
# ghiav = ghia400[:,2:]

# uu = dict()
# for tol in [0.001, 0.0001]:

#     name = f'Reynolds_{400}__Mesh_{80}_Tol_{tol}.pickle'
#     path = 'results/'

#     with open(path + name, 'rb') as handle:
#         output = pickle.load(handle)

#     um = output['um']
#     pmesh = output['pmesh']
#     prime_it = output['prime_it']
#     comp_time = output['comp_time']

#     print(f'Tolerance: {tol}')
#     print(f'    Iterações: {prime_it}')
#     print(f'    Tempo [s]: {comp_time}')

#     um = um.reshape((nx, ny))

#     mid = nx // 2

#     uu[str(tol)] = um[:, mid]
#     yyU = y[:, mid]



# ax.plot(ghiau[:,0], ghiau[:,1], 'x', mew=2, markersize=7)
# ax.set_xlabel('u/U', fontsize=14)  
# ax.set_ylabel('y', fontsize=14)
# ax.grid()

# ax.plot(uu['0.001'], yyU, color='k', linestyle='--', label = '0.001')
# ax.plot(uu['0.0001'], yyU, color='k', label = '0.0001')
# ax.legend()
# plt.show()
