import pickle
import numpy as np

from assembler import solve_cavity




nx = 80#120
ny = 80#120
reynolds = 400
tolerance = 1e-4
experimental_results = np.genfromtxt('experimental_data/Re400.csv', delimiter=',')

output = solve_cavity(nx, ny, reynolds, tolerance, experimental_results, save=False)


## RECOVER SOLUTIONS
# e = ''
# name = f'Reynolds_{400}__Mesh_{80}x{80}{e}.pickle'
# path = 'results/'


# with open(path + name, 'rb') as handle:
#     output = pickle.load(handle)

# um = output['um']
# vm = output['vm']
# p = output['p']

# pmesh = output['pmesh']
# umesh = output['umesh']
# vmesh = output['vmesh']

# tol = output['tol']
# comp_time = output['comp_time']
# prime_it = output['prime_it']

# RUN MESH STUDY
