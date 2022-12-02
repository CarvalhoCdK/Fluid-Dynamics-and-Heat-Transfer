import pickle
import numpy as np
import matplotlib.pyplot as plt

from assembler import solve_cavity
from time import perf_counter, ctime



# Load ghia results for verification
ghia100 = np.genfromtxt('experimental_data/Re100.csv', delimiter=',')
ghia400 = np.genfromtxt('experimental_data/Re400.csv', delimiter=',')
ghia1000 = np.genfromtxt('experimental_data/Re1000.csv', delimiter=',')



# # RUN FOR MULTIPLE MESHES
reynolds = 400
tolerance = 1e-4
meshes = [20]
for n in meshes:

    print(f' __________ MESH : {n} __________')
    print(f'{ctime()}')

    try:
        output = solve_cavity(n, n, reynolds, tolerance, ghia1000, save=True)

    except:
        print(f'No convergence for mesh {n}x{n}')


