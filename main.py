import matplotlib.pyplot as plt
import numpy as np

from sirion import tdma1, tdma_2d
from sirion import Mesh



## MALHA
nx = 3
ny = 3
a = 1
b = 1
## Expande malha para volumes fictícios
# nxe = nx + 0
# nye = ny + 0
# # Nova origem dos eixos x,y
# xe0 = a / nx
# ye0 = b / ny

# a = a + 2*xe0
# b = b + 2*ye0
# Cronstução da malha expandida
mesh = Mesh(nx, ny, a, b)

nodes = mesh.nodes
volumes = mesh.elements
neighbours = mesh.neighbours


mesh.plot(anotate=True)


# Pressure mesh
pmesh = Mesh(nx, ny, a, b)

# u mesh
#deltax = 
#deltay = 

umesh = Mesh(nx + 1, ny, a, b)
#umesh.elements['x'] 

# v mesh

vmesh = Mesh(nx, ny + 1, a, b)
#mesh2.plot(anotate=True)

for P in np.arange(umesh.elements['number']):
    line = P // (nx + 1)

    s = P - line - 1
    se = s + 1
    p = P + (nx+1) - line - 2
    e = p + 1

    print(f'\n P : {P}')
    print(f'line : {line}')
    print(f's : {s}')
    print(f'se : {se}')
    print(f'e : {e}')
    print(f'p : {p}')

    
for P in np.arange(vmesh.elements['number']):
    line = P // (nx)

    n = P + line + 1
    nw = n - 1
    p = P + line - nx
    w = p - 1

    print(f'\n P : {P}')
    print(f'line : {line}')
    print(f'n : {n}')
    print(f'nw : {nw}')
    print(f'w : {w}')
    print(f'p : {p}')
