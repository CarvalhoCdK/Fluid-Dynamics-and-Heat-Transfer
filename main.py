import matplotlib.pyplot as plt
import numpy as np

from sirion import tdma1, tdma_2d
from sirion import Mesh



## MALHA
nx = 5
ny = 5
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
