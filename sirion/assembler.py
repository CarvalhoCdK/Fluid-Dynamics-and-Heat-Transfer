import numpy as np

from mesh import Mesh
from interp import NS_x, NS_y


# Re = 100, 400 e 1000
# Parâmtros da malha
nx = 5
ny = 5

# Reynolds
Re = 100

# Geometria
L = 5
H = 5

# Propriedades
rho = 1
gamma = 1

U = Re*gamma / (rho*L)

umesh = Mesh(nx + 1, ny, L, H)
vmesh = Mesh(nx, ny + 1, L, H)
pmesh = Mesh(nx, ny, L, H)

# Initial Guess
u = np.ones(umesh.elements['number'])*U
v = np.ones(vmesh.elements['number'])*U/2
p = np.ones(pmesh.elements['number'])

model = {
    'u' : u,  # Velocidade em x
    'v' : v,  # Velocidade em y
    'p' : p,  # Pressão
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

id = 2
w = umesh.neighbours['W'][id]
e = umesh.neighbours['E'][id]
au = momentum_u.u_internal(id, w, e)


momentum_v = NS_y(model)

id = 2
s = vmesh.neighbours['S'][id]
n = vmesh.neighbours['N'][id]
av = momentum_v.v_internal(id, s, n)