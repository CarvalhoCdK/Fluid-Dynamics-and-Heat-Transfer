## INTERPOLATIONS
import numpy as np
from numba import njit, jit


class WUDS(object):
    """
    Gera os coeficientes [Ap, Aw, Ae, As, An, B] para os volumes
    utilizando a interpolação WUDS
    """
    def __init__(self, model) -> None:
        self.model = model        
        

    def internal(self):
        """
        Coeficientes para volumes internos.
        """
        u = self.model['u']
        v = self.model['v']
        rho = self.model['rho']
        gamma = self.model['gamma']
        deltax = self.model['deltax']
        deltay = self.model['deltay']
        sp = -self.model['he']
        sc = self.model['he'] * self.model['t_chapa']

        fx = rho*u*deltay
        fy = rho*v*deltax
        dx = gamma*deltay / deltax
        dy = gamma*deltax / deltay
        # x :>
        Pe = rho*u*dx / gamma
        alfax = Pe**2 / (10 + 2*Pe**2)
        betax = (1 + 0.005*Pe**2) / (1 + 0.05*Pe**2)
        # y :>
        Pe = rho*v*dy / gamma
        alfay = Pe**2 / (10 + 2*Pe**2)
        betay = (1 + 0.005*Pe**2) / (1 + 0.05*Pe**2)

        Aw =  (0.5 + alfax)*fx + betax*dx
        Ae = -(0.5 - alfax)*fx + betax*dx
        As =  (0.5 + alfay)*fy + betay*dy
        An = -(0.5 + alfay)*fy + betay*dy
        Ap = Aw + Ae + As + An - sp*deltax*deltay

        B = sc*deltax*deltay

        return np.array([Ap, Aw, Ae, As, An, B])


    def boundary(self, face, t):
        """
        Coeficientes para volumes na fonteira.
        """
        u = self.model['u']
        v = self.model['v']
        rho = self.model['rho']
        gamma = self.model['gamma']
        deltax = self.model['deltax']
        deltay = self.model['deltay']

        dx = gamma*deltay / deltax
        dy = gamma*deltax / deltay

        # x :>
        Pe = rho*u*dx / gamma
        alfax = Pe**2 / (10 + 2*Pe**2)
        # y :>
        Pe = rho*v*dy / gamma
        alfay = Pe**2 / (10 + 2*Pe**2)

        A =  np.zeros(6)
        A[-1] = t  # B

        # [Ap, Aw, Ae, As, An, B]
        # [0,  1,  2,  3,  4,  5]
        if face =='W':
            A[0] = 0.5 + alfax  # Ap
            A[2] = -0.5 + alfax  # Ae
          
        if face =='E':
            A[0] = 0.5 + alfax  # Ap
            A[1] = -0.5 + alfax  # Aw

        if face =='S':
            A[0] = 0.5 + alfay  # Ap
            A[4] = -0.5 + alfay  # An
            # Fronteira isolada (Problema comparativo 2)
            #A[0] = 1  # Ap
            #A[4] = 1  # An

        if face =='N':
            A[0] = 0.5 + alfay  # Ap
            A[3] = -0.5 + alfay  # As
            # Fronteira isolada (Problema comparativo 2)
            #A[0] = 1  # Ap
            #A[3] = 1  # As

        return A

####################################
## MALHA

#import numpy as np


class Mesh(object):
    """
    A class used to create and manipulate retangular meshes.

    y
    |
    6---7---8
    | 2 | 3 |
    3---4---5
    | 0 | 1 |
    0---1---2 --> x
    ...

    Attributes
    ----------
    elements : list
    
    Methods
    -------
    map()
    border()
    """

    def __init__(self, nx, ny, lx, ly):
        '''
        nx : int
            Number of elements in 'x' direction
        ny : int
            Number of elements in 'y' direction
        lx : float
            Length in 'x' direction
        ly : float
            Length in 'y' direction
        '''
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly

        self.deltax = lx / nx
        self.deltay = ly / ny

        self.volumes = self.map()

        self.neighbours = self.border()


    def map(self):
        '''
        '''
        # Get grid nodes coordinates:
        nx = self.nx
        ny = self.ny
        lx = self.lx
        ly = self.ly

        x = np.linspace(0, lx, nx+1)
        y = np.linspace(0, ly, ny+1)
        grid = np.meshgrid(x, y)
        xv = grid[0].ravel()
        yv = grid[1].ravel()

        elements = np.empty([nx * ny, 2])
        for el in range(nx * ny):
            line = el // nx

            n0 = el + line
            n1 = n0 + 1
            n2 = n1 + nx
            n3 = n2 + 1

            xP = (xv[n1] + xv[n0]) / 2
            yP = (yv[n2] + yv[n0]) / 2

            elements[el,:] = [xP, yP]

        return elements

        
    def border(self):
        '''
        Identifica os vizinhos de cada volume elementar pelo seu número.
        Vizinhança com a fronteira recebe o índice -1.
        '''
        lx = self.lx
        ly = self.ly
        nx = self.nx
        ny = self.ny
        deltax = self.deltax
        deltay = self.deltay

        margin = 1e-6 * lx

        borders = np.empty([nx * ny, 4])
        for i, el in enumerate(self.volumes):
                  
            xP = el[0]
            yP = el[1]

            border = np.array([i - 1, i + 1, i - nx, i + nx])
 
            wb = abs(xP - deltax/2) <= margin
            eb = abs(xP + deltax/2 - lx) <= margin
            sb = abs(yP - deltay/2) <= margin
            nb = abs(yP + deltay/2 - ly) <= margin

            bc = np.array([wb, eb, sb, nb])

            border[bc] = -1


            borders[i, :] = border

        return borders

        ##############################

        

@njit
def tdma1(a, b, c, d):
    """
    1D version
    """
    n = a.shape[0]
    t = np.ones(n)

    ## Foward loop for p and q
    p = np.zeros(n)
    q = np.zeros(n)
    p[0] = -b[0] / a[0]
    q[0] = d[0] / a[0]

    for i in np.arange(1,n):
      p[i] = -b[i] / (a[i] + c[i]*p[i-1])
      q[i] = (d[i] - c[i]*q[i-1]) / (a[i] + c[i]*p[i-1])

    ## Backward loop for t
    t[-1] = q[-1]

    for i in np.arange(n-2,-1,-1):
      t[i] = p[i]*t[i+1] + q[i]

    return t

#@jit
def tdma_2d(C, B, T0, nxe: int, nye: int, sweep ='lines', tol=1e-6,max_it=1e6)->np.ndarray:
    """
    A = [[Ap], [Aw], [Ae], [As], [An]]
    """
    a = C[:,0]  # Ap
    c = -C[:,1] # Aw
    b = -C[:,2] # Ae
    ds = C[:,3] # As
    dn = C[:,4] # An

    n = len(a)
    t = np.ones(n)
    # Extend 't' array
    ne = n + 2*nxe
    te = np.zeros(ne)
    # Fit 't' in 'te'
    te[nxe:n+nxe] = t
    
    it = 0
    diff = 1

    while diff > tol:
        t0 = np.copy(te)
  
        # First and last line: 
        for start in np.array((1, n - nxe + 1)):

            stop = start + nxe - 2
            estart = start + nxe
            estop = stop + nxe

            al = a[start:stop]
            bl = b[start:stop]
            cl = c[start:stop]

            dl = np.multiply(dn[start:stop], te[estart+nxe:estop+nxe]) + \
                 np.multiply(ds[start:stop], te[estart-nxe:estop-nxe]) + \
                 B[start:stop]

            tl = tdma1(al, bl, cl, dl)
            
            te[estart:estop] = tl

        # Middle lines
        lines = np.arange(1,nye-1)

        for line in lines:
            
            start = line*nxe
            stop = start + nxe

            estart = start + nxe
            estop = stop + nxe

            al = a[start:stop]
            bl = b[start:stop]
            cl = c[start:stop]

            dl = np.multiply(dn[start:stop], te[estart+nxe:estop+nxe]) + \
                 np.multiply(ds[start:stop], te[estart-nxe:estop-nxe]) + \
                 B[start:stop]

            tl = tdma1(al, bl, cl, dl)
            te[estart:estop] = tl
        
        diff = np.max(np.abs(te-t0))

        #if it % 100 == 0:
            #print(f'it : {it}')
            #print(f'Error : {diff} \n')

        it += 1
        if it > max_it:
            print('Excedido limite de iterações')
            break

    print('Solução convergida')
    print(f'Iterações : {it}')
    print(f'Erro : {diff} \n')

    return te[nxe:n+nxe]
        
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
##################
## MAIN
#import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

#from mesh import Mesh
#from interpolations import WUDS

## PARAMETROS

## MALHA
nx = 150
ny = 150
a = 1
b = 1

model = {
    'u' : 1.0,  # Velocidade em x
    'v' : 0.0,  # Velocidade em y
    'rho' : 1.0,  # Densidade
    'gamma' : 1.0,
    'a' : a,  # Largura
    'b' : b,  # Altura
    'deltax' : a/nx,  # Número de elementos em x
    'deltay' : b/ny,   # Número de elementos em y
    'he' : 0.0,  # Para termo fonte convectivo em chapa metálica
    't_chapa' : 4.0 # temperatura da chapa
    }
## Condições de contorno
tw = 0
te = 0
tn = 0

def ts(x,a):
    t = np.sin(np.pi*x/a)
    #t = 0
    return t

## Expande malha para volumes fictícios
nxe = nx + 2
nye = ny + 2
# Nova origem dos eixos x,y
xe0 = model['deltax']
ye0 = model['deltay']

a = a + 2*xe0
b = b + 2*ye0
# Cronstução da malha expandida
mesh = Mesh(nxe, nye, a, b)

volumes = mesh.volumes
neighbours = mesh.neighbours


interp = WUDS(model)

dof = (nxe) * (nye)
u_dof = np.ones(dof, dtype=bool)
real_t = np.zeros(dof, dtype=bool)

B = np.zeros(dof)
#A = np.zeros((dof, dof))

# coeficientes tdma
c = np.zeros((dof, 5))
# [Ap, Aw, Ae, As, An, B]
# [0,  1,  2,  3,  4,  5]
at = np.zeros(dof)
bt = np.zeros(dof)
ct = np.zeros(dof)
dn = np.zeros(dof)
ds = np.zeros(dof)

start_time1 = perf_counter()
print('Build matrix')
corner = []

for i, nb in enumerate(neighbours):
    # [Ap, Aw, Ae, As, An, B]
    # [0,  1,  2,  3,  4,  5]
    # w, e, s, n
    nb = np.array(nb, dtype="int")

    if np.sum(nb < 0) > 1:
        u_dof[i] = False
        # Não há volumes fictícios nos cantos
        corner.append(i)
    
    # Fronteira W
    elif nb[0] == -1:
        aa = interp.boundary('W', tw)
        #A[i, i] = aa[0]
        #A[i, nb[1]] = -aa[2]
        B[i] = aa[-1]
        # tdma
        c[i,0] = aa[0] # Ap
        c[i,2] = aa[2] # Ae

    # Fronteira E
    elif nb[1] == -1:
        aa = interp.boundary('E', te)
        #A[i, i] = aa[0]
        #A[i, nb[0]] = -aa[1]
        B[i] = aa[-1]
        # tdma
        c[i,0] = aa[0] # Ap
        c[i,1] = aa[1] # Aw
    
    # Fronteira N
    elif nb[3] == -1:
        aa = interp.boundary('N', tn)
        #A[i, i] = aa[0]
        #A[i, nb[2]] = -aa[3]
        B[i] = aa[-1]
        # tdma
        c[i,0] = aa[0] # Ap
        c[i,3] = aa[3] # As
                

    # Fronteira S
    elif nb[2] == -1:
        x = volumes[i][0]
        aa = interp.boundary('S', ts(x, a))
        #A[i, i] = aa[0]
        #A[i, nb[3]] = -aa[4]
        B[i] = aa[-1]
        # tdma
        c[i,0] = aa[0] # Ap
        c[i,4] = aa[4] # An

    else:

        real_t[i] = True

        aa = interp.internal()
        #A[i, i] = aa[0]
        #A[i, nb[0]] = -aa[1]
        #A[i, nb[1]] = -aa[2]
        #A[i, nb[2]] = -aa[3]
        #A[i, nb[3]] = -aa[4]
        B[i] = aa[-1]
        # tdma
        c[i,0] = aa[0] # Ap
        c[i,1] = aa[1] # Aw
        c[i,2] = aa[2] # Ae
        c[i,3] = aa[3] # As
        c[i,4] = aa[4] # An

end_time1 = perf_counter()

print(f'Builder')
print(f'Tempo de execução: {end_time1 - start_time1}\n')

print(f'Slice corners')
# Exclui os cantos do sistema

#@njit
# def foo2(arr,i):
#     N = arr.shape[0]
#     res = np.empty((N-1,N-1), arr.dtype)
#     res[:i, :i] = arr[:i, :i]
#     res[:i, i:] = arr[:i, i+1:]
#     res[i:, :i] = arr[i+1:, :i]
#     res[i:, i:] = arr[i+1:, i+1:]
#     return res

# Aff = np.copy(A)
# corner = np.array(corner)
# corner = corner - np.arange(4)
# for id in corner:
#     #print(id)
#     Aff = foo2(Aff, id)


#Auu = A[u_dof, :][:, u_dof]
end_time2 = perf_counter()
print(f'Tempo de execução: {end_time2 - end_time1} \n')

#Buu = B[u_dof]




#c = c[u_dof]

real_tuu = real_t[u_dof]

#### Solver
start_time = perf_counter()
print(f'Solução direta') 
#t = np.linalg.solve(Auu, Buu)

end_time = perf_counter()

print(f'Tempo de execução: {end_time - start_time}')

## Plots
# Separa os volumes não fictícios
x = volumes[real_t][:,0]
y = volumes[real_t][:,1]
#z = t[real_tuu]

#print(f't_direto : {z[0:10]}')

#z_min, z_max = np.abs(z).min(), np.abs(z).max()

x = x.reshape((nx, ny))-xe0
y = y.reshape((nx, ny))-ye0
#z = z.reshape((nx, ny))

# fig, ax = plt.subplots()
# color = ax.pcolormesh(x, y, z, cmap='coolwarm',  vmin=z_min, vmax=z_max)
# ax.axis([x.min(), x.max(), y.min(), y.max()])
# fig.colorbar(color, ax=ax, label='T')
# ax.set_xlabel('x', fontsize=14)  
# ax.set_ylabel('y', fontsize=14)
# ax.set_title('Direta')
# ax.grid()

np.set_printoptions(precision = 2)
#print(Auu)
#### Solver
print(f'\n Solução TDMA')
start_time = perf_counter()
print(f'c : \n{c}')
t_tdma = tdma_2d(c, B, 1, nxe, nye, tol = 1e-6)[u_dof]

end_time = perf_counter()


print(f'Tempo de execução: {end_time - start_time}')


zt = t_tdma[real_tuu]

z_min, z_max = np.abs(zt).min(), np.abs(zt).max()

#print(f't_tdma : {z[0:10]}')

zt = zt.reshape((nx, ny))

fig, ax = plt.subplots()
color = ax.pcolormesh(x, y, zt, cmap='coolwarm',  vmin=z_min, vmax=z_max)
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax, label='T')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_title('TDMA')
ax.grid()


## SOLUÇÕES ANALÍTICAS

## Difusão pura: u = v = 0
# Com interpolações como definidas no problema original, alteram-se as velocida-
# u e v para zero dentro do dic 'model'
a = model['a']
b = model['b']

t_anl = np.sinh(np.pi*y/a) * np.sin(np.pi*x/b) / np.sinh(np.pi*b/a)
z1 = t_anl.reshape((nx, ny))

fig, ax = plt.subplots()#figsize=(6, 6))
color = ax.pcolormesh(x, y, np.flip(z1), cmap='coolwarm')#,  vmin=z_min, vmax=z_max)
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(color, ax=ax, label='T')
ax.set_xlabel('x', fontsize=14)  
ax.set_ylabel('y', fontsize=14)
ax.set_title('Solução Analítica')
ax.grid()

## Adveccao unidimensional em x
# As condições de contorno definidas em 'interpolations.boundary são alteradas
# para condições de fronteira isoladas nas faces Norte e Sul. Além disso altera-
#se a temperatura na face Oeste para 1.

# Phi_0 = 1.0
# Phi_1 = 0.0

# pe = model['rho']*1*x/model['gamma']
# pl = model['rho']*1*model['a']/model['gamma']

# r = (np.exp(pe) - 1) / (np.exp(pl) - 1)

# tn2 = (Phi_1 - Phi_0)*r + Phi_0

# z2 = tn2.reshape((nx, ny))

# fig, ax = plt.subplots()#figsize=(6, 6))
# color = ax.pcolormesh(x, y, z2, cmap='coolwarm')#,  vmin=z_min, vmax=z_max)
# ax.axis([x.min(), x.max(), y.min(), y.max()])
# fig.colorbar(color, ax=ax, label='T')
# ax.set_xlabel('x', fontsize=14)  
# ax.set_ylabel('y', fontsize=14)
# ax.set_title('Solução Analítica')
# ax.grid()
