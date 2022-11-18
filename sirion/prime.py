import numpy as np
from time import perf_counter
from numba import njit

from interp import NS_x, NS_y
from solver import tdma_2d, tdma1


class Prime(object):
    """
    """

    def __init__(self, model) -> None:

        self.model = model

    def solve(self,u, uequation, v, vequation, p):

        
        deltax = self.model['deltax']
        deltay = self.model['deltay']
        nx = self.model['nx']
        
        print(f'Calculate velocities uh and vh')
        op = perf_counter()

        uh, Apu, vh, Apv = self._get_velocity(u, uequation, v, vequation, p)

        ed = perf_counter()
        print(f'Time : {ed - op} \n')
        ####

        print(f'Map pressure elements')
        op = perf_counter()

        [west, east, south, north, internal] = self._map_pressure()

        ed = perf_counter()
        print(f'Time : {ed - op} \n')
        ####

        print('Solve Pressure')
        op = perf_counter()

        c = self.solve_pressure(uh, Apu, vh, Apv, p, nx, deltax, deltay, west, east, south, north, internal)
        print(c)
        

    def _get_velocity(self, u, uequation, v, vequation, p):
        """
        Solve algebric equations for uh and vh
        Return : uh, Apu, vh, Apv

        Fake volumes only on one direction for u and v, so no corner volumes
        """

        umesh = self.model['umesh']
        vmesh = self.model['vmesh']
        U = self.model['U']
        
        # u     
        apu = np.zeros(umesh.elements['number'])
        uh = np.zeros(umesh.elements['number'])

        for el in np.arange(u.shape[0]):
            w = int(umesh.neighbours['W'][el])
            e = int(umesh.neighbours['E'][el])
            s = int(umesh.neighbours['S'][el])
            n = int(umesh.neighbours['N'][el])

            if w == -1:
                a = uequation.boundary(el, 'W', 0, u, v, p)

            elif e == -1:
                a = uequation.boundary(el, 'E', 0, u, v, p)        

            elif s == -1:
                a = uequation.boundary(el, 'S', 0, u, v, p)        

            elif n == -1:
                a = uequation.boundary(el, 'N', U, u, v, p)
            
            else:
                a = uequation.internal(el, u, v, p)

            unb = np.array([u[w], u[e], u[s], u[n]])
            uh[el] = (np.dot(a[1:5], unb) + a[-1]) / a[0]
            apu[el] = a[0]

        # v     
        apv = np.zeros(vmesh.elements['number'])
        vh = np.zeros(vmesh.elements['number'])
        
        for el in np.arange(u.shape[0]):
            w = int(vmesh.neighbours['W'][el])
            e = int(vmesh.neighbours['E'][el])
            s = int(vmesh.neighbours['S'][el])
            n = int(vmesh.neighbours['N'][el])

            if s == -1:
                a = vequation.boundary(el, 'S', 0, u, v, p)        

            elif n == -1:
                a = vequation.boundary(el, 'N', 0, u, v, p)

            elif w == -1:
                a = vequation.boundary(el, 'W', 0, u, v, p)

            elif e == -1:
                a = vequation.boundary(el, 'E', 0, u, v, p)        
            
            else:
                a = vequation.internal(el, u, v, p)

            vnb = np.array([v[w], v[e], v[s], v[n]])
            vh[el] = (np.dot(a[1:5], vnb) + a[-1]) / a[0]
            apv[el] = a[0]

        return uh, apu, vh, apv


    def _map_pressure(self):
        """
        **Maybe optimize for numba, take dict out and call directly for w,e,s,n
        """
        
        pmesh = self.model['pmesh']
        n = pmesh.elements['number']

        west = []
        east = []
        south = []
        north = []
        internal = []

        for el in np.arange(n):
            w = int(pmesh.neighbours['W'][el])
            e = int(pmesh.neighbours['E'][el])
            s = int(pmesh.neighbours['S'][el])
            n = int(pmesh.neighbours['N'][el])

            if w == -1:
                west.append(el)

            elif e == -1:
                east.append(el)      

            elif s == -1:
                south.append(el)      

            elif n == -1:
                north.append(el)
            
            else:
                internal.append(el)

        west = np.array(west)
        east = np.array(east)
        south = np.array(south)
        north = np.array(north)
        internal = np.array(internal)
        
        return west, east, south, north, internal


    
    
    def solve_pressure(self,uh, Apu, vh, Apv, p, nx, deltax, deltay, west, east, south, north, internal):

        deltax = self.model['deltax']
        deltay = self.model['deltay']
        nx = self.model['nx']
        ny = self.model['ny']

        @njit
        def build_pressure(uh, Apu, vh, Apv, p, nx, deltax, deltay, west, east, south, north, internal):
            
            dof = p.shape[0]
            c = np.zeros((dof, 5))
            B = np.zeros(dof)
            
            for el in internal:
                line = el // nx

                w = el + line
                e = w + 1
                s = el
                n = s + nx
                
                c[el,1] = deltay*deltay / Apu[w] # Aw
                c[el,2] = deltay*deltay / Apu[e] # Ae
                c[el,3] = deltax*deltax / Apv[s] # As
                c[el,4] = deltax*deltax / Apv[s] # An
                c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

                B[el] = deltay*(uh[w] - uh[e]) + deltay*(vh[s] - vh[n])

            for el in west:
                c[el,0] = 1 # Ap
                c[el,2] = 1 # Ae

            for el in east:
                c[el,0] = 1 # Ap
                c[el,1] = 1 # Aw

            for el in south:
                c[el,0] = 1 # Ap
                c[el,4] = 1 # An

            for el in north:
                c[el,0] = 1 # Ap
                c[el,3] = 1 # As
            
            return c, B

        c, B =  build_pressure(uh, Apu, vh, Apv, p, nx, deltax, deltay, west, east, south, north, internal)
        print(f'{c} \n')
        print(B)

        pressures = tdma_2d(c, B, p, nx, ny, sweep ='lines', tol=1e-6, max_it=1e6)

        print(pressures)




    def correct_velocity():
        pass
