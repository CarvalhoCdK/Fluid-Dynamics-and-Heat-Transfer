import numpy as np
from time import perf_counter
from numba import njit
import matplotlib.pyplot as plt

from interp import NS_x, NS_y
from solver import tdma_2d, tdma1


class Prime(object):
    """
    """

    def __init__(self, model) -> None:

        self.model = model

    def solve(self, u, uequation, v, vequation, p, tolerance):
        """
        Steps:
        1) _get_velocity
        2) map_pressure
        3) solve_pressure
            1 - build_pressure
            2 - tdma
        4) correct_velocity
        """

        diff = 1
        tol = tolerance
        it = 0
        max_it = 100000

        error = np.zeros((max_it, 4))
        op_prime = perf_counter()

        # Time profile
        profile = {
            'get_velocity' : np.array(0.0),
            'map_pressure' : np.array(0.0),
            'build_pressure' : np.array(0.0),
            'tdma' : np.array(0.0),
            'correct_velocity' : np.array(0.0),
            'eval_errors' : np.array(0.0)
            }

        # Build pressure map
        opmap = perf_counter()
        [west, east, south, north, internal, corner] = self._map_pressure()
        edmap = perf_counter()
        profile['map_pressure'] += edmap - opmap

        while diff > tol:

            it += 1

            if it > max_it:
                break

            print(f'\n PRIME ITERATION : {it}')
            print('    Solving step ...')
        
            p0 = np.copy(p)
            u0 = np.copy(u)
            v0 = np.copy(v)

            # Calculate Velocity
            t0 = perf_counter()
            uh, Apu, vh, Apv, uunknown, vunknown = self._get_velocity(u, uequation, v, vequation, p)
            t1 = perf_counter()
            profile['get_velocity'] += t1 - t0
           
           # Solve pressure with tdma
            p, build_time, tdma_time = self.solve_pressure(uh, Apu, vh, Apv, p, west, east, south, north, internal, corner)
            profile['build_pressure'] += build_time
            profile['tdma'] += tdma_time

            t7 = perf_counter()
            u, v = self.correct_velocity(uh, Apu, vh, Apv, p, uunknown, vunknown)
            t8 = perf_counter()
            profile['correct_velocity'] += t8 - t7
           
            # Evaluate errors           
            e = 1e-9
            perro = np.max(np.abs(p - p0)) / np.abs(np.max(p) - np.min(p) + e)
            uerro = np.max(np.abs(u - u0)) / np.abs(np.max(u) - np.min(u) + e)
            verro = np.max(np.abs(v - v0)) / np.abs(np.max(v) - np.min(v) + e)
          
            error[it, 0] = perro
            error[it, 1] = uerro
            error[it, 2] = verro
            error[it, 3] = it

            dp = perro - error[it-1,0]
            du = uerro - error[it-1,1]
            dv = verro - error[it-1,2]

            print(f'    perro : {perro:.5f}  ({dp:.2f})')
            print(f'    uerro : {uerro:.5f}  ({du:.2f})')
            print(f'    verro : {verro:.5f}  ({dv:.2f})')

            diff = np.max(np.array([perro, uerro, verro]))
            
            t9 = perf_counter()
            profile['eval_errors'] += t9 - t8
            
            
        ed_prime = perf_counter()
        print(f'PRIME solution concluded in {ed_prime - op_prime}s')
        
        ## Time profile
        print(f'\n Time profile')
        for key in profile.keys():
            print(f'{key} : {profile[key]:.2f} [s]')



        return p, u, v, uunknown, vunknown, it
        

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
        uunknown = []

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
                uunknown.append(el)    

            

            unb = np.array([u[w], u[e], u[s], u[n]])
            uh[el] = (np.dot(a[1:5], unb) + a[-1]) / a[0]
            apu[el] = a[0]

        # v     
        apv = np.zeros(vmesh.elements['number'])
        vh = np.zeros(vmesh.elements['number'])
        vunknown = []

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
                vunknown.append(el)

            vnb = np.array([v[w], v[e], v[s], v[n]])
            vh[el] = (np.dot(a[1:5], vnb) + a[-1]) / a[0]
            apv[el] = a[0]

        uunknown = np.array(uunknown)
        vunknown = np.array(vunknown)

        return uh, apu, vh, apv, uunknown, vunknown


    def _map_pressure(self):
        """
        **Maybe optimize for numba, take dict out and call directly for w,e,s,n
        """
        
        pmesh = self.model['pmesh']
        n = pmesh.elements['number']
        nx = self.model['nx']

        corner = np.array([0, nx-1, n - nx, n - 1])

        west = []
        east = []
        south = []
        north = []
        internal = []

        # westnew = np.arange(0, n, nx)
        # eastnew = np.arange(nx-1, n+1, nx)
        # southnew = np.arange(0,nx)
        # northnew = np.arange(n-nx, n)

        # print(f'new : {westnew}')
        # print(f'new : {eastnew}')
        # print(f'new : {southnew}')
        # print(f'new : {northnew}')


        for el in np.arange(n):
            w = int(pmesh.neighbours['W'][el])
            e = int(pmesh.neighbours['E'][el])
            s = int(pmesh.neighbours['S'][el])
            n = int(pmesh.neighbours['N'][el])

            if w == -1:
                west.append(el)

            if e == -1:
                east.append(el)      

            if s == -1:
                south.append(el)      

            if n == -1:
                north.append(el)
            
            if np.sum(np.array([w, e, s, n]) < 0) < 1:
                internal.append(el)

        west = np.array(west)
        east = np.array(east)
        south = np.array(south)
        north = np.array(north)
        internal = np.array(internal)

        # print(f'old : {west}')
        # print(f'old : {east}')
        # print(f'old : {south}')
        # print(f'old : {north}')
        
        return west, east, south, north, internal, corner
  
    
    def solve_pressure(self,uh, Apu, vh, Apv, p, west, east, south, north, internal, corner):

        deltax = self.model['deltax']
        deltay = self.model['deltay']
        nx = self.model['nx']
        ny = self.model['ny']

        #@jit
        def build_pressure(uh, Apu, vh, Apv, p, nx, deltax, deltay, west, east, south, north, internal, corner):
            
            dof = p.shape[0]
            c = np.zeros((dof, 5))
            B = np.zeros(dof)
            el = corner[0]

            line = el // nx

            w = el + line + nx + 1
            e = w + 1
            s = el + (2*line + 1)
            n = s + nx + 2

            # Border balance
            c[el,1] = 0 # deltay*deltay / Apu[w] # Aw
            c[el,2] = deltay*deltay / Apu[e] # Ae
            c[el,3] = 0 # deltax*deltax / Apv[s] # As
            c[el,4] = deltax*deltax / Apv[n] # An
            c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

            B[el] += deltay*(- uh[e]) + deltay*(- vh[n]) + 0

            # South-east
            el = corner[1]
            line = el // nx

            w = el + line + nx + 1
            e = w + 1
            s = el + (2*line + 1)
            n = s + nx + 2

            # Border balance
            c[el,1] = deltay*deltay / Apu[w] # Aw
            c[el,2] = 0#deltay*deltay / Apu[e] # Ae
            c[el,3] = 0 # deltax*deltax / Apv[s] # As
            c[el,4] = deltax*deltax / Apv[n] # An
            c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

            B[el] += deltay*(uh[w]) + deltay*(- vh[n])

            # North-west
            el = corner[2]
            #print('North-west')
            line = el // nx

            w = el + line + nx + 1
            e = w + 1
            s = el + (2*line + 1)
            n = s + nx + 2
                            
            c[el,1] = 0#deltay*deltay / Apu[w] # Aw
            c[el,2] = deltay*deltay / Apu[e] # Ae
            c[el,3] = deltax*deltax / Apv[s] # As
            c[el,4] = 0#deltax*deltax / Apv[n] # An
            c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

            B[el] = deltay*(- uh[e]) + deltay*(vh[s])

            # North-east
            el = corner[3]
            #print('North-east')
            line = el // nx

            w = el + line + nx + 1
            e = w + 1
            s = el + (2*line + 1)
            n = s + nx + 2
                            
            c[el,1] = deltay*deltay / Apu[w] # Aw
            c[el,2] = 0#deltay*deltay / Apu[e] # Ae
            c[el,3] = deltax*deltax / Apv[s] # As
            c[el,4] = 0#deltax*deltax / Apv[n] # An
            c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

            B[el] = deltay*(uh[w]) + deltay*(vh[s])
            
            for el in internal:
                #print(f'el : {el}')
                #print('Internal \n')
                line = el // nx

                w = el + line + nx + 1
                e = w + 1
                s = el + (2*line + 1)
                n = s + nx + 2
                              
                c[el,1] = deltay*deltay / Apu[w] # Aw
                c[el,2] = deltay*deltay / Apu[e] # Ae
                c[el,3] = deltax*deltax / Apv[s] # As
                c[el,4] = deltax*deltax / Apv[n] # An
                c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

                B[el] = deltay*(uh[w] - uh[e]) + deltay*(vh[s] - vh[n])

            for el in west[1:-1]:
                line = el // nx

                w = el + line + nx + 1
                e = w + 1
                s = el + (2*line + 1)
                n = s + nx + 2

                # Border balance
                c[el,1] = 0 # deltay*deltay / Apu[w] # Aw
                c[el,2] = deltay*deltay / Apu[e] # Ae
                c[el,3] = deltax*deltax / Apv[s] # As
                c[el,4] = deltax*deltax / Apv[n] # An
                c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

                B[el] += deltay*(- uh[e]) + deltay*(vh[s] - vh[n]) + 0

            for el in east[1:-1]:
                line = el // nx

                w = el + line + nx + 1
                e = w + 1
                s = el + (2*line + 1)
                n = s + nx + 2
                
                # Border balance
                c[el,1] = deltay*deltay / Apu[w] # Aw
                c[el,2] = 0 # deltay*deltay / Apu[e] # Ae
                c[el,3] = deltax*deltax / Apv[s] # As
                c[el,4] = deltax*deltax / Apv[n] # An
                c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

                B[el] += deltay*(uh[w]) + deltay*(vh[s] - vh[n])

            for el in south[1:-1]:
                line = el // nx

                w = el + line + nx + 1
                e = w + 1
                s = el + (2*line + 1)
                n = s + nx + 2

                # Border balance
                c[el,1] = deltay*deltay / Apu[w] # Aw
                c[el,2] = deltay*deltay / Apu[e] # Ae
                c[el,3] = 0 # deltax*deltax / Apv[s] # As
                c[el,4] = deltax*deltax / Apv[n] # An
                c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

                B[el] += deltay*(uh[w] - uh[e]) + deltay*(-vh[n])

            for el in north[1:-1]:
                line = el // nx

                w = el + line + nx + 1
                e = w + 1
                s = el + (2*line + 1)
                n = s + nx + 2

                # Border balance
                c[el,1] = deltay*deltay / Apu[w] # Aw
                c[el,2] = deltay*deltay / Apu[e] # Ae
                c[el,3] = deltax*deltax / Apv[s] # As
                c[el,4] = 0 # deltax*deltax / Apv[s] # An
                c[el,0] = c[el,1] + c[el,2] + c[el,3] + c[el,4] # Ap

                B[el] += deltay*(uh[w] - uh[e]) + deltay*(vh[s])
            
            return c, B 

        t3 = perf_counter()
        c, B =  build_pressure(uh, Apu, vh, Apv, p, nx, deltax, deltay, west, east, south, north, internal, corner)
        t4 = perf_counter()
        build_time = t4 - t3


        pressures = tdma_2d(c, B, p, nx, ny, sweep ='lines', tol=1e-4)
        t5 = perf_counter()
        tdma_time = t5 - t4
        #print(f'pressures {pressures}')

        return pressures, build_time, tdma_time




    def correct_velocity(self, uh, Apu, vh, Apv, pressure, uunknown, vunknown):

        deltax = self.model['deltax']
        deltay = self.model['deltay']
        unx = self.model['unx']
        vnx = self.model['vnx']

        nx = self.model['nx']

        dof = uh.shape[0]
        #u = np.zeros(dof)
        #v = np.zeros(dof)

        #print(f'avg uh: {np.average(uh)}')
        #print(f'uu : {uunknown}')
        #print(f'vu : {vunknown}')

        # u
        for el in uunknown:
            
            line = el // unx
            p = el - line - 1 - nx
            e = p + 1

            uh[el] = uh[el] - deltay / Apu[el] * (pressure[e] - pressure[p])

            #print(f'correct: {deltay / Apu[el] * (pressure[e] - pressure[p])}')
        
        # v
        for el in vunknown:
            
            line = el // vnx
            p = el - (2*line + 1) - nx
            n = el- (2*line + 1)

            # print(f'el:{el}')
            # print(f'p:{p}')
            # print(f'n:{n}')

            vh[el] = vh[el] - deltax / Apv[el] * (pressure[n] - pressure[p])

        return uh, vh



        #nao é só interno, norte e sul