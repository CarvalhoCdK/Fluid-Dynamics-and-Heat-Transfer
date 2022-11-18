import numpy as np
from numba import njit

from interp import NS_x, NS_y

class Prime(object):
    """
    """

    def __init__(self, model) -> None:

        self.model = model   
        

    def solve(self, u, uequation, v, vequation, p):
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
                 

    def get_pressure(self, uh, Apu, vh, Apv):
        
        pmesh = self.model['pmesh']
        deltax = self.model['deltax']
        deltay = self.model['deltay']







    def correct_velocity():
        pass
