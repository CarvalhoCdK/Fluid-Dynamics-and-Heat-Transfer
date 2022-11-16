
import numpy as np

from mesh import Mesh


class NS_x(object):
    """
    """
    def __init__(self, model) -> None:
        self.model = model
        

    def _wuds(self,u, rho, gamma, delta):

            Pe = rho*u*delta / gamma
            alfa = Pe**2 / (10 + 2*Pe**2)
            beta = (1 + 0.005*Pe**2) / (1 + 0.05*Pe**2)

            return alfa, beta
        
    
    def u_internal(self, id: int, W: int, E: int):
        """
        Ap*uP = Aw*uW + Ae*uE + As*uS + An*uN + Lp + B

        Return:
            [Ap, Aw, Ae, As, An, Lp, B]
        """
        u = self.model['u']
        v = self.model['v']
        p = self.model['p']

        rho = self.model['rho']
        gamma = self.model['gamma']
        deltax = self.model['deltax']
        deltay = self.model['deltay']
        nx = self.model['nx']
        ny = self.model['ny']

        ## Index mapping
        # u : centered
        P0 = id
        W0 = int(W)#neighbours['W']
        E0 = int(E)#neighbours['E']

        uW0 = u[W0]
        uP0 = u[P0]
        uE0 = u[E0]

        # v : staggered
        line = id // nx
        P0 = int(id + nx - line - 2)
        E0 = int(P0 + 1)
        S0 = int(id - line - 1)
        SE0 = int(S0 + 1)     

        vP0 = v[P0]
        vE0 = v[E0]
        vS0 = v[S0]
        vSE0 = v[SE0]

        # pressure : staggered
        P0 = int(id - line - 1)
        E0 = int(P0 + 1)
        
        pP0 = p[P0]
        pE0 = p[E0]

        alfax, betax = self._wuds(uP0, rho, gamma, deltax)
        print(f'uP0 : {uP0}')

        print(f'alfax : {alfax}')
        print(f'betax : {betax}')

        vC0 = (vS0 + vSE0 + vE0 + vP0) / 4 # v at volume center
        alfay, betay = self._wuds(vC0, rho, gamma, deltay)
        print(f'alfay : {alfay}')

        fw = rho*deltay*(uW0 + uP0) / 2
        fe = rho*deltay*(uE0 + uP0) / 2
        fs = rho*deltax*(vS0 + vSE0) / 2
        fn = rho*deltax*(vE0 + vP0) / 2

        dx = gamma*deltay / deltax
        dy = gamma*deltax / deltay

        Lp = -deltay*(pE0 - pP0)

        Aw =  (0.5 + alfax)*fw + betax*dx
        Ae = -(0.5 - alfax)*fe + betax*dx
        As =  (0.5 + alfay)*fs + betay*dy
        An = -(0.5 + alfay)*fn + betay*dy
        Ap = Aw + Ae + As + An

        return np.array([Ap, Aw, Ae, As, An, Lp, 0])


    def boundary(self, face, t):
        """
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
        alfax = Pe**2 / (10 + 2*Pe*2)
        # y :>
        Pe = rho*v*dy / gamma
        alfay = Pe**2 / (10 + 2*Pe*2)

        A =  np.zeros(6)
        A[-1] = t  # B

        # [Ap, Aw, Ae, As, An, B]
        # [0,  1,  2,  3,  4,  5]
        if face =='W':
            A[0] = 0.5 + alfax
            A[2] = -0.5 + alfax  # Ae

        if face =='E':
            A[0] = 0.5 + alfax  # Ap
            A[1] = -0.5 + alfax  # Aw

        if face =='S':
            A[0] = 0.5 + alfay  # Ap
            A[4] = -0.5 + alfay  # An

        if face =='N':
            A[0] = 0.5 + alfay  # Ap
            A[3] = -0.5 + alfay  # As


        return A

####
####
class NS_y(object):
    """
    """
    def __init__(self, model) -> None:
        self.model = model
        

    def _wuds(self,u, rho, gamma, delta):

            Pe = rho*u*delta / gamma
            alfa = Pe**2 / (10 + 2*Pe**2)
            beta = (1 + 0.005*Pe**2) / (1 + 0.05*Pe**2)

            return alfa, beta
        
    
    def v_internal(self, id: int, S: int, N: int):
        """
        Ap*uP = Aw*uW + Ae*uE + As*uS + An*uN + Lp + B

        Return:
            [Ap, Aw, Ae, As, An, Lp, B]
        """
        u = self.model['u']
        v = self.model['v']
        p = self.model['p']

        rho = self.model['rho']
        gamma = self.model['gamma']
        deltax = self.model['deltax']
        deltay = self.model['deltay']
        nx = self.model['nx']
        ny = self.model['ny']

        ## Index mapping
        # u : centered
        P0 = id
        S0 = int(S)#neighbours['S']
        N0 = int(N)#neighbours['N']

        vS0 = v[S0]
        vP0 = v[P0]
        vN0 = v[N0]

        # v : staggered
        line = id // nx
        P0 = int(id + line - nx)
        W0 = int(P0 - 1)
        N0 = int(id + line + 1)
        NW0 = int(N0 + 1)     

        uP0 = u[P0]
        uW0 = u[W0]
        uN0 = u[N0]
        uNW0 = u[NW0]

        # pressure : staggered
        P0 = int(id - nx)
        N0 = int(id)
        
        pP0 = p[P0]
        pN0 = p[N0]

        uC0 = (uP0 + uW0 + uN0 + uNW0) / 4 # v at volume center
        print(f'uC0 : {uC0}')
        alfax, betax = self._wuds(uC0, rho, gamma, deltax)

        print(f'alfax : {alfax}')
        print(f'betax : {betax}')
        
        alfay, betay = self._wuds(vP0, rho, gamma, deltay)
        print(f'alfay : {alfay}')
        print(f'betay : {betay}')

        fw = rho*deltay*(uW0 + uNW0) / 2
        fe = rho*deltay*(uP0 + uN0) / 2
        fs = rho*deltax*(vP0 + vS0) / 2
        fn = rho*deltax*(vP0 + vN0) / 2

        dx = gamma*deltay / deltax
        dy = gamma*deltax / deltay

        Lp = -deltax*(pN0 - pP0)

        Aw =  (0.5 + alfax)*fw + betax*dx
        Ae = -(0.5 - alfax)*fe + betax*dx
        As =  (0.5 + alfay)*fs + betay*dy
        An = -(0.5 + alfay)*fn + betay*dy
        Ap = Aw + Ae + As + An

        return np.array([Ap, Aw, Ae, As, An, Lp, 0])