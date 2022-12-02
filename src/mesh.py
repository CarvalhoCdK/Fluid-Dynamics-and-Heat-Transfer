import numpy as np
import matplotlib.pyplot as plt


class Mesh(object):
    """
    Represents a 2D retangular mesh.
    ...
    
    Parameters
    ----------
        nx : int
            Number of elements in x direction
        ny : int
            Number of elements in y direction
        lx : float
            Length in x direction
        ly : float
            Length in y direction
        
    Attributes
    ----------
    nodes : dict
        'number' : Number of nodes in mesh
        'x' : ndarray (number)
        'y' : ndarray (number)

    elements : dict
        'number' : Number of elements in mesh.
        'x' : ndarray (number)
        'y' : ndarray (number)
        'nodes' : 'ndarray Nodes in element [i], by number

    neighbours : dict
        'W' : ndarray (number)
        'E' : ndarray (number)
        'S' : ndarray (number)
        'N' : ndarray (number)
    
    
    Methods
    -------
    plot()
        Display the mesh in a graph
    """


    def __init__(self, nx: int, ny: int, lx: float, ly: float):
        
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly

        self.nodes, self.elements = self._map()
        self.neighbours = self._neighbours()


    def _map(self):
        '''
        '''
        # Node coordinates
        nx = self.nx
        ny = self.ny
        lx = self.lx
        ly = self.ly

        x = np.linspace(0, lx, nx+1)
        y = np.linspace(0, ly, ny+1)
        grid = np.meshgrid(x, y)
        xv = grid[0].ravel()
        yv = grid[1].ravel()

        nodes = {
            'x' : xv,
            'y' : yv,
            'number' : len(xv)
        }
        
        # Element coordinates
        n = nx*ny
        xp = np.zeros(n)
        yp = np.zeros(n)
        el_nodes = np.zeros((nx*ny, 4))

        for el in range(nx * ny):

            line = el // nx

            n0 = el + line
            n1 = n0 + 1
            n2 = n1 + nx
            n3 = n2 + 1

            
            xp[el] = (xv[n1] + xv[n0]) / 2
            yp[el] = (yv[n2] + yv[n0]) / 2
            el_nodes[el, :] = [n0, n1, n2, n3]

        elements = {
            'x' : xp,
            'y' : yp,
            'nodes' : el_nodes,
            'number' : n
        }

        return nodes, elements

        
    def _neighbours(self):
        '''
        Identifica os vizinhos de cada volume elementar pelo seu número.
        Vizinhança com a fronteira recebe o índice -1.
        '''
        lx = self.lx
        ly = self.ly
        nx = self.nx
        ny = self.ny
        deltax = lx / nx
        deltay = ly / ny

        margin = 1e-6 * deltax

        borders = np.empty([nx * ny, 4])

        n = nx*ny

        w = np.zeros(n)
        e = np.zeros(n)
        s = np.zeros(n)
        n = np.zeros(n)

        for i in np.arange(self.elements['number']):
                  
            xP = self.elements['x'][i]
            yP = self.elements['y'][i]

            border = np.array([i - 1, i + 1, i - nx, i + nx])
 
            wb = abs(xP - deltax/2) <= margin
            eb = abs(xP + deltax/2 - lx) <= margin
            sb = abs(yP - deltay/2) <= margin
            nb = abs(yP + deltay/2 - ly) <= margin

            bc = np.array([wb, eb, sb, nb])
            border[bc] = -1

            w[i] = int(border[0])
            e[i] = int(border[1])
            s[i] = int(border[2])
            n[i] = int(border[3])

        neigh = {
            'W' : w,
            'E' : e,
            'S' : s,
            'N' : n
        }

        return neigh


    def plot(self, anotate=True):
        """
        """
        lx = self.lx
        ly = self.ly
        nx = self.nx
        ny = self.ny
        deltax = lx / nx
        deltay = ly / ny

        fig, ax = plt.subplots()

        ## Anotations
        if anotate:

            ax.scatter(self.nodes['x'], self.nodes['y'])
            ax.scatter(self.elements['x'], self.elements['y'])
            off = min(deltax, deltay) / 10

            for i in np.arange(self.elements['number']):

                plt.annotate(
                    i, # this is the text
                    (self.elements['x'][i] + off, self.elements['y'][i] + off)
                    )
        else:

            plt.tick_params(labelleft = False ,labelbottom = False)
            ax.scatter(self.nodes['x'], self.nodes['y'], marker='')
            ax.scatter(self.elements['x'], self.elements['y'], marker='')

        ax.set_aspect(1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Mesh')

        # Config ticks ad grid
        majorx = np.linspace(0, lx, nx+1)
        majory = np.linspace(0, ly, ny+1)

        minorx = np.linspace(0, lx, 2*(nx)+1)
        minory = np.linspace(0, ly, 2*(ny)+1)

        ax.set_xticks(majorx)
        ax.set_xticks(minorx, minor=True)
        ax.set_yticks(majory)
        ax.set_yticks(minory, minor=True)

        ax.grid(which='both', linewidth=2)
        ax.grid(which='minor', alpha=0.2, linestyle='--', color='dimgray')
        ax.grid(which='major', alpha=0.5, color='k')

        ax.set_xlim(left=0, right=lx)
        ax.set_ylim(bottom=0, top=ly)
       
        plt.show()

