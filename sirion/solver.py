import numpy as np
import numba as nb

@nb.njit
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


def tdma_2d(C, B, T0, nxe: int, nye: int, sweep ='lines', tol=1e-4,max_it=1e6)->np.ndarray:
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
        # for start in np.array((0, n - nxe )):

        #     stop = start + nxe - 2
        #     estart = start + nxe
        #     estop = stop + nxe

        #     al = a[start:stop]
        #     bl = b[start:stop]
        #     cl = c[start:stop]

        #     dl = np.multiply(dn[start:stop], te[estart+nxe:estop+nxe]) + \
        #          np.multiply(ds[start:stop], te[estart-nxe:estop-nxe]) + \
        #          B[start:stop]

        #     tl = tdma1(al, bl, cl, dl)
        #     te[estart:estop] = tl

        # Middle lines
        lines = np.arange(0,nye)

      
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

        if it // 5000 > 0:
          print(f'            tdma current it : {it}')
          print(f'           Error : {diff} \n')

        it += 1
        if it > max_it:
            print('    TDMA: Excedido limite de iterações')
            break

    #print('Solução convergida')
    print(f'        Iterações : {it}')
    print(f'        Erro : {diff} \n')

    return te[nxe:n+nxe]
        