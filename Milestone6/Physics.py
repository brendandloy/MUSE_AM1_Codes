from numpy import concatenate, array, reshape, zeros, sqrt
from numpy.linalg import norm
from Math_tools import Newton
from scipy.optimize import newton


def Kepler(U, t):

    r = U[0:2]
    rdot = U[2:4]

    return concatenate((rdot, -r/norm(r)**3), axis=None)


def Oscilador(U, t):
    """
    d2x/dt2 + x = 0
    """

    x = U[0]
    xdot = U[1]

    return array((xdot, -x))


def N_Body_Problem(U, t, Nb, Nc):

    Us = reshape(U, (Nb, Nc, 2))
    r = reshape(Us[:,:,0], (Nb, Nc))
    v = reshape(Us[:,:,1], (Nb, Nc))

    F = zeros(Nb*Nc*2)
    Fs = reshape(F, (Nb, Nc, 2))
    drdt = reshape(Fs[:,:,0], (Nb, Nc))
    dvdt = reshape(Fs[:,:,1], (Nb, Nc))

    for i in range(Nb):
        drdt[i,:] = v[i,:] # drdt = v is wrong

        for j in range(Nb):
            if i != j:
                dvdt[i,:] += (r[j,:] - r[i,:]) / norm(r[j,:] - r[i,:])

    return F


def CR3BP(U, t, mu):
    """
    Circular Restricted Three Body Problem (m3 << m1, m2)
    
    mu = m2/(m1 + m2)
    1 - mu = m1/(m1 + m2)
    """

    x, y, z = U[0:3]
    xdot, ydot, zdot = U[3:6]

    r1 = sqrt((x + mu)**2 + y**2 + z**2)
    r2 = sqrt((x - (1 - mu))**2 + y**2 + z**2)

    F = zeros(6, dtype=complex)
    F[0:3] = xdot, ydot, zdot
    F[3] = 2*ydot + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    F[4] = -2*xdot + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    F[5] = -(1 - mu)*z/r1**3 - mu*z/r2**3

    return F


def Lagrange_Points(mu):
    
    def L1_poly(gamma):
        return gamma**5 + (mu-3)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 + 2*mu*gamma - mu
    
    def L2_poly(gamma):
        return gamma**5 + (3-mu)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 - 2*mu*gamma - mu
    
    def L3_poly(gamma):
        return gamma**5 +(2+mu)*gamma**4+(1+2*mu)*gamma**3-(1-mu)*gamma**2-2*(1-mu)*gamma-(1-mu)    
   
    L1x = newton(L1_poly, (mu/3)**(1/3))
    L2x = newton(L2_poly, (1-(7/12)*mu))
    L3x = newton(L3_poly, (1-(7/12)*mu)) 

    L1 = array([-(L1x+mu-1), 0, 0])
    L2 = array([-(-L2x+mu-1), 0, 0])
    L3 = array([-(L3x+mu), 0, 0])
    L4 = array([-mu+0.5, sqrt(3)/2, 0])
    L5 = array([-mu+0.5, -sqrt(3)/2, 0])

    return L1, L2, L3, L4, L5

