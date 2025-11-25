"""
Poner API
"""

from numpy import linspace, concatenate, array, meshgrid, transpose
from numpy.linalg import norm
from Cauchy import Cauchy_problem, Cauchy_problem_2_steps
from Temporal_schemes import *
from Convergence_and_Stability import convergence_rate, stability_region
import matplotlib.pyplot as plt


def F(U, t):

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


def test_Cauchy_1_step():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)

    # Initial conditions
    U0 = ([0, 1])        

    U = Cauchy_problem(Oscilador, U0, t, Inverse_Euler)
    
    plt.plot(U[:, 0], U[:, 1]) 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory in the XY plane')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def test_Cauchy_2_steps():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)

    # Initial conditions
    U0 = ([0, 1]) 
    U1 = Euler(U0, t[0], t[1], Oscilador)           

    U = Cauchy_problem_2_steps(Oscilador, U0, U1, t, LeapFrog)
    
    plt.plot(U[:, 0], U[:, 1])  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory in the XY plane')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def test_Stability():

    # No me funciona para Inverse_Euler, CrankNicolson y LeapFrog
    rho, x, y = stability_region(CrankNicolson)    
    
    plt.contour(x, y, transpose(rho), linspace(0, 1, 11))
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.axis('equal')
    plt.grid()
    plt.show()


test_Stability()