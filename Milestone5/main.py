"""
Poner API
"""

from numpy import linspace, zeros, reshape
from numpy.linalg import norm
from Cauchy import Cauchy_problem, Cauchy_problem_2_steps
from Temporal_schemes import *
from Convergence_and_Stability import convergence_rate, stability_region
from Physics import N_Body_Problem
import matplotlib.pyplot as plt


def Initial_Conditions(Nc, Nb): 
 
    U0 = zeros(2*Nc*Nb)
    U1 = reshape(U0, (Nb, Nc, 2))  
    r0 = reshape(U1[:, :, 0], (Nb, Nc))     
    v0 = reshape(U1[:, :, 1], (Nb, Nc))

    # body 1 
    r0[0,:] = [1, 0, 0]
    v0[0,:] = [0, 0.4, 0]

    # body 2 
    v0[1,:] = [0, -0.4, 0] 
    r0[1,:] = [-1, 0, 0]

    # body 3 
    r0[2,:] = [0, 1, 0] 
    v0[2,:] = [-0.4, 0, 0]       
    
    return U0  


def test_N_Body_Problem():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)
    Nb = 3
    Nc = 3

    def F(U, t): 
       return N_Body_Problem(U, t, Nb, Nc)     

    # Initial conditions
    U0 = Initial_Conditions(Nb, Nc)       

    U = Cauchy_problem(F, U0, t, CrankNicolson)

    Us = reshape(U, (N+1, Nb, Nc, 2)) 
    r = Us[:, :, :, 0]
    rs = reshape(r, (N+1, Nb, Nc)) 
    
    for i in range(Nb):
        plt.plot(rs[:, i, 0], rs[:, i, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory in the XY plane')
    plt.axis('equal')
    plt.grid()
    plt.show()
    

test_N_Body_Problem()