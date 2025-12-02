"""
Poner API
"""

from numpy import linspace, zeros, reshape, transpose, pi
from numpy.linalg import norm
from Cauchy import Cauchy_problem, Cauchy_error
from Temporal_schemes import *
from Convergence_and_Stability import convergence_rate, stability_region, stability
from Physics import Lagrange_Points, CR3BP
import matplotlib.pyplot as plt


def test_CR3BP():

    def F(U, t):
        return CR3BP(U, t, mu)

    T = 100
    N = 10000
    t = linspace(0, T, N+1)
    mu = 0.0121505856  # Sistema Tierra-Luna por ejemplo
    U0 = array([0.8, 0, 0, 0.5, 0.3, 0])

    U = Cauchy_problem(F, U0, t, RK56)

    # Primarios
    m1 = array([-mu, 0])
    m2 = array([1 - mu, 0])

    plt.figure(figsize=(10,5))
    plt.plot(m1[0], m1[1], 'o', color='gray', label='M1 (primario)')    
    plt.plot(m2[0], m2[1], 'o', color='blue', label='M2 (secundario)')
    
    plt.plot(U[:, 0], U[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory in the XY plane')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_Lagrange_Points():

    mu = 0.0121505856  # Sistema Tierra-Luna por ejemplo

    L1, L2, L3, L4, L5 = Lagrange_Points(mu)

    # Primarios
    m1 = array([-mu, 0])
    m2 = array([1 - mu, 0])

    plt.figure(figsize=(10,5))
    plt.axhline(0, color='gray', linewidth=0.5)
    
    plt.plot(m1[0], m1[1], 'o', color='gray', label='M1 (primario)')    
    plt.plot(m2[0], m2[1], 'o', color='blue', label='M2 (secundario)')

    # Puntos de Lagrange
    plt.plot(L1[0], L1[1], 'ro', label="L1")
    plt.plot(L2[0], L2[1], 'ro', label="L2")
    plt.plot(L3[0], L3[1], 'ro', label="L3")
    plt.plot(L4[0], L4[1], 'bo', label="L4")
    plt.plot(L5[0], L5[1], 'bo', label="L5")
    
    plt.title("Puntos de Lagrange en el CR3BP")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def test_stability():

    def F(U, t):
        return CR3BP(U, t, mu)
    
    T = 100
    N = 10000
    t = linspace(0, T, N+1)
    mu = 0.0121505856  # Sistema Tierra-Luna por ejemplo

    L1, L2, L3, L4, L5 = Lagrange_Points(mu)

    eigenvalues = zeros((5, 6), dtype=complex) # 6 autovalores por cada punto de Lagrange
    for i, L in enumerate([L1, L2, L3, L4, L5]):
        U_crit = array([L[0], L[1], L[2], 0, 0, 0]) # Punto de equilibrio
        eigenvalues[i, :] = stability(F, U_crit, t)   

        print(f"Eigenvalues at L{i+1}:")
        for j in range(6):            
            print(f"lambda_{j+1}: ", eigenvalues[i, j])
    


# test_CR3BP()
# test_Lagrange_Points()
test_stability()