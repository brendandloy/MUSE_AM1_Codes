from numpy import zeros, linspace, arange, log
from numpy.linalg import norm
from Cauchy import Cauchy_error
from scipy.stats import linregress 


def convergence_rate(Temporal_scheme, F, U0, t):
    
    N_mesh = arange(10, 3001, 10)    
    N_meshes = len(N_mesh)
    N = len(t) - 1

    E = zeros(N_meshes)

    for n in range(N_meshes):
        t_n = linspace(t[0], t[N], N_mesh[n])
        # La primera q puede ser cualquiera, despues se recalcula log(E)
        U1, E_n = Cauchy_error(F, U0, t_n, Temporal_scheme) 
        E[n] = norm(E_n)


    logN = log(N_mesh)
    logE = log(E)

    q = abs(linregress(logN, logE).slope)

    for n in range(N_meshes):
        t_n = linspace(t[0], t[N], N_mesh[n])
        U1, E_n = Cauchy_error(F, U0, t_n, Temporal_scheme, q)
        E[n] = norm(E_n)

    logN = log(N_mesh)
    logE = log(E)

    return logN, logE, q, E


def stability_region(Temporal_scheme, x0=-4, xf=2, y0=-4, yf=4, Np=100):
    """
    Ver Calculo Numerico en Ecuaciones Diferenciales Ordinarias, Juan A. Hernandez
    """

    x = linspace(x0, xf, Np)
    y = linspace(y0, yf, Np)
    rho = zeros((Np, Np))

    U1 = 1
    t1 = 0
    t2 = 1

    for i in range(Np):
        for j in range(Np):
            w = complex(x[i], y[j])

            # Aplica un paso del esquema al problema test:
            #   du/dt = λ u ⇒ F(u,t) = w * u   (con Δt = 1 implícito en w)
            # Para eso, usamos t1=0, t2=1, U1=1 (condición inicial)
            
            def F(u, t):
                return w*u

            r = Temporal_scheme(U1, t1, t2, F)            
            
            rho[i,j] = abs(r)

    return rho, x, y
