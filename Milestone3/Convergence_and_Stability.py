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


