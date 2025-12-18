from numpy import zeros, linspace, log10, polyfit, array
from numpy.linalg import norm, eigvals
from Cauchy import Cauchy_error
from Math_tools import Jacobian


def refine_mesh(t1):
    """
    Ejemplo: t1 = [0, 1, 2] --> t2 = [0, 0.5, 1, 1.5, 2]
    """
    N = len(t1) - 1  

    t2 = zeros(2*N+1) 
    for i in range(0, N): 
        t2[2*i] = t1[i]
        t2[2*i+1] = (t1[i] + t1[i+1])/2 
    
    t2[2*N] = t1[N]      

    return t2


def convergence_rate(Temporal_scheme, F, U0, t):
    
    N_meshes = 10        

    logN = zeros(N_meshes)
    logE = zeros(N_meshes) 

    t_i = t
    for i in range(N_meshes):
        N = len(t_i) - 1
        U, E = Cauchy_error(F, U0, t_i, Temporal_scheme) # Cualquier q, por defecto q = 1

        logN[i] = log10(N)
        logE[i] = log10(norm(E[N, :])) # Norma del punto con mas error (el ultimo)
        
        # Se refina la malla para la siguiente iteracion
        t_i = refine_mesh(t_i)       

    y = logE[(logE > -11) & (logE < -3)]
    x = logN[0:len(y)]
    m, b = polyfit(x, y, 1)    
    q = -m

    # Se recalcula logE una vez obtenido el orden
    logE = logE - log10(1 - 1/2**abs(q))   

    return logN, logE, q


def stability_region(Temporal_scheme, x0=-6, xf=3, y0=-6, yf=6, Np=100):
    """
    Ver GitHub de Juan A. Hernandez

    Es valido para todos los esquemas unipaso
    """

    x = linspace(x0, xf, Np)
    y = linspace(y0, yf, Np)
    rho = zeros((Np, Np))

    U1 = array([1.0], dtype=complex)    
    t1 = 0
    t2 = 1

    for i in range(Np):
        for j in range(Np):
            w = complex(x[i], y[j])           
            
            def F(U, t): 
                return w * U
            
            # F esta evaluada en U = U1 = 1, por lo que devuelve w 
            r = Temporal_scheme(U1, t1, t2, F)           
            
            rho[i,j] = abs(r[0])

    return rho, x, y


def stability(F, U_crit, t):

    def f(U):
        return F(U, t)

    J = Jacobian(f, U_crit)

    return eigvals(J)
