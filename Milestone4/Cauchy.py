from numpy import zeros, linspace


def Cauchy_problem(F, U0, t, Temporal_scheme):

    N = len(t) - 1 # t0, t1, ..., tN is of length N+1
    Nv = len(U0) # Number of variables

    U = zeros((N+1, Nv))
    U[0,:] = U0

    for n in range(N):
        U[n+1,:] = Temporal_scheme(U[n,:], t[n], t[n+1], F)
        
    return U


def Cauchy_problem_2_steps(F, U0, U1, t, Temporal_scheme):

    N = len(t) - 1 # t0, t1, ..., tN is of length N+1
    Nv = len(U0) # Number of variables

    U = zeros((N+1, Nv))
    U[0,:] = U0
    U[1,:] = U1

    for n in range(1, N):
        U[n+1,:] = Temporal_scheme(U[n-1,:], U[n,:], t[n], t[n+1], F)
        
    return U


def Cauchy_error(F, U0, t, Temporal_scheme, q=1):
    """
    Richardson's extrapolation to determine error
    """

    N = len(t) - 1
    Nv = len(U0)

    E = zeros((N+1, Nv))
    t1 = t[:]
    t2 = linspace(t[0], t[N], 2*N+1)

    U1 = Cauchy_problem(F, U0, t1, Temporal_scheme)
    U2 = Cauchy_problem(F, U0, t2, Temporal_scheme)

    for n in range(0, N+1):
        E[n,:] = (U2[2*n,:] - U1[n,:])/(1-1/2**q)

    return U1, E