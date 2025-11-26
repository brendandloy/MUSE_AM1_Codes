from numpy import concatenate, array, reshape, zeros
from numpy.linalg import norm


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