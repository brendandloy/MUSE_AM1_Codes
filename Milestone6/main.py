"""
Poner API
"""

from numpy import linspace, zeros, reshape, transpose, pi, meshgrid, polyfit, polyval
from numpy.linalg import norm
from Cauchy import Cauchy_problem, Cauchy_error
from Temporal_schemes import *
from Convergence_and_Stability import convergence_rate, stability_region, stability
from Physics import Lagrange_Points, CR3BP, Oscilador
import matplotlib.pyplot as plt
import os


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
    tol = 1e-8
    
    for i, L in enumerate([L1, L2, L3, L4, L5]):
        U_crit = array([L[0], L[1], L[2], 0, 0, 0])
        eigenvalues[i, :] = stability(F, U_crit, t)  
        
        print(f"Eigenvalues at L{i+1}:")
        for j in range(6):
            lam = eigenvalues[i, j]
            re = lam.real
            im = lam.imag                       

            # Limpiar parte real e imaginaria si son pequeñas
            re_clean = 0 if abs(re) < tol else re
            im_clean = 0 if abs(im) < tol else im

            # Estabilidad
            if re_clean > 0:
                status = "Unstable"
            else:
                status = "Stable" 

            # Crear un número complejo limpio
            lam_clean = complex(re_clean, im_clean)

            # Formateo inteligente para imprimir
            # if re_clean == 0 and im_clean != 0:
            #     lam_clean = f"{im_clean}j"
            # elif im_clean == 0 and re_clean != 0:
            #     lam_clean = f"{re_clean}"
            # elif re_clean == 0 and im_clean == 0:
            #     lam_clean = "0"
            # else:
            #     lam_clean = f"{re_clean} + {im_clean}j"

            print(f"lambda_{j+1}: {lam_clean}  stability: {status}")


def test_ERK():

    U0 = ([1, 0])
    T = 8*pi
    N = 5000
    t = linspace(0, T, N+1)

    logN, logE, q = convergence_rate(RK8713M, Oscilador, U0, t)

    print(f"The order of the temporal scheme is: {q}")    

    # Plot trajectory
    plt.plot(logN, logE)
    plt.xlabel('logN')
    plt.ylabel('logE')
    plt.title('Order of the temporal scheme')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def save_figure(filename, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def test_stability_region(name, scheme):
    
    rho, x, y = stability_region(scheme)

    # Configuración correcta
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')   

    X, Y = meshgrid(x, y)

    plt.figure(figsize=(7, 6))
    
    # Contourf con mapa de colores suave
    contour = plt.contourf(X, Y, transpose(rho), levels=linspace(0, 1, 200), cmap='viridis')

    # Líneas de contorno
    plt.contour(X, Y, transpose(rho), levels=linspace(0, 1, 11), colors='k', linewidths=0.6, alpha=0.5)

    # Barra de color
    cbar = plt.colorbar(contour)
    cbar.set_label(r'$\|r\|$', fontsize=12)
    cbar.set_ticks(linspace(0, 1, 11))

    # Etiquetas
    plt.xlabel(r'$\mathrm{Re}(w)$', fontsize=14)
    plt.ylabel(r'$\mathrm{Im}(w)$', fontsize=14)

    # Título
    plt.title(rf'{name} ($N_l = 4$)', fontsize=15)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"stability_{name}_Nl_4.png"
    folder = "Region Estabilidad Esquemas"
    save_figure(filename, folder)   


def test_convergence_q(name, scheme):    

    U0 = [1, 0]
    T = 8*pi
    N = 10
    t = linspace(0, T, N+1)

    logN, logE, q = convergence_rate(scheme, Oscilador, U0, t)       

    plt.figure(figsize=(7, 6))       

    # Configuración tipográfica
    plt.rc('text', usetex=False)
    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family='serif')    

    # Datos
    plt.plot(logN, logE, marker='o', linestyle='-', linewidth=1.8, markersize=6)    

    # Recta teórica de convergencia  
    mask = (logE > -11) & (logE < -3)
    # xmin, xmax = plt.gca().get_xlim()       
    logN_line = linspace(1.2, 2.2, 200)
    logE_line = logE[mask][0] - q * (logN_line - logN[mask][0])
    plt.plot(logN_line, logE_line, '--', linewidth=2, label=rf'Convergence rate $(q = {q:.2f})$')     

    # Etiquetas
    plt.xlabel(r'$\log(N)$', fontsize=13)
    plt.ylabel(r'$\log(\|E\|)$', fontsize=13)
    plt.title(rf'{name} ($N_l = 4$)', fontsize=14)    

    plt.grid(True, which='both', linestyle='--', alpha=0.5)    
    plt.legend()
    plt.tight_layout()

    filename = f"convergence_{name}_Nl_4.png"
    folder = "Convergencia Esquemas"
    save_figure(filename, folder)   


def test_convergence_p(name, scheme):    

    U0 = [1, 0]
    T = 8*pi
    N = 10
    t = linspace(0, T, N+1)

    logN, logE, p = convergence_rate(scheme, Oscilador, U0, t)       

    plt.figure(figsize=(7, 6))       

    # Configuración tipográfica
    plt.rc('text', usetex=False)
    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family='serif')    

    # Datos
    plt.plot(logN, logE, marker='o', linestyle='-', linewidth=1.8, markersize=6)    

    # Recta teórica de convergencia    
    mask = (logE > -11) & (logE < -5)
    # xmin, xmax = plt.gca().get_xlim()       
    logN_line = linspace(1.5, 3, 200)
    logE_line = logE[mask][0] - p * (logN_line - logN[mask][0])
    plt.plot(logN_line, logE_line, '--', linewidth=2, label=rf'Convergence rate $(p = {p:.2f})$')     

    # Etiquetas
    plt.xlabel(r'$\log(N)$', fontsize=13)
    plt.ylabel(r'$\log(\|E\|)$', fontsize=13)
    plt.title(f'{name} (error)', fontsize=14)     

    plt.grid(True, which='both', linestyle='--', alpha=0.5)    
    plt.legend()
    plt.tight_layout()

    filename = f"convergence_{name}_error.png"
    folder = "Convergencia Esquemas"
    save_figure(filename, folder)   


def test_stability_region_GBS():
    
    Nl_values = [1]

    # Configuración tipográfica global
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')

    fig, axes = plt.subplots(1, 1, figsize=(18, 8))
    #axes = axes.flatten()

    for ax, Nl in zip(axes, Nl_values):

        # Fijamos Nl usando lambda
        GBS_Nl = lambda U1, t1, t2, F: GBS(U1, t1, t2, F, Nl=Nl)

        rho, x, y = stability_region(GBS_Nl)
        X, Y = meshgrid(x, y)

        contour = ax.contourf(
            X, Y, transpose(rho),
            levels=linspace(0, 1, 200),
            cmap='viridis'
        )

        ax.contour(
            X, Y, transpose(rho),
            levels=linspace(0, 1, 11),
            colors='k',
            linewidths=0.5
        )

        ax.set_title(rf'GBS ($N_l = {Nl}$)', fontsize=15)
        ax.set_xlabel(r'$\mathrm{Re}(w)$')
        ax.set_ylabel(r'$\mathrm{Im}(w)$')
        ax.grid(alpha=0.3)

    # Barra de color común
    cbar = fig.colorbar(contour, ax=axes, shrink=0.95)
    cbar.set_label(r'$\|r\|$', fontsize=14)
    cbar.set_ticks(linspace(0, 1, 11))

    #plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(f"stability_GBS_Nl_1.png", "Region Estabilidad Esquemas")


def test_convergence_GBS():

    U0 = [1, 0]
    T = 8*pi
    N = 10
    t = linspace(0, T, N+1)

    Nl_values = [1]

    # Configuración tipográfica
    plt.rc('text', usetex=False)
    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family='serif')

    plt.figure(figsize=(8, 6))

    for Nl in Nl_values:

        # Fijamos Nl en el esquema
        GBS_Nl = lambda U1, t1, t2, F: GBS(U1, t1, t2, F, Nl=Nl)

        logN, logE, q = convergence_rate(GBS_Nl, Oscilador, U0, t)

        # Curva numérica
        plt.plot(
            logN, logE,
            marker='o',
            linestyle='-',
            linewidth=1.8,
            markersize=6,
            label=rf'$N_l = {Nl}$  $(q = {q:.2f})$'
        )         

    # Etiquetas
    plt.xlabel(r'$\log(N)$', fontsize=13)
    plt.ylabel(r'$\log(\|E\|)$', fontsize=13)
    plt.title('GBS', fontsize=15)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_figure("convergence_GBS_Nl_1.png", "Convergencia Esquemas")


schemes = [
    # ("RK4", RungeKutta4),
    # ("RK45", RK45),
    # ("RK547M", RK547M),
    # ("RK56", RK56),
    # ("RK658M", RK658M),
    # ("RK67", RK67),
    # ("RK78", RK78),
    # ("RK8713M", RK8713M), 
    ("GBS", GBS)   
] 

for name, scheme in schemes:
    test_stability_region(name, scheme)
    test_convergence_q(name, scheme)
    
# test_stability_region_GBS()
# test_convergence_GBS()