import matplotlib.pyplot as plt
import numpy as np

# Physical constant
G = 1

# Simulation parameters
delta   = 0.0005
epsilon = 1e-7
size    = 2

# Physical parameters
# M=1, a=1, 
M = 1
q = 0.25
a = 1
Omega = np.sqrt(G*M/a**3)

# Constraint physical parameters: mass and orbital radii of two stars
M1 = M/(1 + q)
M2 = M - M1
a2 = a/(1+q)
a1 = a - a2
a1 = a1
a2 = -a2

def roche_potential(X, Y):
    R = np.sqrt(X**2 + Y**2)
    R1 = np.sqrt((X - a1)**2 + Y**2)
    R2 = np.sqrt((X - a2)**2 + Y**2)
    R1_eff = np.where(R1 < epsilon, epsilon, R1)
    R2_eff = np.where(R2 < epsilon, epsilon, R2)
    # Adds epsilon to avoid the singularity at the star position
    Phi1 = - (G * M1)/R1_eff
    Phi2 = - (G * M2)/R2_eff
    PhiC = (1/2) * (Omega**2) * (R**2)
    PhiRoche = Phi1 + Phi2 - PhiC
    return PhiRoche


mu=q/(1+q)
def sgn(x):
    return np.where(x>0,1.0, np.where(x<0,-1.0,0.0))
def f(x, mu=mu):
    return (x-(1-mu)/((x-mu)**2)*sgn(x-mu)-mu/((x-mu+1)**2)*sgn(x-mu+1))
xmin, xmax = -2.0, 2.0
eps = 1e-6
xs1 = np.linspace(xmin, mu-1-eps, 4000)
xs2 = np.linspace(mu-1+eps, mu-eps, 4000)
xs3 = np.linspace(mu+eps, xmax, 4000)
def bisect_root(func, a, b, err=1e-8):
    fa, fb = func(a), func(b)
    if not np.isfinite(fa) or not np.isfinite(fb) or fa*fb > 0:
        return None
    while abs(a-b)>err:
        m = 0.5*(a+b)
        fm = func(m)
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

if __name__ == "__main__":
    x = np.arange(-size, size, delta)
    y = np.arange(-size, size, delta)
    X, Y = np.meshgrid(x, y)
    Z = roche_potential(X, Y)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    zmin, zmax = np.percentile(Z, [30, 100])
    levels = np.linspace(zmin, zmax, 30)
    print(Z.min())
    print(Z.max())
    #levels = np.linspace(zmin, zmax, 100)
    #ax.scatter(0, 0, s=60, marker=".")
    #ax.scatter(a1, 0, s=60, marker=(5, 1))
    #ax.scatter(a2, 0, s=60, marker=(5, 1))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.plot(0, 0, 'k+', ms=10)      # CM
    ax.plot(a1, 0, 'ko', ms=5)     # M1
    ax.plot(a2, 0, 'ko', ms=5)     # M2

    CS = ax.contour(
        X, Y, Z,
        levels=levels,
        colors='black',
        linewidths=0.4
    )
    ax.set_title("Roche equipotentials for q=" + str(q))
    plt.show()

    Zplot = np.clip(Z, np.percentile(Z, 0.8), np.percentile(Z, 100))
    fig2 = plt.figure(figsize=(10, 7), dpi=300)
    ax2 = fig2.add_subplot(111, projection='3d')

    ax2.plot_surface(
        X, Y, Zplot,
        rstride=100, cstride=100,
        color='white',
        edgecolor='k',
        linewidth=0.3,
        shade=True
    )
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel(r'$\Phi_{\rm Roche}$')
    ax2.set_box_aspect((1, 1, 0.3))
    # view
    ax2.view_init(elev=35, azim=-60)
    ax2.set_title("Roche potential for q=" + str(q))

    plt.show()


    #Solve the x of L1 L2 L3
    roots = []
    roots += [bisect_root(lambda x: f(x, mu), xmin, mu-1-eps)]
    roots += [bisect_root(lambda x: f(x, mu), mu-1+eps, mu-eps)]
    roots += [bisect_root(lambda x: f(x, mu), mu+eps, xmax)]

    print("Roots (x):", roots)

    xL2 = [r for r in roots if r < mu-1][0]  
    xL1 = [r for r in roots if mu-1 < r < mu][0] 
    xL3 = [r for r in roots if r > mu][0]

    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)

    ax3.plot(xs1, f(xs1), 'b', lw=1)
    ax3.plot(xs2, f(xs2), 'r', lw=1)
    ax3.plot(xs3, f(xs3), 'g', lw=1)

    ax3.axhline(0, color='gray', lw=0.8)
    ax3.axvline(mu-1, color='gray', lw=0.8, ls='--')
    ax3.axvline(mu,   color='gray', lw=0.8, ls='--')

    ax3.set_title(
        r"$f(x)=x-\frac{1-\mu}{(x-\mu)^2}\operatorname{sgn}(x-\mu)"
        r"-\frac{\mu}{(x-\mu+1)^2}\operatorname{sgn}(x-\mu+1)$"
        f",  $q={q}$"
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("f(x)")
    ax3.set_ylim(-10, 10)

    ax3.plot(0, 0, 'k+', ms=10)      # CM
    ax3.plot(a1, 0, 'ko', ms=5)      # M1
    ax3.plot(a2, 0, 'ko', ms=5)      # M2

    # Lagrange points
    ax3.plot(xL2, f(xL2), 'b^', ms=5)
    ax3.plot(xL1, f(xL1), 'r^', ms=5)
    ax3.plot(xL3, f(xL3), 'g^', ms=5)

    ax3.text(xL2+0.01, f(xL2)-0.3, ' L2', color='b', fontsize=10,
            va='top', ha='left')
    ax3.text(xL1+0.01, f(xL1)-0.3, ' L1', color='r', fontsize=10,
            va='top', ha='left')
    ax3.text(xL3+0.01, f(xL3)-0.3, ' L3', color='g', fontsize=10,
            va='top', ha='left')

    ax3.text(-2, -2.5,
            rf'$x(L_1)={xL1:.8f}$' '\n'
            rf'$x(L_2)={xL2:.8f}$' '\n'
            rf'$x(L_3)={xL3:.8f}$',
            fontsize=10, ha='left', va='top')

    plt.show()
