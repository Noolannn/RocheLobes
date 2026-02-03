import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
# Physical constant
G = 1

# Simulation parameters
delta   = 0.0005
epsilon = 1e-7
size    = 2

# Physical parameters
# M=1, a=1
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

def roche_potential_3D(X, Y, Z):
    R = np.sqrt(X**2 + Y**2)
    R1 = np.sqrt((X - a1)**2 + Y**2 + Z**2)
    R2 = np.sqrt((X - a2)**2 + Y**2 + Z**2)
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
    #print(Z.min())
    #print(Z.max())
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
    ax.text(a1+0.01, -0.01, r'$M_1$', color='k', fontsize=10,
         va='top', ha='left')
    ax.text(a2+0.01, -0.01, r'$M_2$', color='k', fontsize=10,
         va='top', ha='left')
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

    #print("Roots (x):", roots)

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


    xL4,xL5 = mu-1/2, mu-1/2
    yL4,yL5 = np.sqrt(3)/2, -np.sqrt(3)/2
    RP_L1 = roche_potential(xL1, 0)
    RP_L2 = roche_potential(xL2, 0)
    RP_L3 = roche_potential(xL3, 0)
    RP_L4 = roche_potential(xL4, yL4)
    RP_L5 = roche_potential(xL5, yL5)
    fig4, ax4 = plt.subplots(figsize=(10, 10), dpi=300)
    ax4.set_aspect('equal')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_frame_on(False)

    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)

    RP_Ls = np.array([RP_L1, RP_L2, RP_L3, RP_L4, RP_L5])
    RP_Ls = np.unique(RP_Ls)
    RP_Ls.sort()

    CS_base = ax4.contour(
        X, Y, Z,
        levels=levels,
        colors='black',
        linewidths=0.4
    )
    CS_Ls = ax4.contour(
        X, Y, Z,
        levels=RP_Ls.tolist(),
        colors='black',
        linewidths=2.0
    )

    Rmask = np.sqrt(X**2 + Y**2)

    mask = Rmask > 1.4

    Z_masked = np.ma.array(Z, mask=mask)
    CS_L1 = ax4.contour(
        X, Y, Z_masked,
        levels=[RP_L1],
        colors='black',
        linewidths=4.0
    )

    ax4.plot(0, 0, 'k+', ms=10)      # CM
    ax4.plot(a1, 0, 'ko', ms=5)     # M1
    ax4.plot(a2, 0, 'ko', ms=5)     # M2
    ax4.text(a1+0.01, -0.01, r'$M_1$', color='k', fontsize=10,
            va='top', ha='left')
    ax4.text(a2+0.01, -0.01, r'$M_2$', color='k', fontsize=10,
            va='top', ha='left')

    ax4.plot(xL1, 0, 'r^', ms=6); ax4.text(xL1+0.03, 0.02, r'$L_1$', fontsize=11, color='r')
    ax4.plot(xL2, 0, 'r^', ms=6); ax4.text(xL2-0.10, 0.02, r'$L_2$', fontsize=11, color='r')
    ax4.plot(xL3, 0, 'r^', ms=6); ax4.text(xL3+0.03, 0.02, r'$L_3$', fontsize=11, color='r')
    ax4.plot(xL4, yL4, 'r^', ms=6); ax4.text(xL4+0.03, yL4+0.03, r'$L_4$', fontsize=11, color='r')
    ax4.plot(xL5, yL5, 'r^', ms=6); ax4.text(xL5+0.03, yL5-0.08, r'$L_5$', fontsize=11, color='r')

    ax4.set_title(rf"Roche equipotentials with Lagrange points ($q={q}$)")
    plt.show()

    x5, y5, z5 = 1.1* np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
    vol = roche_potential_3D(x5, y5, z5)

    dx5 = x5[1,0,0] - x5[0,0,0]
    dy5 = y5[0,1,0] - y5[0,0,0]
    dz5 = z5[0,0,1] - z5[0,0,0]

    iso_val = roche_potential_3D(-0.43807595845641933, 0, 0)
    verts, faces, normals, values = measure.marching_cubes(
        vol, level=iso_val, spacing = (dx5, dy5, dz5)
    )

    verts_phys = verts.copy()
    verts_phys[:, 0] += x5.min()
    verts_phys[:, 1] += y5.min()
    verts_phys[:, 2] += z5.min()

    fig5 = plt.figure(figsize=(10, 6), dpi=300)
    ax5 = fig5.add_subplot(111, projection='3d')

    ax5.plot_trisurf(
        verts_phys[:, 0], verts_phys[:, 1], verts_phys[:, 2],
        triangles=faces,
        cmap='Spectral',
        linewidth=0.2
    )
    ax5.set_box_aspect((2, 1, 1)) 
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.set_title(rf"Roche lobe ($q={q}$)")
    ax5.view_init(elev=30, azim=-60)
    plt.show()