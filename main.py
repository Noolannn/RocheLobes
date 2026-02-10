import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from scipy.spatial import ConvexHull

q = 0.25

# Physical constant
G = 1

# Simulation parameters
delta   = 0.0005
epsilon = 1e-7
size    = 2

# Physical parameters
# M=1, a=1
M = 1
a = 1
Omega = np.sqrt(G*M/a**3)

def roche_potential(X, Y, q):
    M1 = M/(1 + q)
    M2 = M - M1
    a2 = a/(1+q)
    a1 = a - a2
    a1 = a1
    a2 = -a2
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

def roche_potential_3D(X, Y, Z, q):
    M1 = M/(1 + q)
    M2 = M - M1
    a2 = a/(1+q)
    a1 = a - a2
    a1 = a1
    a2 = -a2
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

def keep_smallest_component(faces):
    # faces: (M,3) triangles index into verts
    M = faces.shape[0]

    edge2tris = {}
    for t, (a,b,c) in enumerate(faces):
        edges = [(a,b),(b,c),(c,a)]
        for u,v in edges:
            if u > v: u, v = v, u
            edge2tris.setdefault((u,v), []).append(t)

    adj = [[] for _ in range(M)]
    for tris in edge2tris.values():
        if len(tris) == 2:
            t1, t2 = tris
            adj[t1].append(t2)
            adj[t2].append(t1)

    comp_id = -np.ones(M, dtype=int)
    comp_sizes = []
    cid = 0
    for i in range(M):
        if comp_id[i] != -1: 
            continue
        stack = [i]
        comp_id[i] = cid
        size = 0
        while stack:
            t = stack.pop()
            size += 1
            for nb in adj[t]:
                if comp_id[nb] == -1:
                    comp_id[nb] = cid
                    stack.append(nb)
        comp_sizes.append(size)
        cid += 1

    smallest = int(np.argmin(comp_sizes))
    keep = (comp_id == smallest)
    return faces[keep]

def polyhedron_volume(vertices):
    hull = ConvexHull(vertices)
    return hull.volume

def mesh_volume(verts, faces):    
    tri = verts[faces]          # (M,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    vol = np.einsum('ij,ij->i', v0, np.cross(v1, v2)) / 6.0
    return abs(vol.sum())

def roche_lobe_plot(q, whether_plot=False):
    mu=q/(1+q)
    print("mu:", mu)
    x_L1_q = bisect_root(lambda x: f(x, mu), mu-1+1e-6, mu-1e-6)
    x_L2_q = bisect_root(lambda x: f(x, mu), -5, mu-1-1e-6)
    print("xL1:", x_L1_q)
    print("xL2:", x_L2_q)
    R_L2 = mu-1-x_L2_q
    
    sim_range = 4
    x5, y5, z5 = sim_range* np.mgrid[-0.5:0.5:201j, -0.5:0.5:201j, -0.5:0.5:201j]
    vol = roche_potential_3D(x5, y5, z5, q)
        
    
    dx5 = x5[1,0,0] - x5[0,0,0]
    dy5 = y5[0,1,0] - y5[0,0,0]
    dz5 = z5[0,0,1] - z5[0,0,0]

    iso_val = roche_potential_3D(x_L1_q, 0, 0, q)
    
    #print(vol)
    #points with Roche potential smaller than that of L1
    #points with x smaller than x_L1
    #points with radii<L2's radius
    print(vol.size)
    mask = (vol < iso_val) & (x5 < x_L1_q) & ((x5**2+y5**2+z5**2) <= x_L2_q**2)
    volume = vol[mask].size/vol.size*sim_range**3
    print("volume:", volume)
    
    
    verts, faces, normals, values = measure.marching_cubes(
        vol, level=iso_val, spacing = (dx5, dy5, dz5)
    )

    verts_phys = verts.copy()
    verts_phys[:, 0] += x5.min()
    verts_phys[:, 1] += y5.min()
    verts_phys[:, 2] += z5.min()
    

    faces_lobe = keep_smallest_component(faces)
    faces_lobe_points=np.unique(faces_lobe)
    #print("vertex:", verts_phys)
    #print("face_points:", faces_lobe_points)
    #print("(verts_phys[:, 0]<x_L1_q):", verts_phys[:, 0]<x_L1_q)
    verts_lobe = verts_phys[faces_lobe_points]
    verts_lobe_companion = verts_lobe[verts_lobe[:, 0]<x_L1_q]
    volume_convex = polyhedron_volume(verts_lobe_companion)
    print("volume_convex:", volume_convex)
    
    mask_faces = (verts_phys[faces_lobe][:, :, 0] < x_L1_q).all(axis=1)
    faces_companion = faces_lobe[mask_faces]
    
    volume_mesh = mesh_volume(verts_phys, faces_companion)
    print("volume_mesh:", volume_mesh)
    
    if (whether_plot):
        fig5 = plt.figure(figsize=(10, 6), dpi=300)
        ax5 = fig5.add_subplot(111, projection='3d')
        ax5.plot_trisurf(
            verts_phys[:, 0], verts_phys[:, 1], verts_phys[:, 2],
            triangles=faces_lobe,
            cmap='Spectral',
            linewidth=0.2
        )
    
        ax5.set_box_aspect((2, 1, 1)) 
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        ax5.set_zlabel('z')
        ax5.set_xlim(-1.2,1.2)
        ax5.set_ylim(-0.5,0.5)
        ax5.set_zlim(-0.5,0.5)
        ax5.set_title(rf"Roche lobe ($q={q:.3f}$)")
        ax5.view_init(elev=30, azim=-90)
        plt.show()
    return volume,volume_convex, volume_mesh

if __name__ == "__main__":
    a2 = a/(1+q)
    a1 = a - a2
    a1 = a1
    a2 = -a2
    x = np.arange(-size, size, delta)
    y = np.arange(-size, size, delta)
    X, Y = np.meshgrid(x, y)
    Z = roche_potential(X, Y, q)

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
    ax2.set_xlabel('x/a')
    ax2.set_ylabel('y/a')
    ax2.set_zlabel(r'$\Phi_{\rm Roche}(GM/a)$')
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

    ax3.plot(xs1, f(xs1,mu), 'b', lw=1)
    ax3.plot(xs2, f(xs2,mu), 'r', lw=1)
    ax3.plot(xs3, f(xs3,mu), 'g', lw=1)

    ax3.axhline(0, color='gray', lw=0.8)
    ax3.axvline(mu-1, color='gray', lw=0.8, ls='--')
    ax3.axvline(mu,   color='gray', lw=0.8, ls='--')

    ax3.set_title(
        r"$f(x)=x-\frac{1-\mu}{(x-\mu)^2}\operatorname{sgn}(x-\mu)"
        r"-\frac{\mu}{(x-\mu+1)^2}\operatorname{sgn}(x-\mu+1)$"
        f",  $q={q}$"
    )
    ax3.set_xlabel("x/a")
    ax3.set_ylabel("f(x/a)")
    ax3.set_ylim(-10, 10)

    ax3.plot(0, 0, 'k+', ms=10)      # CM
    ax3.plot(a1, 0, 'ko', ms=5)      # M1
    ax3.plot(a2, 0, 'ko', ms=5)      # M2

    # Lagrange points
    ax3.plot(xL2, f(xL2, mu), 'b^', ms=5)
    ax3.plot(xL1, f(xL1, mu), 'r^', ms=5)
    ax3.plot(xL3, f(xL3, mu), 'g^', ms=5)

    ax3.text(xL2+0.01, f(xL2, mu)-0.3, ' L2', color='b', fontsize=10,
            va='top', ha='left')
    ax3.text(xL1+0.01, f(xL1, mu)-0.3, ' L1', color='r', fontsize=10,
            va='top', ha='left')
    ax3.text(xL3+0.01, f(xL3, mu)-0.3, ' L3', color='g', fontsize=10,
            va='top', ha='left')

    ax3.text(a1+0.01, -0.3, ' M1', color='k', fontsize=10,
            va='top', ha='left')
    ax3.text(a2+0.01, -0.3, ' M2', color='k', fontsize=10,
            va='top', ha='left')
    ax3.text(-2, -2.5,
            rf'$x(L_1)={xL1:.8f}a$' '\n'
            rf'$x(L_2)={xL2:.8f}a$' '\n'
            rf'$x(L_3)={xL3:.8f}a$',
            fontsize=10, ha='left', va='top')

    plt.show()




    xL4,xL5 = mu-1/2, mu-1/2
    yL4,yL5 = np.sqrt(3)/2, -np.sqrt(3)/2
    RP_L1 = roche_potential(xL1, 0, q)
    RP_L2 = roche_potential(xL2, 0, q)
    RP_L3 = roche_potential(xL3, 0, q)
    RP_L4 = roche_potential(xL4, yL4, q)
    RP_L5 = roche_potential(xL5, yL5, q)
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

    print(RP_Ls)
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
        X, Y, Z,
        levels=[RP_L1],
        colors='blue',
        linewidths=2.0
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


    fig2_ = plt.figure(figsize=(10, 7), dpi=300)
    ax2_ = fig2_.add_subplot(111, projection='3d')

    #ax2.plot_wireframe(X2, Y2, Z2plot, rstride=1, cstride=1, linewidth=0.5, color='k')

    ax2_.set_zlim(min(Zplot.min(), RP_L1), max(Zplot.max(), RP_L1))

    ax2_.plot_surface(
        X, Y, Zplot,
        rstride=100, cstride=100,
        color='white',
        edgecolor='k',
        linewidth=0.3,
        shade=False,
        alpha=0.3
    )

    ax2_.contour3D(X, Y, Z_masked, levels=[RP_L1], colors='r', linewidths=3.0)

    ax2_.set_xlabel('x/a')
    ax2_.set_ylabel('y/a')
    ax2_.set_zlabel(r'$\Phi_{\rm Roche}(GM/a)$')
    ax2_.set_box_aspect((1, 1, 0.3))
    # view
    ax2_.view_init(elev=35, azim=-60)
    ax2_.set_title("Roche potential for q=" + str(q))

    plt.show()


    print(roche_lobe_plot(0.01, 1)[0])

    volume_list = []
    volume_convex_list = []
    volume_mesh_list = []
    q_list = np.linspace(0.02, 1.0, 50)
    for i in q_list:
        volume_list.append(roche_lobe_plot(i)[0])
        volume_convex_list.append(roche_lobe_plot(i)[1])
        volume_mesh_list.append(roche_lobe_plot(i)[2])
    
    fig6 = plt.figure(figsize=(10, 6), dpi=300)
    ax6 = fig6.add_subplot(111)

    mu_list = q_list / (1 + q_list)
    beta = 0.462
    R_list = beta * mu_list**(1/3)
    theor_volume = 4*np.pi/3 * R_list**3

    ax6.plot(q_list, theor_volume,
            'k--', lw=2,
            label=rf"Theory (Eggleton-like $R={beta:.3f}"r"\mu^{1/3}$)")

    ax6.plot(q_list, volume_list,
            'o-', lw=1.8, ms=5,
            label=r"Roche lobe volume (grid)")

    ax6.plot(q_list, volume_mesh_list,
            'd-', lw=1.8, ms=5,
            label=r"Roche lobe volume (mesh)")

    ax6.plot(q_list, volume_convex_list,
            's:', lw=1.5, ms=4,
            label=r"Convex hull (upper bound)")

    rel_err = (theor_volume-volume_mesh_list)/volume_mesh_list

    ax6_err = ax6.twinx()

    ax6_err.plot(q_list, rel_err,
                'r^-', lw=1.5, ms=5,
                label=r"Relative error $(V_{\rm th}-V_{\rm mesh})/V_{\rm mesh}$")

    ax6_err.set_ylabel(r"Relative error", fontsize=14, color='r', rotation=270)
    ax6_err.yaxis.set_label_coords(1.09, 0.5)
    ax6_err.tick_params(axis='y', labelcolor='r')


    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_err.get_legend_handles_labels()

    ax6.legend(lines1 + lines2, labels1 + labels2,
            frameon=False, fontsize=12, loc='best')

    ax6.set_xlabel(r"Mass ratio $q = M_2/M_1$", fontsize=14)
    ax6.set_ylabel(r"Roche lobe volume ($a^3$ units)", fontsize=14)

    ax6.set_title("Roche lobe volume vs mass ratio", fontsize=14)

    ax6.grid(True, alpha=0.5, linestyle='--')
    #ax6.legend(frameon=False, fontsize=14)

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 0.25)

    plt.tight_layout()
    plt.show()
