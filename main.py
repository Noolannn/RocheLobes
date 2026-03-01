import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from random import random
import time

# Physical constant
G = 0.1

# Simulation parameters
dbg = True # Optional output info for debug
delta = 0.01 # Grid step
vector_field_delta = 0.5 # Should be smaller than delta
wireframe_delta = 0.05
epsilon = 0.01 # Used for regularization near singularities
diff_step = 0.01 # Used for differentiation
error = 1e-9 # When an iterative method won't increase the accuracy of the result more than error, it will stop
size = 8.0

# Physical parameters
M = 10
q = 2
a = 4
Omega = 0.1 # 0.1

# Constraint physical parameters
M1 = M/(1 + q)
M2 = M - M1
a1 = (q * a)/(q + 1)
a2 = - a1/q

# Compute the Roche potential on the 2D plane containing stars
def roche_potential(X, Y):
    R = np.sqrt(X**2 + Y**2)
    # Adds epsilon to avoid the singularity at the star position and keep the C1 behavior
    R1 = np.sqrt((X - a1)**2 + Y**2 + epsilon**2)
    R2 = np.sqrt((X - a2)**2 + Y**2 + epsilon**2)
    Phi1 = - (G * M1)/(R1)
    Phi2 = - (G * M2)/(R2)
    PhiC = (1/2) * (Omega**2) * (R**2)
    PhiRoche = Phi1 + Phi2 - PhiC
    return PhiRoche

# Returns the 2D gradient field of the Roche potential in the plane containing the stars
def grad_roche_potential(X, Y):
    # R1 = np.sqrt((X - a1)**2 + Y**2 + epsilon**2)
    # R2 = np.sqrt((X - a2)**2 + Y**2 + epsilon**2)
    # Diff_X = - (G * M1 * (X - a1))/np.pow(R1, 3) - (G * M2 * (X - a2))/np.pow(R2, 3) + (Omega**2) * X
    # Diff_Y = - (G * M1 * Y)/np.pow(R1, 3) - (G * M2 * Y)/np.pow(R2, 3) + (Omega**2) * Y
    Partial_Y, Partial_X = np.gradient(roche_potential(X, Y)) # The order of gradient components returned is a bit confusing, be careful
    return Partial_X, Partial_Y

# Returns the opposite of the 2D gradient of the Roche potential at a given point
def grad_roche(x, y):
    r1 = np.sqrt((x - a1)**2 + y**2 + epsilon**2)
    r2 = np.sqrt((x - a2)**2 + y**2 + epsilon**2)
    return (- (G * M1 * (x - a1))/pow(r1, 3) - (G * M2 * (x - a2))/pow(r2, 3) + (Omega**2) * x), (- (G * M1 * y)/pow(r1, 3) - (G * M2 * y)/pow(r2, 3) + (Omega**2) * y)

def grad_roche_x(x, y):
    x, _ = grad_roche(x, y)
    return x

def grad_roche_y(x, y):
    _, y = grad_roche(x, y)
    return y

def partial_x_grad_roche_x(x, y):
    return (grad_roche_x(x + diff_step, y) - grad_roche_x(x - diff_step, y))/(2 * diff_step)

def partial_y_grad_roche_x(x, y):
    return (grad_roche_x(x, y + diff_step) - grad_roche_x(x, y - diff_step))/(2 * diff_step)

def partial_x_grad_roche_y(x, y):
    return (grad_roche_y(x + diff_step, y) - grad_roche_y(x - diff_step, y))/(2 * diff_step)

def partial_y_grad_roche_y(x, y):
    return (grad_roche_y(x, y + diff_step) - grad_roche_y(x, y - diff_step))/(2 * diff_step)

def jacobian_grad_roche(x, y):
    return np.array([[partial_x_grad_roche_x(x, y), partial_y_grad_roche_x(x, y)], [partial_x_grad_roche_y(x, y), partial_y_grad_roche_y(x, y)]])

point_buffer = []
def newton_method(x0, y0, step):
    point_buffer.append((x0, y0))
    if dbg: print("x = " + str(x0) + " y = " + str(y0))
    if step == 0:
        return x0, y0
    
    X = np.array([[x0], [y0]])
    F = np.array([[grad_roche_x(x0, y0)], [grad_roche_y(x0, y0)]])
    J = jacobian_grad_roche(x0, y0)
    X_new = X - np.linalg.solve(J, F)
    x_new = float(X_new[0][0])
    y_new = float(X_new[1][0])

    if abs(x0 - x_new) < error and abs(y0 - y_new) < error:
        if dbg: print("Return earlier")
        return x_new, y_new

    return newton_method(x_new, y_new, step-1)

def read_pos_from_file(file):
    pos_list = []
    with open(file) as f:
        for line in f.readlines():
            split_line = line.split(" ")
            x = float(split_line[0])
            y = float(split_line[1])
            pos_list.append((x, y))

    return pos_list

def test_particule_euler(x0, y0, vx0, vy0, duration, dt):
    """
    Uses Euler method to compute the trajectory of a particule
    
    :param x0: Starting position x
    :param y0: Starting position y
    :param vx0: Starting velocity vx
    :param vy0: Starting velocity vy
    :param duration: Duration of the trajectory for the particule
    :param dt: Timestep (the smaller, the more precise)
    :returns: Position list along the trajectory
    """
    n = int(duration/dt) # Number of iterations
    x = x0
    y = y0
    vx = vx0
    vy = vy0
    position_list = [(x0, y0)]
    for _ in range(0, n):
        ax = grad_roche_x(x, y) + 2 * Omega * vy
        ay = grad_roche_y(x, y) - 2 * Omega * vx
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
        if np.sqrt(x**2 + y**2) > size:
            if dbg: print("Distance is too large, stop")
            break

        position_list.append((x, y))
    
    return position_list

# Uses RK4 method (smaller error)
def test_particule_RK4(x0, y0, vx0, vy0, duration, dt):
    """
    Uses RK4 method (Runge-Kutta order 4) to compute the trajectory of a particule.
    It is more accurate than Euler method.
    
    :param x0: Starting position x
    :param y0: Starting position y
    :param vx0: Starting velocity vx
    :param vy0: Starting velocity vy
    :param duration: Duration of the trajectory for the particule
    :param dt: Timestep (the smaller, the more precise)
    :returns: Position list along the trajectory and Jacobi constant (which should be conserved along the trajectory)
    """
    n = int(duration/dt) # Number of iterations
    x = x0
    y = y0
    vx = vx0
    vy = vy0
    position_list = [(x0, y0)]
    jacobi_cst = [(-2) * float(roche_potential(x, y)) - (vx**2 + vy**2)]
    for _ in range(0, n):
        k1x = grad_roche_x(x, y) + 2 * Omega * vy
        k1y = grad_roche_y(x, y) - 2 * Omega * vx
        k2x = grad_roche_x(x + (dt/2) * vx, y + (dt/2) * vy) + 2 * Omega * (vy + (dt/2) * k1y)
        k2y = grad_roche_y(x + (dt/2) * vx, y + (dt/2) * vy) - 2 * Omega * (vx + (dt/2) * k1x)
        k3x = grad_roche_x(x + (dt/2) * vx + ((dt**2)/4) * k1x, y + (dt/2) * vy + ((dt**2)/4) * k1y) + 2 * Omega * (vy + (dt/2) * k2y)
        k3y = grad_roche_y(x + (dt/2) * vx + ((dt**2)/4) * k1x, y + (dt/2) * vy + ((dt**2)/4) * k1y) - 2 * Omega * (vx + (dt/2) * k2x)
        k4x = grad_roche_x(x + dt * vx + ((dt**2)/2) * k2x, y + dt * vy + ((dt**2)/2) * k2y) + 2 * Omega * (vy + dt * k3y)
        k4y = grad_roche_y(x + dt * vx + ((dt**2)/2) * k2x, y + dt * vy + ((dt**2)/2) * k2y) - 2 * Omega * (vx + dt * k3x)

        x = x + dt * vx + ((dt**2)/6) * (k1x + k2x + k3x)
        y = y + dt * vy + ((dt**2)/6) * (k1y + k2y + k3y)
        vx = vx + (dt/6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vy = vy + (dt/6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        # print("x = " + str(x) + " y = " + str(y))
        if np.sqrt(x**2 + y**2) > size:
            if dbg: print("Distance is too large, stop")
            break

        position_list.append((x, y))
        jacobi_cst.append((-2) * float(roche_potential(x, y)) - (vx**2 + vy**2))
    
    return position_list, jacobi_cst

def scatter_ellipsis(c_s, x_L1):
    B = (1/2) * ((G * M1)/pow(abs(x_L1 - a1), 3) + (G * M2)/pow(abs(x_L1 - a2), 3) - (Omega**2))
    C = (1/2) * ((G * M1)/pow(abs(x_L1 - a1), 3) + (G * M2)/pow(abs(x_L1 - a2), 3))
    E = (1/2) * (c_s**2)
    b = np.sqrt(E/B)
    c = np.sqrt(E/C)
    y = 2 * b * random()
    z = 2 * c * random()
    y = y - b
    z = z - c
    while (y**2)/(b**2) + (z**2)/(c**2) > 1:
        y = 2 * b * random()
        z = 2 * c * random()
        y = y - b
        z = z - c
    
    return y, z

def rk4_step(x, y, vx, vy, dt0, force_sync = False):
    norm0 = epsilon
    norm = np.sqrt(vx**2 + vy**2 + epsilon**2)
    dt = dt0 * (norm0/norm)
    # Used to keep up with real time and sync all particles
    if force_sync:
        # print("Force sync enabled")
        dt = dt0
    k1x = grad_roche_x(x, y) + 2 * Omega * vy
    k1y = grad_roche_y(x, y) - 2 * Omega * vx
    k2x = grad_roche_x(x + (dt/2) * vx, y + (dt/2) * vy) + 2 * Omega * (vy + (dt/2) * k1y)
    k2y = grad_roche_y(x + (dt/2) * vx, y + (dt/2) * vy) - 2 * Omega * (vx + (dt/2) * k1x)
    k3x = grad_roche_x(x + (dt/2) * vx + ((dt**2)/4) * k1x, y + (dt/2) * vy + ((dt**2)/4) * k1y) + 2 * Omega * (vy + (dt/2) * k2y)
    k3y = grad_roche_y(x + (dt/2) * vx + ((dt**2)/4) * k1x, y + (dt/2) * vy + ((dt**2)/4) * k1y) - 2 * Omega * (vx + (dt/2) * k2x)
    k4x = grad_roche_x(x + dt * vx + ((dt**2)/2) * k2x, y + dt * vy + ((dt**2)/2) * k2y) + 2 * Omega * (vy + dt * k3y)
    k4y = grad_roche_y(x + dt * vx + ((dt**2)/2) * k2x, y + dt * vy + ((dt**2)/2) * k2y) - 2 * Omega * (vx + dt * k3x)

    x = x + dt * vx + ((dt**2)/6) * (k1x + k2x + k3x)
    y = y + dt * vy + ((dt**2)/6) * (k1y + k2y + k3y)
    vx = vx + (dt/6) * (k1x + 2 * k2x + 2 * k3x + k4x)
    vy = vy + (dt/6) * (k1y + 2 * k2y + 2 * k3y + k4y)
    
    jacobi_cst = (-2) * float(roche_potential(x, y)) - (vx**2 + vy**2)

    return (x, y), (vx, vy), dt, jacobi_cst

def simulate_flux(n0, x_L1, v, noise, time, dt_real, dt0, reinject = False):
    t = 0
    step = 0
    particles = [[]]
    while t < time:
        print("Simulating t = " + str(t))
        if reinject or step == 0:
            for _ in range(0, n0):
                y, z = scatter_ellipsis(v, x_L1)
                vx = v * abs(np.random.normal(0, 0.2)) + v
                vy = v * np.random.normal(0, 0.2)
                vz = v * np.random.normal(0, 0.2)
                particles[step].append((x_L1, y, z, t, vx, vy, vz))
        particles.append([])
        print("There is " + str(len(particles[step])) + "particles")
        for part in particles[step]:
            x0 = part[0]
            y0 = part[1]
            z = part[2]
            vx0 = part[4]
            vy0 = part[5]
            vz = part[6]
            t_part = part[3]
            last_dt = 0
            while t_part < t + dt_real:
                if np.sqrt(x0**2 + y0**2) > size:
                    if dbg: print("Distance is too large, stop")
                    break
                (x, y), (vx, vy), dt, j = rk4_step(x0, y0, vx0, vy0, dt0)
                if t_part + dt > t + dt_real:
                    (x, y), (vx, vy), dt, j = rk4_step(x0, y0, vx0, vy0, t + dt_real - t_part, True)
                x0 = x
                y0 = y
                vx0 = vx
                vy0 = vy
                last_dt = dt
                t_part = t_part + dt
            particles[step+1].append((x0, y0, z, t + dt_real, vx0, vy0, vz))
        
        t = t + dt_real
        step = step + 1

    return particles



def test_particule_RK4_adaptative(x0, y0, vx0, vy0, step, dt0):
    start = time.time()
    """
    Uses RK4 method (Runge-Kutta order 4) to compute the trajectory of a particule.
    It is more accurate than Euler method.
    The timestep is adaptative (can be smaller or greater than dt0 when the simulation requires it)
    
    :param x0: Starting position x
    :param y0: Starting position y
    :param vx0: Starting velocity vx
    :param vy0: Starting velocity vy
    :param step: Number of simulation step. Because the timestep is adaptative, it is not equivalent to duration/dt for other methods
    :param dt0: Base timestep, which will be adapted based on the particule velocity
    :returns: Position list along the trajectory and Jacobi constant (which should be conserved along the trajectory)
    """
    total_time = 0
    x = x0
    y = y0
    vx = vx0
    vy = vy0
    norm0 = np.sqrt(vx0**2 + vy0**2 + epsilon**2)
    norm = np.sqrt(vx**2 + vy**2 + epsilon**2)
    position_list = [(x0, y0)]
    jacobi_cst = [(-2) * float(roche_potential(x, y)) - (vx**2 + vy**2)]
    for _ in range(0, step):
        dt = dt0 * (norm0/norm)
        k1x = grad_roche_x(x, y) + 2 * Omega * vy
        k1y = grad_roche_y(x, y) - 2 * Omega * vx
        k2x = grad_roche_x(x + (dt/2) * vx, y + (dt/2) * vy) + 2 * Omega * (vy + (dt/2) * k1y)
        k2y = grad_roche_y(x + (dt/2) * vx, y + (dt/2) * vy) - 2 * Omega * (vx + (dt/2) * k1x)
        k3x = grad_roche_x(x + (dt/2) * vx + ((dt**2)/4) * k1x, y + (dt/2) * vy + ((dt**2)/4) * k1y) + 2 * Omega * (vy + (dt/2) * k2y)
        k3y = grad_roche_y(x + (dt/2) * vx + ((dt**2)/4) * k1x, y + (dt/2) * vy + ((dt**2)/4) * k1y) - 2 * Omega * (vx + (dt/2) * k2x)
        k4x = grad_roche_x(x + dt * vx + ((dt**2)/2) * k2x, y + dt * vy + ((dt**2)/2) * k2y) + 2 * Omega * (vy + dt * k3y)
        k4y = grad_roche_y(x + dt * vx + ((dt**2)/2) * k2x, y + dt * vy + ((dt**2)/2) * k2y) - 2 * Omega * (vx + dt * k3x)

        x = x + dt * vx + ((dt**2)/6) * (k1x + k2x + k3x)
        y = y + dt * vy + ((dt**2)/6) * (k1y + k2y + k3y)
        vx = vx + (dt/6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vy = vy + (dt/6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        norm = np.sqrt(vx**2 + vy**2 + epsilon**2)
        # print("x = " + str(x) + " y = " + str(y))
        if np.sqrt(x**2 + y**2) > size:
            if dbg: print("Distance is too large, stop")
            break

        total_time += dt
        position_list.append((x, y))
        jacobi_cst.append((-2) * float(roche_potential(x, y)) - (vx**2 + vy**2))
    
    end = time.time()
    if dbg: print("Elapsed time : " + str(end - start) + "s")
    return position_list, jacobi_cst

def roche_potential_expansion_L1(x_L1, X, Y, Z=0):
    PhiRocheL1 = roche_potential(x_L1, 0)
    PartialX2 = (- (G * M1)/pow(abs(x_L1-a1), 3) - (G * M2)/pow(abs(x_L1-a2), 3) - (1/2) * (Omega**2))
    PartialY2 = (1/2) * ((G * M1)/pow(abs(x_L1-a1), 3) + (G * M2)/pow(abs(x_L1-a2), 3) - (Omega**2))
    PartialZ2 = (1/2) * ((G * M1)/pow((x_L1-a1), 3) + (G * M2)/pow((x_L1-a2), 3))
    return PhiRocheL1 + PartialX2 * ((x_L1 - X)**2) + PartialY2 * (Y**2) + PartialZ2 * (Z**2)

def wireframe_potential(ax, x_L1):
    x = np.arange(-size, size, wireframe_delta)
    y = np.arange(-size, size, wireframe_delta)
    X, Y = np.meshgrid(x, y)
    Z = roche_potential(X, Y)
    z_limit = -2
    Z = np.where(Z > z_limit, Z, z_limit)
    r_limit = a/6
    u = np.arange(-size, size, wireframe_delta/2)
    v = np.arange(-size, size, wireframe_delta/2)
    U, V = np.meshgrid(u, v)
    X2 = np.where(abs(x_L1 - U) < r_limit, U, r_limit)
    Y2 = np.where(abs(V) < r_limit, V, r_limit)
    Z2 = roche_potential_expansion_L1(x_L1, X2, Y2)
    Z2 = np.where(Z2 > z_limit, Z2, z_limit)
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
    ax.plot_surface(X2, Y2, Z2, edgecolor='red', lw=0.5, rstride=8, cstride=8, alpha=0.3)

def potential_slice(ax, x_L1):
    y = np.arange(-size/5, size/5, delta/5)
    dx = a/100
    ax.plot(y, roche_potential(x_L1, y))
    ax.plot(y, roche_potential_expansion_L1(x_L1, x_L1, y))
    ax.plot(y, roche_potential(x_L1 + 4 * dx, y))
    ax.plot(y, roche_potential_expansion_L1(x_L1, x_L1 + 4 * dx, y))

def yz_potential_slice(ax: axes3d.Axes3D, x_L1):
    scale = 5
    y = np.arange(-size/scale, size/scale, wireframe_delta/scale)
    z = np.arange(-size/scale, size/scale, wireframe_delta/scale)
    Y, Z = np.meshgrid(y, z)
    V = roche_potential_expansion_L1(x_L1, x_L1, Y, Z)
    # v_limit = -2
    # V = np.where(Z > v_limit, Z, v_limit)
    ax.plot_surface(Y, Z, V, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
    ax.set_xlabel("y")
    ax.set_ylabel("z")

if __name__ == "__main__":
    x = np.arange(-size, size, delta)
    y = np.arange(-size, size, delta)
    X, Y = np.meshgrid(x, y)
    R1 = np.sqrt((X - a1)**2 + Y**2)
    R2 = np.sqrt((X - a2)**2 + Y**2)
    Z = roche_potential(X, Y)
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(0, 0, s=60, marker=".")

    x_L1, y_L1 = newton_method(0, 0, 10)
    x_L2, y_L2 = newton_method(2 * a2, 0, 10)
    x_L3, y_L3 = newton_method(2 * a1, 0, 10)

    print("x_L1 = " + str(x_L1) + " y_L1 = " + str(y_L1))
    print("x_L2 = " + str(x_L2) + " y_L2 = " + str(y_L2))
    print("x_L3 = " + str(x_L3) + " y_L3 = " + str(y_L3))

    x_L4, y_L4 = newton_method(0, a, 100)
    x_L5, y_L5 = newton_method(0, -a, 100)

    ax.scatter(x_L1, y_L1, s=40, marker="+", c=[(1, 0, 0)])
    ax.scatter(x_L2, y_L2, s=40, marker="+", c=[(1, 0, 0)])
    ax.scatter(x_L3, y_L3, s=40, marker="+", c=[(1, 0, 0)])
    ax.scatter(x_L4, y_L4, s=40, marker="+", c=[(1, 0, 0)])
    ax.scatter(x_L5, y_L5, s=40, marker="+", c=[(1, 0, 0)])

    potential_slice(ax3, x_L1)
    wireframe_potential(ax2, x_L1)
    yz_potential_slice(ax4, x_L1)

    ax.scatter(a1, 0, s=60, marker=(5, 1))
    ax.scatter(a2, 0, s=60, marker=(5, 1))
    zmin, zmax = np.percentile(Z, [1, 99])
    levels = np.geomspace(zmin, zmax, 10)
    CS = ax.contour(X, Y, Z, levels=levels)

    u = np.arange(-size, size, vector_field_delta)
    v = np.arange(-size, size, vector_field_delta)
    U, V = np.meshgrid(u, v)
    Partial_U, Partial_V = grad_roche_potential(U, V)
    Norm = np.hypot(Partial_U, Partial_V)
    Partial_U = np.where(abs(Norm) < 0.1, Partial_U, 0)
    Partial_V = np.where(abs(Norm) < 0.1, Partial_V, 0)
    quiver = ax.quiver(U, V, Partial_U, Partial_V, angles='xy', scale_units='xy', scale=0.1)

    ax.set_title("Roche equipotentials for M=" + str(M) + ", q=" + str(q))

    # Generate trajectory
    pos_list, jacobi_cst = test_particule_RK4_adaptative(x_L1, y_L1 - 0.001 * a, 0, 0, 1000000, 0.01)
    print("min = " + str(min(jacobi_cst)) + " max = " + str(max(jacobi_cst)))

    # pos_list = read_pos_from_file("pos.txt") # Read trajectory generated from the Rust code

    point_number = 5000
    step = int((len(pos_list) - 1)/point_number)
    for i in range(0, point_number + 1):
        xp, yp = pos_list[i * step]
        ax.scatter(xp, yp, s=5, marker=".", c=[(0, 0, 1)])

    fig5, ax5 = plt.subplots()
    ax5.set_xlabel("y")
    ax5.set_ylabel("z")
    for _ in range(0, 100):
        y, z = scatter_ellipsis(1, x_L1)
        ax5.scatter(y, z, s=40, marker="+", c=[(1, 0, 0)])

    # particles = simulate_flux(100, x_L1, epsilon, 0, 100, 0.1, 0.1)

    # x_all = []
    # y_all = []

    # for bundle in particles:
    #     for part in bundle:
    #         x = part[0]
    #         y = part[1]
    #         x_all.append(x)
    #         y_all.append(y)
    #         # ax.scatter(x, y, s=5, marker=".", c=[(0, 0, 1)])

    # np.histogram2d(x_all, y_all)

    # H, xedges, yedges = np.histogram2d(x_all, y_all, bins=200)
    # ax.imshow(
    #     np.log(H.T + 1),
    #     origin='lower',
    #     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    #     aspect='equal'
    # )

    plt.show()
