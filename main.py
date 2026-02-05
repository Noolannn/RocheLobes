import matplotlib.pyplot as plt
import numpy as np

# Physical constant
G = 0.1

# Simulation parameters
dbg = True # Optional output info for debug
delta = 0.01 # Grid step
vector_field_delta = 0.5 # Should be smaller than delta
epsilon = 1 # Used for regularization near singularities
diff_step = 0.01 # Used for differentiation
error = 1e-6 # When an iterative method won't increase the accuracy of the result more than error, it will stop
size = 8.0

# Physical parameters
M = 10
q = 2
a = 4
Omega = 0.1

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

# Returns the 2D gradient of the Roche potential at a given point
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
    J_inv = np.linalg.inv(J)
    X_new = X - J_inv * F
    x_new = float(X_new[0][0])
    y_new = float(X_new[1][0])

    if abs(x0 - x_new) < error and abs(y0 - y_new) < error:
        if dbg: print("Return earlier")
        return x_new, y_new

    return newton_method(x_new, y_new, step-1)


if __name__ == "__main__":
    x = np.arange(-size, size, delta)
    y = np.arange(-size, size, delta)
    X, Y = np.meshgrid(x, y)
    R1 = np.sqrt((X - a1)**2 + Y**2)
    Z = roche_potential(X, Y)
    fig, ax = plt.subplots()
    ax.scatter(0, 0, s=60, marker=".")

    x_L1, y_L1 = newton_method(0, 0, 10)
    x_L2, y_L2 = newton_method(2 * a2, 0, 10)
    x_L3, y_L3 = newton_method(2 * a1, 0, 10)

    print("x_L1 = " + str(x_L1) + " y_L1 = " + str(y_L1))
    print("x_L2 = " + str(x_L2) + " y_L2 = " + str(y_L2))
    print("x_L3 = " + str(x_L3) + " y_L3 = " + str(y_L3))

    x_L4, y_L4 = newton_method(0, a, 100)

    for xp, yp in point_buffer:
        ax.scatter(xp, yp, s=40, marker="+", c=[(1, 0, 0)])

    ax.scatter(a1, 0, s=60, marker=(5, 1))
    ax.scatter(a2, 0, s=60, marker=(5, 1))
    CS = ax.contour(X, Y, Z, 10)

    u = np.arange(-size, size, vector_field_delta)
    v = np.arange(-size, size, vector_field_delta)
    U, V = np.meshgrid(u, v)
    Partial_U, Partial_V = grad_roche_potential(U, V)
    quiver = ax.quiver(U, V, Partial_U, Partial_V)

    ax.set_title("Roche equipotentials for M=" + str(M) + ", q=" + str(q))

    plt.show()

    
