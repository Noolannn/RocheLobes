import matplotlib.pyplot as plt
import numpy as np

# Physical constant
G = 0.1

# Simulation parameters
delta = 0.01
epsilon = 1
size = 8.0

# Physical parameters
M = 10
q = 2
a1 = 2
Omega = 0.1

# Constraint physical parameters
M1 = M/(1 + q)
M2 = M - M1
a2 = - a1/q

def roche_potential(X, Y):
    R = np.sqrt(X**2 + Y**2)
    R1 = np.sqrt((X - a1)**2 + Y**2)
    R2 = np.sqrt((X - a2)**2 + Y**2)
    # Adds epsilon to avoid the singularity at the star position
    Phi1 = - (G * M1)/(R1 + epsilon)
    Phi2 = - (G * M2)/(R2 + epsilon)
    PhiC = (1/2) * (Omega**2) * (R**2)
    PhiRoche = Phi1 + Phi2 - PhiC
    return PhiRoche

if __name__ == "__main__":
    x = np.arange(-size, size, delta)
    y = np.arange(-size, size, delta)
    X, Y = np.meshgrid(x, y)
    R1 = np.sqrt((X - a1)**2 + Y**2)
    Z = roche_potential(X, Y)
    fig, ax = plt.subplots()
    ax.scatter(0, 0, s=60, marker=".")
    ax.scatter(a1, 0, s=60, marker=(5, 1))
    ax.scatter(a2, 0, s=60, marker=(5, 1))
    CS = ax.contour(X, Y, Z, 20)
    ax.set_title("Roche equipotentials for M=" + str(M) + ", q=" + str(q))
    plt.show()
