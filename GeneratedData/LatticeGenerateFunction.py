import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator


# Fix random seed for reproducibility
np.random.seed(42)

def LatticeGenerate(R1, R2, L=30, N=1024, 
                    ux=None, uy=None, 
                    noise_level=0.0, 
                    strain_level=0.0,
                    ):
    
    N_edge = 25
    N_large = 250
    def random_strain_kernel():
        random_kernel = np.random.randn(N_large, N_large)
        strain_kernel = convolve2d(random_kernel, np.ones((N_edge, N_edge)), mode='same')
        strain_kernel = convolve2d(strain_kernel, np.ones((N_edge, N_edge)), mode='same')
        strain_kernel = convolve2d(strain_kernel, np.ones((N_edge, N_edge)), mode='same')
        strain_kernel = convolve2d(strain_kernel, np.ones((N_edge, N_edge)), mode='valid')
        
        return strain_kernel
    
    x_kernel = np.linspace(0, L, N_large - N_edge + 1)
    y_kernel = np.linspace(0, L, N_large - N_edge + 1)
    fx_interp = RegularGridInterpolator((x_kernel, y_kernel), random_strain_kernel())
    fy_interp = RegularGridInterpolator((x_kernel, y_kernel), random_strain_kernel())
    
    
    
    x_vals = np.linspace(0, L, N)
    y_vals = np.linspace(0, L, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    random_strain_x = fx_interp((X, Y))
    random_strain_x = (random_strain_x - np.mean(random_strain_x)) / np.std(random_strain_x)
    random_strain_y = fy_interp((X, Y))
    random_strain_y = (random_strain_y - np.mean(random_strain_y)) / np.std(random_strain_y)

    # No strain by default
    if ux is None:
        def ux(x, y):
            return np.zeros_like(x)

    if uy is None:
        def uy(x, y):
            return np.zeros_like(x)

    # Apply strain 
    poly_disp_x = ux(X, Y)
    poly_disp_y = uy(X, Y)
    
    rand_disp_x = random_strain_x * strain_level
    rand_disp_y = random_strain_y * strain_level
    
    total_disp_x = poly_disp_x + rand_disp_x
    total_disp_y = poly_disp_y + rand_disp_y
    X_strain = X - total_disp_x
    Y_strain = Y - total_disp_y

    # Calculate Q1, Q2
    v = abs(R1[0] * R2[1] - R1[1] * R2[0])
    Q1 = np.array([R2[1], -R2[0]]) / v * np.pi * 2
    Q2 = -np.array([R1[1], -R1[0]]) / v * np.pi * 2

    # Calculate the lattice image using Q1, Q2 and plane waves
    I = np.exp(1j * (Q1[0] * X_strain + Q1[1] * Y_strain)) + \
        np.exp(1j * (Q2[0] * X_strain + Q2[1] * Y_strain))
        # 0.5 * np.exp(1j * (2 * Q1[0] * X_strain + 2 * Q1[1] * Y_strain)) + \
        # 0.5 * np.exp(1j * (2 * Q2[0] * X_strain + 2 * Q2[1] * Y_strain)) - \
        # 0.2 * np.exp(1j * ((Q1[0] + Q2[0]) * X_strain + (Q1[1] + Q2[1]) * Y_strain))

    # Take real part
    I = np.real(I)

    # Add noise 
    if noise_level > 0:
        I += noise_level * np.random.uniform(-1, 1, (N, N))
    return I, poly_disp_x, poly_disp_y, rand_disp_x, rand_disp_y