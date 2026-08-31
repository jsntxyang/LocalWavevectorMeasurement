import numpy as np 
from LatticeGenerateFunction import LatticeGenerate
import findiff
from DisplacementFunctions import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    L0 = 30
    N0 = 1024
    R1 = np.array([0.25, 0.0])
    R2 = np.array([0.0, 0.25])
    
    
    
    I, poly_disp_x, poly_disp_y, rand_disp_x, rand_disp_y= LatticeGenerate(R1, R2, L=L0, N=N0, 
                                                                           ux=FUx, uy=FUy, 
                                                                           noise_level=0.0, strain_level=0.02,
                                              )
    x = np.linspace(0, L0, N0)
    y = np.linspace(0, L0, N0)
    
    I = 2 * (I - np.min(I)) / (np.max(I) - np.min(I)) - 1.0
    noise = np.random.uniform(-0.25, 0.25, size=I.shape)
    I = I + noise
    
    generated_data = {'I0': I,
                      'U_dr_x': poly_disp_x,
                      'U_dr_y': poly_disp_y,
                      'U_ph_x': rand_disp_x,
                      'U_ph_y': rand_disp_y,
                      'noise': noise,
                      'R1': R1,
                      'R2': R2,
                      'L0': L0,
                     }
    
    
    np.savez('./Generated_Lattice_Data.npz', **generated_data)
    
    Dx = findiff.FinDiff(1, L0 / (N0 - 1))
    
    eps_disp_x = Dx(rand_disp_x)
    eps_disp_y = Dx(rand_disp_y)
    
    
    fig_lattice = plt.figure(figsize=(6, 6))
    ax_lattice = fig_lattice.add_subplot(111)
    
    fig_disp = plt.figure(figsize=(6, 9))
    ax_disp_x = fig_disp.add_subplot(325)
    ax_disp_y = fig_disp.add_subplot(326)
    ax_ph_x = fig_disp.add_subplot(323)
    ax_ph_y = fig_disp.add_subplot(324)
    ax_poly_x = fig_disp.add_subplot(321)
    ax_poly_y = fig_disp.add_subplot(322)
    
    ax_lattice.pcolorfast(x, y, I, cmap='gray_r', vmin=-1.0, vmax=1.0)
    ax_lattice.set_title('Generated Lattice Image')
    ax_lattice.set_xlabel('x (nm)')
    ax_lattice.set_ylabel('y (nm)')
    
    
    
    ax_disp_x.pcolorfast(x, y, rand_disp_x+poly_disp_x, cmap='plasma')
    ax_disp_x.set_title('Total Displacement x')
    ax_disp_x.set_xlabel('x (nm)')
    ax_disp_x.set_ylabel('y (nm)')
    ax_disp_x.set_aspect('equal')
    
    ax_disp_y.pcolorfast(x, y, rand_disp_y+poly_disp_y, cmap='plasma')
    ax_disp_y.set_title('Total Displacement y')
    ax_disp_y.set_xlabel('x (nm)')
    ax_disp_y.set_ylabel('y (nm)')
    ax_disp_y.set_aspect('equal')
    
    ax_ph_x.pcolorfast(x, y, rand_disp_x, cmap='plasma')
    ax_ph_x.set_title('Random Displacement x')
    ax_ph_x.set_xlabel('x (nm)')
    ax_ph_x.set_ylabel('y (nm)')
    ax_ph_x.set_aspect('equal')
    
    ax_ph_y.pcolorfast(x, y, rand_disp_y, cmap='plasma')
    ax_ph_y.set_title('Random Displacement y')
    ax_ph_y.set_xlabel('x (nm)')
    ax_ph_y.set_ylabel('y (nm)')
    ax_ph_y.set_aspect('equal') 
    
    ax_poly_x.pcolorfast(x, y, poly_disp_x, cmap='plasma')
    ax_poly_x.set_title('Polynomial Displacement x')
    ax_poly_x.set_xlabel('x (nm)')
    ax_poly_x.set_ylabel('y (nm)')
    ax_poly_x.set_aspect('equal')
    
    ax_poly_y.pcolorfast(x, y, poly_disp_y, cmap='plasma')
    ax_poly_y.set_title('Polynomial Displacement y')
    ax_poly_y.set_xlabel('x (nm)')
    ax_poly_y.set_ylabel('y (nm)')
    ax_poly_y.set_aspect('equal')

    plt.show()