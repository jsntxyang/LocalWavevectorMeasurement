import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib import patches


def Q_module(I, Q, Lx=None, Ly=None):
    '''
    Modulate the image by the wave exp(i * r * Q)
    
    I  (ndarray) -- Original image
    Q  (2D vector) -- Q vector of the modulation
    Lx (number) --- width of the image
    Ly (number) --- height of the image
    '''
    
    if Lx is None:
        Lx = 1.0
    if Ly is None:
        Ly = 1.0
        
    Nx = I.shape[1]
    Ny = I.shape[0]
    
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    
    MOD = np.exp(1.0j * (X * Q[0] + Y * Q[1]))
    
    return I * MOD


def Q_filter(I, Q, Lx=None, Ly=None, Q_width=None, alpha=None,
             plot_figure=False, ax_filter=None):
    
    '''
    Filter out the modulation with vector Q from the image I
    This function get real space image and output filtered real space image
    
    I  (ndarray)--- original image
    Q  (2D vector)--- q vector of the mudulation
    Lx (number) --- width of the image
    Ly (number) --- height of the image
    Q_width (number) --- width of the Q filter
    alpha (number) --- Sharpness of the edge of the filter
    '''
    
    if Lx is None:
        Lx = 1.0
    if Ly is None:
        Ly = 1.0
        
    if Q_width is None:
        Q_width = 0.25 * np.sqrt(Q[0]**2 + Q[1]**2)
    
    Nx = I.shape[1]
    Ny = I.shape[0]
    
    kx = np.linspace(-np.pi/Lx * Nx, np.pi/Lx * Nx, Nx)
    ky = np.linspace(-np.pi/Ly * Ny, np.pi/Ly * Ny, Ny)
    KX, KY = np.meshgrid(kx, ky)
    
    # Create a mask (Gaussial if alpha is None. Otherwise use Fermi-Dirac destribution)
    R = np.sqrt((KX - Q[0])**2 + (KY - Q[1])**2)
    if alpha is None:
        MASK = np.exp(-(R / Q_width)**2 / 2)
    else:
        MASK = 1 / (np.exp(alpha * (R / Q_width - 1)) + 1)
    
    # Do fft and apply the mask
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    I_fft_filtered = I_fft * MASK
    I_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(I_fft_filtered)))
    
    # Plot figures if required by the user
    if plot_figure and not (ax_filter is None):
        cmask = patches.Circle(Q, Q_width, linewidth=1, edgecolor='r', facecolor='none')
        ax_filter.pcolor(kx, ky, abs(I_fft_filtered), vmin=0.0, vmax=0.5 * np.max(abs(I_fft)))
        ax_filter.add_patch(cmask)

    return I_filtered


def interp(I, N_interp=2):
    I_interp = zoom(I, N_interp, order=4)
    return I_interp


def Q_mask(I, Q, Lx=None, Ly=None, Q_width=None, alpha=10.0):
    
    '''
    Filter out the modulation with vector Q from the image I
    This function get real space image and output filtered fft.
    
    I  (ndarray)--- original image
    Q  (2D vector)--- q vector of the mudulation
    Lx (number) --- width of the image
    Ly (number) --- height of the image
    Q_width (number) --- width of the Q filter
    alpha (number) --- Sharpness of the edge of the filter
    '''
    
    if Lx is None:
        Lx = 1.0
    if Ly is None:
        Ly = 1.0
        
    if Q_width is None:
        Q_width = 0.25 * np.sqrt(Q[0]**2 + Q[1]**2)
    
    Nx = I.shape[1]
    Ny = I.shape[0]
    
    kx = np.linspace(-np.pi/Lx * Nx, np.pi/Lx * Nx, Nx)
    ky = np.linspace(-np.pi/Ly * Ny, np.pi/Ly * Ny, Ny)
    KX, KY = np.meshgrid(kx, ky)
    
    # Create a mask (Fermi-Dirac destribution)
    R = np.sqrt((KX - Q[0])**2 + (KY - Q[1])**2)
    MASK = 1 / (np.exp(alpha * (R / Q_width - 1)) + 1)
    
    # Do fft and apply the mask
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    I_fft_filtered = I_fft * MASK
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.pcolorfast(kx, ky, abs(I_fft_filtered), cmap='gray_r')
    ax2.pcolorfast(kx, ky, abs(I_fft), cmap='gray_r')
    plt.show()
    
    return I_fft_filtered


def pretreat(I, L_image, Q1, Q2, Q1_calc1, Q1_calc2, Q2_calc1, Q2_calc2, Q_width=None, alpha=None, N_interp=2, plot_figure=False):
    
    # Interpolate the image
    I_int = I
    
    
    dQ11 = Q1_calc1 - Q1
    dQ12 = Q1_calc2 - Q1
    dQ21 = Q2_calc1 - Q2
    dQ22 = Q2_calc2 - Q2

    # Modulate images
    I_Q1_calc1 = Q_module(I_int, Lx=L_image, Ly=L_image, Q=dQ11)
    I_Q1_calc2 = Q_module(I_int, Lx=L_image, Ly=L_image, Q=dQ12)
    I_Q2_calc1 = Q_module(I_int, Lx=L_image, Ly=L_image, Q=dQ21)
    I_Q2_calc2 = Q_module(I_int, Lx=L_image, Ly=L_image, Q=dQ22)
    
    
    # Filter images
    I_Q1_calc1_filtered = Q_filter(I_Q1_calc1, Lx=L_image, Ly=L_image, 
                                   Q=Q1_calc1, Q_width=Q_width, alpha=alpha)
    I_Q1_calc2_filtered = Q_filter(I_Q1_calc2, Lx=L_image, Ly=L_image, 
                                   Q=Q1_calc2, Q_width=Q_width, alpha=alpha)
    I_Q2_calc1_filtered = Q_filter(I_Q2_calc1, Lx=L_image, Ly=L_image, 
                                   Q=Q2_calc1, Q_width=Q_width, alpha=alpha)
    I_Q2_calc2_filtered = Q_filter(I_Q2_calc2, Lx=L_image, Ly=L_image, 
                                   Q=Q2_calc2, Q_width=Q_width, alpha=alpha)
    
    
    I_Q1_calc1_filtered = interp(I_Q1_calc1_filtered, N_interp=N_interp)
    I_Q1_calc2_filtered = interp(I_Q1_calc2_filtered, N_interp=N_interp)
    I_Q2_calc1_filtered = interp(I_Q2_calc1_filtered, N_interp=N_interp)
    I_Q2_calc2_filtered = interp(I_Q2_calc2_filtered, N_interp=N_interp)
    
    plt.matshow(I_Q1_calc1_filtered)
    plt.show()
    

    return I_Q1_calc1_filtered, I_Q1_calc2_filtered, I_Q2_calc1_filtered, I_Q2_calc2_filtered



def preprocessing(I, L_image, Q1, Q2, 
                  Q1_calc, Q2_calc, 
                  Q_width=None, alpha=None, N_interp=2):
    
    Nx = I.shape[1]
    Ny = I.shape[0]
    N_q = Q1_calc.shape[0]
    
    dQ1 = Q1_calc - np.reshape(Q1, (1, 2))
    dQ2 = Q2_calc - np.reshape(Q2, (1, 2))
    I_calc_Q1 = np.zeros((N_q, round(Ny * N_interp), round(Nx * N_interp)))
    I_calc_Q2 = np.zeros((N_q, round(Ny * N_interp), round(Nx * N_interp)))
    
    for i in range(N_q):
        
        I_Q1_calc = Q_module(I, Lx=L_image, Ly=L_image, Q=dQ1[i])
        
        I_Q1_calc_filtered = Q_filter(I_Q1_calc, Lx=L_image, Ly=L_image, 
                                      Q=Q1_calc[i], Q_width=Q_width, alpha=alpha)
        # I_Q1_calc_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(I_Q1_calc_filtered)))
        I_Q1_calc_filtered = interp(I_Q1_calc_filtered, N_interp=N_interp)
        
        I_Q2_calc = Q_module(I, Lx=L_image, Ly=L_image, Q=dQ2[i])
        I_Q2_calc_filtered = Q_filter(I_Q2_calc, Lx=L_image, Ly=L_image, 
                                       Q=Q2_calc[i], Q_width=Q_width, alpha=alpha)
        # I_Q2_calc_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(I_Q2_calc_filtered)))
        I_Q2_calc_filtered = interp(I_Q2_calc_filtered, N_interp=N_interp)
        
        I_calc_Q1[i] = I_Q1_calc_filtered
        I_calc_Q2[i] = I_Q2_calc_filtered
        
        
    return I_calc_Q1, I_calc_Q2
