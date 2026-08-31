import numpy as np
import numbers, warnings
from Pretreatment import preprocessing, pretreat
from skimage import feature
from HoughAngle import hough_angle
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import matplotlib.pyplot as plt

class LocalWavevectorMap:
    def __init__(self,  I: np.array, L_image, 
                    L_region=None, N_region=None, L_mask=None, x0=None, y0=None,
                    ):
            
        '''
        Valrable Naming Rule:
        
        region -- A variable with the word 'region' is related to the region where the strain calculation is conducted.
        mask ---- A variable with the word 'mask' is related to the local mask.
        image --- A variable with the word 'image' is related to the original image.
        
        '''
        
        '''
        Input:
        
        I (2D array) -- Original Image
        L_image (folat) -- Image Size
        L_region (array or float) -- Size of the region to be considered
        N_region (array or int) -- Sampling number of the region
        L_mask (array or float) -- Size of the mask for local image
        x0 (float) -- the x coordinate of the region
        y0 (float) -- the y coordinate of the region
        '''
        
        ##############
        # Set Original Image Parameters
        ##############
        
        # Get Image and Image Size
        self.I = I
        self.L_image = L_image
        
        Nx = self.I.shape[0]
        Ny = self.I.shape[1]
        if Nx != Ny:
            raise(ValueError('The image must have same x and y pixels.'))
        else:
            self.N_image = Nx
        
        self.x_image = np.linspace(0, self.L_image, self.N_image)
        self.y_image = np.linspace(0, self.L_image, self.N_image)
        self.X_image, self.Y_image = np.meshgrid(self.x_image, self.y_image)
        
        
        ##############
        # Set Mask parameters
        ##############
        
        # Set mask size
        # If user didn't specified, the mask size is chosen to be L_image / 10
        if L_mask is None:
            L_mask = 0.1 * self.L_image
            self.L_mask = np.array([L_mask, L_mask])
        elif isinstance(L_mask, numbers.Number): # If user specify only a number
            self.L_mask = np.array([L_mask, L_mask])
        else: # If user specify Lx and Ly
            self.L_mask = np.array(L_mask)
            
        # Round the mask size to ensure it is a integer numbers of pixel sizes.
        N_half_mask = np.round(0.5 * self.L_mask / L_image * self.N_image)
        self.N_mask = 2 * N_half_mask
        self.N_mask = self.N_mask.astype(int)
        self.L_mask = self.N_mask * self.L_image / self.N_image
        
        # r grid of the mask
        self.x_mask = np.linspace(0, self.L_mask[0], self.N_mask[0])
        self.y_mask = np.linspace(0, self.L_mask[1], self.N_mask[1])
        self.X_mask, self.Y_mask = np.meshgrid(self.x_mask, self.y_mask)
        
        # k grid of the mask
        self.kx_mask = np.linspace(-np.pi/self.L_mask[0]*self.N_mask[0], np.pi/self.L_mask[0]*self.N_mask[0], self.N_mask[0])
        self.ky_mask = np.linspace(-np.pi/self.L_mask[1]*self.N_mask[1], np.pi/self.L_mask[1]*self.N_mask[1], self.N_mask[1])
        self.Kx_mask, self.Ky_mask = np.meshgrid(self.kx_mask, self.ky_mask)
        
        
        #############
        # Set Region Parameters
        #############
        
        # Set the Center Point of the Region Conduction Analysis
        # If user didn't specify, choose the center of the image.
        if x0 is None:
            self.x0 = self.L_image / 2
        else:
            self.x0 = x0
            
        if y0 is None:
            self.y0 = self.L_image / 2
        else:
            self.y0 = y0
        
        # Set the Size of the Region Conduction Analysis
        # If user didn't specify, the size will be the maximized valid size
        if L_region is None:
            self.L_region = self.L_image - self.L_mask
        elif isinstance(L_region, numbers.Number): # If user specify only a number
            self.L_region = np.array([L_region, L_region])
        else: # If user specify Lx and Ly
            self.L_region = np.array(L_region)

        # Set the Pixel of Region
        # If user didn't specify, the resolution will be about the half of the original pixel.
        if N_region is None:
            N_region = self.L_region / self.L_image * self.N_image * 0.5
            self.N_region = N_region.astype(int)
            # self.N_region = np.array([N_region, N_region], dtype=int)
        elif isinstance(N_region, numbers.Number): # If user specify only a number
            self.N_region = np.array([N_region, N_region], dtype=int)
        else: # If user specify Lx and Ly
            self.N_region = np.array(N_region, dtype=int)

        # Create Grid
        self.x_region = np.linspace(-self.L_region[0]/2 + self.x0, self.L_region[0]/2 + self.x0, self.N_region[0]) # Grid in x
        self.y_region = np.linspace(-self.L_region[1]/2 + self.y0, self.L_region[1]/2 + self.y0, self.N_region[1]) # Grid in y
        self.X_region, self.Y_region = np.meshgrid(self.x_region, self.y_region)
        
        # Give warnning if the region where LFT to be conducted is too large.
        if (self.x0 + self.L_region[0]/2 + self.L_mask[0]/2>=1.01*self.L_image) or (self.x0 - self.L_region[0]/2 - self.L_mask[0]/2<=-0.01*self.L_image):
            warnings.warn('LFT range is either too close to the boundry of the image or even out of the frame!')

        # Give error if the center point of LFT is out of the range of the whole image.
        if (self.x0 + self.L_region[0]/2>self.L_image) or (self.x0 - self.L_region[0]/2<0):
            raise(ValueError('LFT center is out of range!'))
        if (self.y0 + self.L_region[1]/2>self.L_image) or (self.y0 - self.L_region[1]/2<0):
            raise(ValueError('LFT center is out of range!'))
        
        # Grids in the Corrected region, this Grid is the same as that in the originl image
        j_min = round((self.x0 - self.L_region[0]/2) / self.L_image * self.N_image) + 1
        j_max = round((self.x0 + self.L_region[0]/2) / self.L_image * self.N_image) - 1
        i_min = round((self.y0 - self.L_region[0]/2) / self.L_image * self.N_image) + 1
        i_max = round((self.y0 + self.L_region[0]/2) / self.L_image * self.N_image) - 1

        self.I_valid = self.I[i_min:i_max, j_min:j_max]
        self.x_valid = self.x_image[j_min:j_max]
        self.y_valid = self.y_image[i_min:i_max]

        self.X_valid, self.Y_valid = np.meshgrid(self.x_valid, self.y_valid)

    def calc_local_wavevector(self, Q1, Q2, Q_width, 
                              Q1_calc, Q2_calc,
                              alpha=None, N_interp=2, 
                              canny_sigma=1, canny_low_threshold=0.1, canny_high_threshold=0.3,
                              show_process=False,
                              th1_calc=None, th2_calc=None,
                             ):

        
        I_Q1_calc, I_Q2_calc = preprocessing(self.I, self.L_image, 
                                             Q1=Q1, Q2=Q2, 
                                             Q1_calc=Q1_calc, Q2_calc=Q2_calc, 
                                             Q_width=Q_width, alpha=alpha, N_interp=N_interp,
                                             )
         
        dQ1 = Q1_calc - np.reshape(Q1, (1, 2))
        dQ2 = Q2_calc - np.reshape(Q2, (1, 2))
        
        Q1_measured_x = np.zeros(self.N_region)
        Q1_measured_y = np.zeros(self.N_region)
        Q2_measured_x = np.zeros(self.N_region)
        Q2_measured_y = np.zeros(self.N_region)
        
        Nq = Q1_calc.shape[0]

        if show_process:
            plt.figure()
            ax1 =  plt.subplot(221)
            ax2 =  plt.subplot(222)
            ax3 =  plt.subplot(223)
            ax4 =  plt.subplot(224)
        
            ax1.pcolorfast(I_Q1_calc[0, :, :])
            ax2.pcolorfast(I_Q1_calc[1, :, :])
            ax3.pcolorfast(I_Q2_calc[0, :, :])
            ax4.pcolorfast(I_Q2_calc[1, :, :])
            
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')
            ax4.set_aspect('equal')
            plt.show()   
        
        
        for i in range(0, Nq):
            I_Q1 = I_Q1_calc[i]
            I_Q2 = I_Q2_calc[i]
            
            # Canny Edge Detection
            edges_Q1 = feature.canny(I_Q1, sigma=canny_sigma, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)
            edges_Q2 = feature.canny(I_Q2, sigma=canny_sigma, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)
        
            I_Q1_calc[i] = edges_Q1
            I_Q2_calc[i] = edges_Q2
            
        if show_process:
            plt.figure()
            ax1 =  plt.subplot(221)
            ax2 =  plt.subplot(222)
            ax3 =  plt.subplot(223)
            ax4 =  plt.subplot(224)
            
            ax1.pcolorfast(I_Q1_calc[0, :, :])
            ax2.pcolorfast(I_Q1_calc[1, :, :])
            ax3.pcolorfast(I_Q2_calc[0, :, :])
            ax4.pcolorfast(I_Q2_calc[1, :, :])

            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')
            ax4.set_aspect('equal')
            plt.show()   
        
        
        
        
        
        if show_process:
            plt.figure()
            ax1 =  plt.subplot(221)
            ax2 =  plt.subplot(222)
            ax3 =  plt.subplot(223)
            ax4 =  plt.subplot(224)
        
        
        
        angle1 = np.zeros((Nq, self.N_region[1], self.N_region[0]))
        angle2 = np.zeros((Nq, self.N_region[1], self.N_region[0]))
        
        for i in range(0, self.N_region[1]):
            for j in range(0, self.N_region[0]):
                x = self.x_region[i]
                y = self.y_region[j]
                I_Q1_local = self.get_local_image(images=I_Q1_calc, x=x, y=y, padding=False)
                I_Q2_local = self.get_local_image(images=I_Q2_calc, x=x, y=y, padding=False)
                Q1_angle_calc = hough_angle(I_Q1_local, angle=th1_calc)
                Q2_angle_calc = hough_angle(I_Q2_local, angle=th2_calc)
                
                angle1[:, i, j] = Q1_angle_calc
                angle2[:, i, j] = Q2_angle_calc
                
                if show_process:
                    ax1.pcolorfast(I_Q1_local[0, :, :])
                    ax2.pcolorfast(I_Q1_local[1, :, :])
                    ax3.pcolorfast(I_Q2_local[0, :, :])
                    ax4.pcolorfast(I_Q2_local[1, :, :])
                    
                    ax1.set_aspect('equal')
                    ax2.set_aspect('equal')
                    ax3.set_aspect('equal')
                    ax4.set_aspect('equal')
                    plt.pause(0.01)
                    plt.waitforbuttonpress()
                    ax1.clear()
                    ax2.clear()
                    ax3.clear()
                    ax4.clear() 
                
                
                C1 = np.cos(Q1_angle_calc)
                S1 = np.sin(Q1_angle_calc)
                C2 = np.cos(Q2_angle_calc)
                S2 = np.sin(Q2_angle_calc)
                
                P1 = dQ1[:, 0] * S1 - dQ1[:, 1] * C1
                P2 = dQ2[:, 0] * S2 - dQ2[:, 1] * C2
                
                M1 = np.array([-S1, C1]).T
                M2 = np.array([-S2, C2]).T
                
                Q1_calc_measured = np.linalg.lstsq(M1, P1, rcond=-1)
                Q2_calc_measured = np.linalg.lstsq(M2, P2, rcond=-1)
                
                
                Q1_measured = Q1_calc_measured[0]
                Q2_measured = Q2_calc_measured[0]

                
                Q1_measured_x[j, i] = Q1_measured[0]
                Q1_measured_y[j, i] = Q1_measured[1]
                Q2_measured_x[j, i] = Q2_measured[0]
                Q2_measured_y[j, i] = Q2_measured[1]
                
                print('Measuring pixel %5d, %5d' % (i, j), end='\r')
                
        if show_process:
            plt.figure()
            ax1 =  plt.subplot(221)
            ax2 =  plt.subplot(222)
            ax3 =  plt.subplot(223)
            ax4 =  plt.subplot(224)
            
            ax1.pcolorfast(angle1[0, :, :])
            ax2.pcolorfast(angle1[1, :, :])
            ax3.pcolorfast(angle2[0, :, :])
            ax4.pcolorfast(angle2[1, :, :])
            
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')
            ax4.set_aspect('equal')
            plt.show()
        return Q1_measured_x, Q1_measured_y, Q2_measured_x, Q2_measured_y
    
    
    def get_local_image(self, images, x, y, padding=True):
        
        '''
        Get the image within the mask.
        
        x (number) -- x coordinate of the center of the local image
        y (number) -- y coordinate of the center of the local image
        padding (bools) -- If True and the mask is out of boundry, the output image will be periodically padded.
        '''
        
        # Get Indices
        Nx_image = images.shape[2]
        Ny_image = images.shape[1]
        index_i = round(y / self.L_image * Ny_image)
        index_j = round(x / self.L_image * Nx_image)
        
        i_min = index_i - round(self.N_mask[1]/2 * Ny_image / self.N_image)
        i_max = index_i + round(self.N_mask[1]/2 * Ny_image / self.N_image)
        j_min = index_j - round(self.N_mask[0]/2 * Nx_image / self.N_image)
        j_max = index_j + round(self.N_mask[0]/2 * Nx_image / self.N_image)
        
       
        # Pad local image to fit the mask size if necessary
        pad_i_before = 0
        pad_i_after = 0
        pad_j_before = 0
        pad_j_after = 0
        need_padding = False
        # Determine whether the size of the local image is smaller than the size of the mask.
        if padding:
            if i_min < 0:
                pad_i_before = abs(i_min)
                i_min = 0
                need_padding = True
            if i_max > self.N_image:
                pad_i_after = abs(i_max - self.N_image)
                i_max = self.N_image
                need_padding = True
            if j_min < 0:
                pad_j_before = abs(j_min)
                j_min = 0
                need_padding = True
            if j_max > self.N_image:
                pad_j_after = abs(j_max - self.N_image)
                j_max = self.N_image
                need_padding = True
            pad_width = ((0, 0), (pad_i_before, pad_i_after), (pad_j_before, pad_j_after))
        
        # Get local image
        I_local = images[:, i_min:i_max, j_min:j_max]
        if need_padding and padding:
            I_local = np.pad(I_local, pad_width=pad_width, 
                             mode='wrap')
            
        return I_local
    
    
    
    
def get_strain(Q1x, Q1y, Q2x, Q2y, Q1i, Q2i, subtract_linear=True):
    N0 = Q1x.shape[0]
    N1 = Q1x.shape[1]
    
    Q_calc = np.zeros((N0, N1, 2, 2), dtype=float)
    Q_calc[:, :, 0, 0] = Q1x
    Q_calc[:, :, 0, 1] = Q1y
    Q_calc[:, :, 1, 0] = Q2x
    Q_calc[:, :, 1, 1] = Q2y
    
    Q_calc_inv = np.linalg.inv(Q_calc)
    
    Q_matrix = np.array([[Q1i[0], Q1i[1]], [Q2i[0], Q2i[1]]])
    
    exx = Q_calc_inv[:, :, 0, 0] * Q_matrix[0, 0] + Q_calc_inv[:, :, 0, 1] * Q_matrix[1, 0] - 1
    exy = Q_calc_inv[:, :, 0, 0] * Q_matrix[0, 1] + Q_calc_inv[:, :, 0, 1] * Q_matrix[1, 1]
    eyx = Q_calc_inv[:, :, 1, 0] * Q_matrix[0, 0] + Q_calc_inv[:, :, 1, 1] * Q_matrix[1, 0]
    eyy = Q_calc_inv[:, :, 1, 0] * Q_matrix[0, 1] + Q_calc_inv[:, :, 1, 1] * Q_matrix[1, 1] - 1
    
    if subtract_linear:
        exx = exx - np.average(exx)
        exy = exy - np.average(exy)
        eyx = eyx - np.average(eyx)
        eyy = eyy - np.average(eyy)
    
        
    return exx, exy, eyx, eyy




def get_displacement(exx, exy, eyx, eyy, Lx, Ly):
    
    Nx = exx.shape[1]
    Ny = exx.shape[0]
    
    ux = np.zeros((Ny, Nx))
    uy = np.zeros((Ny, Nx))
    
    dx = Lx / Nx
    dy = Ly / Ny
    
    
    F = np.zeros((Ny, Nx, 2, 2), dtype=float)
    
    F[:, :, 0, 0] = 1 + exx
    F[:, :, 0, 1] = exy
    F[:, :, 1, 0] = eyx
    F[:, :, 1, 1] = 1 + eyy
    
    F_inv = np.linalg.inv(F)
    Dx = np.array([dx, 0])
    Dy = np.array([0, dy])
    Dx = F_inv@Dx
    Dy = F_inv@Dy
     
    for i in range(0, Ny):
        for j in range(0, Nx):
            
            if i == 0 and j == 0:
                continue
            elif i==0 and j>0:
                ux[i, j] = np.sum(exx[0, 0:j] * Dx[0, 0:j, 0] + exy[0, 0:j] * Dx[0, 0:j, 1])
                uy[i, j] = np.sum(eyx[0, 0:j] * Dx[0, 0:j, 0] + eyy[0, 0:j] * Dx[0, 0:j, 1])
            elif j==0 and i >0:
                ux[i, j] = np.sum(exx[0:i, 0] * Dy[0:i, 0, 0] + exy[0:i, 0] * Dy[0:i, 0, 1])
                uy[i, j] = np.sum(eyx[0:i, 0] * Dy[0:i, 0, 0] + eyy[0:i, 0] * Dy[0:i, 0, 1])
            else:
                ux_path1 = np.sum(exx[0:i, 0] * Dy[0:i, 0, 0] + exy[0:i, 0] * Dy[0:i, 0, 1]) + np.sum(exx[i, 0:j] * Dx[i, 0:j, 0] + exy[i, 0:j] * Dx[i, 0:j, 1])
                ux_path2 = np.sum(exx[0, 0:j] * Dx[0, 0:j, 0] + exy[0, 0:j] * Dx[0, 0:j, 1]) + np.sum(exx[0:i, j] * Dy[0:i, j, 0] + exy[0:i, j] * Dy[0:i, j, 1])
                uy_path1 = np.sum(eyx[0:i, 0] * Dy[0:i, 0, 0] + eyy[0:i, 0] * Dy[0:i, 0, 1]) + np.sum(eyx[i, 0:j] * Dx[i, 0:j, 0] + eyy[i, 0:j] * Dx[i, 0:j, 1])
                uy_path2 = np.sum(eyx[0, 0:j] * Dx[0, 0:j, 0] + eyy[0, 0:j] * Dx[0, 0:j, 1]) + np.sum(eyx[0:i, j] * Dy[0:i, j, 0] + eyy[0:i, j] * Dy[0:i, j, 1])
                ux[i, j] = (ux_path1 + ux_path2) / 2
                uy[i, j] = (uy_path1 + uy_path2) / 2

    return ux, uy



def correct_strain(I, Lx, Ly, ux, uy):
    Nx_region = ux.shape[1]
    Ny_region = ux.shape[0]
    x_region = np.linspace(0, Lx, Nx_region)
    y_region = np.linspace(0, Ly, Ny_region)
    
    Nx_valid = I.shape[1]
    Ny_valid = I.shape[0]
    x_valid = np.linspace(0, Lx, Nx_valid)
    y_valid = np.linspace(0, Ly, Ny_valid)
    
    X_valid, Y_valid = np.meshgrid(x_valid, y_valid)
    
    Fux = RegularGridInterpolator((y_region, x_region), ux)
    Fuy = RegularGridInterpolator((y_region, x_region), uy)
    
    ux_int = Fux((Y_valid, X_valid))
    uy_int = Fuy((Y_valid, X_valid))
    ux_int = ux_int - ux_int[int(ux_int.shape[0] / 2), int(ux_int.shape[1] / 2)]
    uy_int = uy_int - uy_int[int(uy_int.shape[0] / 2), int(uy_int.shape[1] / 2)]
    X_strain = X_valid - ux_int
    Y_strain = Y_valid - uy_int
    
    Interp = LinearNDInterpolator(list(zip(X_strain.flatten(), Y_strain.flatten())), I.flatten()) 
    I_corrected = Interp(X_valid, Y_valid)
    I_corrected = np.nan_to_num(I_corrected)
    
    return I_corrected
    