import numpy as np
from skimage.transform import hough_line
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings


def hough_angle(I, angle=None):

    
    
    N_th = 1000
    N_q = I.shape[0]
    th = np.linspace(0, np.pi, N_th)
    averageAngle = np.zeros(N_q)
    
    
    def fun(x, B0, A0, sig, mu):
        return B0 + A0 * np.exp(-(x - mu)**2 / (2 * sig**2))
    
    for i in range(0, N_q):
        if angle is None:
            th = np.linspace(0, np.pi, N_th)
        else:
            th = np.linspace(angle[i] - np.pi/6, angle[i] + np.pi/6, int(N_th / 3))
        
        h0, theta, d = hough_line(I[i], th)
        h0 = (h0 - h0.min()) / (h0.max() - h0.min())
        h0 = h0**2
        a0 = h0.sum(axis=0)
        i_max = np.argmax(a0)

        index_min = max(0, i_max - 100)
        index_max = min(N_th - 1, i_max + 100)
        sample = a0[index_min: index_max]
        theta_sample = theta[index_min: index_max]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                coe, cov = curve_fit(fun, theta_sample, sample, p0=[1.0, a0[i_max], 0.1, theta[i_max]])
                averageAngle[i] = coe[3]
            except RuntimeError:
                averageAngle[i] = theta[i_max]
                
            if w:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(theta_sample, sample, 'ro')
                ax.plot(theta, fun(theta, *coe), 'g--')
                plt.show()
        
        
    return averageAngle