import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

def followup_brightness_scale(data, brightness=0.5, hsize=100):
    """
    Rescale intensities for visualization purposes. Original function
    was written by Aaron Brewster and Peter Zwart at LBL and has been
    adapted from there.
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n,n)
        pixels' intensity values
    brightness : float
        parameter controlling brightness
    hsize : int
        number of histogram bins
    
    Returns
    -------
    outvalue : numpy.ndarray, shape (n,n)
        pixels' scaled intensity values
    """
    qave = np.mean(data)
    histogram = np.zeros(hsize)

    for i in range(data.size):
        temp = int((hsize/2)*data.flatten()[i]/qave)
        if temp < 0: 
            histogram[0]+=1
        elif temp >= hsize: 
            histogram[hsize-1]+=1
        else: 
            histogram[temp]+=1

    percentile, accum = 0, 0
    for i in range(hsize):
        accum+=histogram[i]
        if (accum > 0.9*data.size):
            percentile=i*qave/(hsize/2)
            break

    adjlevel = 0.4 
    if percentile > 0.:
        correction = brightness * adjlevel/percentile
    else:
        correction = brightness / 5.0

    outscale = 256
    corrected = data*correction
    outvalue = outscale * (1.0 - corrected)
    outvalue[outvalue < 0] = 0
    outvalue[outvalue >= outscale] = outscale - 1
    return outvalue


def hsv_phase_plot(isel, psel):
    """
    Convert intensities and phases to an array that can be plotted by matplotlib's 
    imshow, in which hue / saturation / color is used to visualize intensities and
    phases simultaneously. Not recommended for large images.
    
    Parameters
    ----------
    isel : numpy.ndarray, shape (n,n)
        intensities
    psel : numpy.ndarray, shape (n,n)
        phases in degrees

    Returns
    -------
    c : numpy.ndarray, shape (n,n)
        HSV representation of isel and psel
    """
    isel = followup_brightness_scale(isel)
    psel = psel.astype(float)
    
    iplot = (256.0 - isel) / 256.0
    pplot = (psel + 180.0) / 360.0
    ones = np.ones(pplot.shape)

    c = colors.hsv_to_rgb(np.array([pplot.flatten(), iplot.flatten(), ones.flatten()]).T)
    return c.reshape(pplot.shape[0],pplot.shape[1],3)


def visualize_peak(ivol, pvol, mvol=None, sl=None, miller=None):
    """
    Visualize peak by displaying cross-sections through the intensity
    and phase subvolumes centered on that peak. The upper panel is a 
    surface plot through the intensities, while the lower panel shows
    the phases as colors and intensities as saturatin.
    
    Parameters
    ----------
    ivol : numpy.ndarray, shape (n,n,n)
        subvolume of Fourier intensities, centered on peak
    pvol : numpy.ndarray, shape (n,n,n)
        subvolume of Fourier phases in degrees, centered on peak
    mvol : numpy.ndarray, shape (n,n,n)
        subvolume mask, where 1 indicates a pixel that was retained
    sl : int, optional
        index for slicing through subvolumes; if None, show central slices
    miller : tuple, shape 3, optional 
        Miller indices for reflection, if provided use as plot title
    """
    if sl is None:
        sl = int(ivol.shape[0]/2) # assume cubic subvolume
        
    # surface plot of intensities
    f = plt.figure(figsize=(12,4))
    
    spacing = np.arange(ivol.shape[0]) - ivol.shape[0]/2
    xmesh,ymesh = np.meshgrid(spacing, spacing)
    
    for i,plane in enumerate([ivol[sl,:,:], ivol[:,sl,:], ivol[:,:,sl]]):
        ax = f.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(xmesh,ymesh,plane, cmap='plasma', vmax=ivol.max())
        
        if (i==1) and (miller is not None):
            ax.set_title(f"h,k,l={miller}", fontsize=16)

    # 2d plot of phases (color), with saturation determined by intensities
    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,3))
    
    iplanes = [ivol[sl,:,:],ivol[:,sl,:],ivol[:,:,sl]]
    pplanes = [pvol[sl,:,:],pvol[:,sl,:],pvol[:,:,sl]]
    if mvol is not None:
        mplanes = [mvol[sl,:,:],mvol[:,sl,:],mvol[:,:,sl]]
    
    for i,ax in enumerate([ax1,ax2,ax3]):
        ax.imshow(hsv_phase_plot(iplanes[i], pplanes[i]))
        if mvol is not None:
             for k in range(iplanes[i].shape[0]):
                for j in range(iplanes[i].shape[1]):
                    if mplanes[i][k,j] == 1:
                        ax.text(j,k,'X', c='black', ha='center', va='center', fontsize=14)
    
    return
