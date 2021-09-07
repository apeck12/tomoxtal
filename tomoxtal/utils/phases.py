import numpy as np
import random, pyfftw

def compute_ft(data):
    """ 
    Compute complex, centered DFT of an n-d numpy array using pyFFTW.
    
    Parameters
    ----------
    data : numpy.ndarray, n-d
        image or volume in real space
    
    Returns
    -------
    ft_data : numpy.ndarray, of shape data.shape
        complex structure factors
    """
    ft_data = pyfftw.empty_aligned(data.shape)
    ft_data[:] = data
    f = pyfftw.interfaces.scipy_fftpack.fftn(ft_data)
    return np.fft.fftshift(f)


def ft_to_I_phase(ft_data, deg=True):
    """
    Convert structure factors to separate numpy arrays of intensity and phase.

    Parameters
    ----------
    ft_data : numpy.ndarray, n-d
        complex structure factors
    deg : bool
        if True, convert phases to degrees
    
    Returns
    -------
    I : numpy.ndarray, of shape ft_data.shape
        intensity component of structure factor
    phase : numpy.ndarray, of shape ft_data.shape
        phase component of structure factor in radians
    """
    I = np.square(np.abs(ft_data))
    phase = np.arctan2(ft_data.imag, ft_data.real)
    if deg:
        phase = np.rad2deg(phase)
    return I, phase


def wrap_phases(phases, deg=True):
    """
    Wrap in the interval between -pi and pi. 
    
    Parameters
    ----------
    phases : numpy.ndarray, shape (N,)
        phase values in degrees
    deg : bool
        if True, assume input phases are in degrees
    
    Returns
    -------
    wrapped : numpy.ndarray, shape (N,)  
        phase values wrapped to the interval [-180,180)
    """
    if deg == True:
        wrapped = (phases + 180.) % (2 * 180.) - 180.
        wrapped[np.around(wrapped,3)==180] = -180
        return wrapped
    else:
        return (phases + np.pi) % (2 * np.pi) - np.pi
    
    
def average_phases(p_vals, weights=None):
    """
    Average phases using a method that works for circular data, with wrapping
    from -180 to 180. Modified code courtesy:
    https://stackoverflow.com/questions/491738/
    how-do-you-calculate-the-average-of-a-set-of-circular-data
    
    Parameters
    ----------
    p_vals : numpy.ndarray, shape (N,)
        phase values in degrees
    weights : numpy.ndarray, shape (N,), optional
        intensity-associated with each phase for weighting the average
    
    Returns
    -------
    p_avg : float
        phase average in degrees
    """
    
    if weights is None:
        weights = np.ones(p_vals.shape)
    
    x,y = 0,0
    for (angle, weight) in zip(p_vals, weights):
        x += np.cos(np.deg2rad(angle)) * weight / np.sum(weights)
        y += np.sin(np.deg2rad(angle)) * weight / np.sum(weights)
    
    return np.rad2deg(np.arctan2(y,x))


def std_phases(p_vals, weights=None):
    """
    Compute the standard deviation of input phases, yielding a value that
    roughly matches the output of scipy.stats.circstd (provided the range
    of phase values is not too large). Unlike scipy.stats.circstd, enable
    value-weighting of the phases.

    Parameters
    ----------
    p_vals : numpy.ndarray, shape (N,)
        phase values in degrees
    weights : numpy.ndarray, shape (N,), optional
        intensity-associated with each phase for weighting the average
    
    Returns
    -------
    p_std : float
        standard deviation of phases in degrees
    """
    p_avg = average_phases(p_vals, weights=weights)
    p_var = np.average(wrap_phases(p_vals - p_avg)**2, weights=weights)

    return np.sqrt(p_var)


def add_random_phase_shift(hkl, phases, fshifts=None):
    """
    Introduce a random phase shift, at most one unit cell length along each axis.
    
    Parameters
    ----------
    hkl : numpy.ndarray, shape (n_refls, 3)
        Miller indices
    phases : numpy.ndarray, shape (n_refls,)
        phase values in degrees, ordered as hkl
    fshifts : numpy.ndarray, shape (3,), optional
        fractional shifts along (a,b,c) to apply; if None, apply random shifts

    Returns
    -------
    shifted_phases : numpy.ndarray, shape (n_refls,)   
        phase values in degrees, ordered as hkl 
    fshifts : numpy.ndarray, shape (3,)
        fractional shifts applied along (a,b,c)
    """   
    if fshifts is None:
        fshifts = np.array([random.random() for i in range(3)])    

    shifted_phases = wrap_phases(phases - 360 * np.dot(hkl, fshifts).ravel())
    return shifted_phases, fshifts


def generate_candidate_origins(cell, grid_spacing):
    """
    Generate candidate origins by providing a list of nodes from the
    unit cell discretized by grid_spacing. While grid_spacing is given
    in Angstrom, the fractional unit cell shifts are returned. The cell
    is sampled at the same frequency along each unit cell axis.

    Parameters
    ----------
    cell : tuple, length 6
        unit cell parameters (a,b,c,alpha,beta,gamma)
    grid_spacing : float
        sampling frequency in Angstrom
    
    Returns
    -------
    fshifts_list : numpy.ndarray of shape (n_origins,3)
        candidate origins in fractional unit cell space
    """
    xshifts, yshifts, zshifts = [np.arange(0, cell[i], grid_spacing) for i in range(3)]
    fshifts_list = np.array(list(itertools.product(xshifts/cell[0], 
                                                   yshifts/cell[1],
                                                   zshifts/cell[2]))) 
    return fshifts_list
