import numpy as np
import mrcfile
import scipy.ndimage, scipy.signal
from tomoxtal.utils import phases as phases_utils
from tomoxtal.utils import visualize


class PeakFitter:

    def __init__(self, volume, coords, hkl, box_size):
        """
        Initialize class.
        
        Parameters
        ----------
        volume : string
            path to real space crystal volume / tomogram in .mrc format
        coords : numpy.ndarray, shape (n_refls, 3), dtype int
            predicted or refined reflection coordinates
        hkl : numpy.ndarray, shape (n_refls, 3), dtype int
            Miller indices, ordered as coords
        box_size : int
            dimensions of box to extract
        """
        self.ivols, self.pvols = self.extract_subvolumes(volume, coords, hkl, box_size)
        self.radii = self.subvol_radii(box_size)
        self.tightmask = np.ones((box_size,box_size,box_size)) 
        
    def subvol_radii(self, box_size):
        """
        Generate a cubic subvolume of length box_size whose values
        are the radii of each voxel, to be used during peak-fitting.

        Parameters
        ----------
        box_size : int
            length of cubic subvolume

        Returns
        -------
        radii : numpy.ndarray, shape (box_size, box_size, box_size)
            subvolume of each voxel's radius from center
        """
        ls = np.linspace(-1, 1, box_size+1)
        ls = (ls[:-1] + ls[1:])/2
        radii = np.meshgrid(ls, ls, ls, indexing='ij')
        radii = np.sqrt(np.sum(np.square(radii), axis=0))
        return radii
    
    def generate_tightmask(self, tm_boxsize):
        """
        Generate a tight mask centered in the subvolume, where the central region 
        of dimensions given by tm_boxsize have a value of 1. An anisotropic tight
        mask is permitted given the anisotropy imposed by the missing wedge.

        Parameters
        ----------
        tm_boxsize : int or tuple of (int,int,int)
            dimensions of region to mask in pixels; if type int, then an isotropic
            subregion is masked

        Returns
        -------
        tightmask : numpy.ndarray, shape (N,N,N)
            mask of self.ivols shape, with a central subregion one-valued
        """
        subvol_dims = np.array(self.radii.shape)
        center = np.array(0.5*subvol_dims).astype(int)

        if type(tm_boxsize) == int:
            tm_boxsize = (tm_boxsize,tm_boxsize,tm_boxsize)
        tm_boxsize = np.array(0.5*np.array(tm_boxsize)).astype(int)

        tightmask = np.zeros(subvol_dims)
        tightmask[center[0]-tm_boxsize[0]:center[0]+tm_boxsize[0]+1,
                  center[1]-tm_boxsize[1]:center[1]+tm_boxsize[1]+1,
                  center[2]-tm_boxsize[2]:center[2]+tm_boxsize[2]+1] = 1

        return tightmask
        
    def eliminate_phase_splitting(self, volume):
        """
        Pre-process real space volume to prevent a circular discontinuity
        when taking the Fourier transform and thus phase splitting. This
        is accomlished by applying a window function and then shifting the
        center of the volume to the corner/origin of the Fourier transform.
        
        Parameters
        ----------
        volume : numpy.ndarray, shape (N,N,N)
            cubic volume of real space crystal
        
        Returns
        -------
        volume : numpy.ndarray, shape (N,N,N)
            volume after applying a window function and FFT-shift
        """
        assert volume.shape[0] == volume.shape[1] == volume.shape[2]
        
        # apply a Tukey kernel so that borders fall to 0
        k = scipy.signal.tukey(volume.shape[0])
        k3d = k[:,np.newaxis,np.newaxis] * k[np.newaxis,:,np.newaxis] * k[np.newaxis,np.newaxis,:]
        volume *= k3d
        
        # shift center to origin of volume
        return np.fft.fftshift(volume)
        
    def extract_subvolumes(self, volume, coords, hkl, box_size):
        """
        Extract intensity and phase subvolumes centered on the reflections.
        
        Parameters
        ----------
        volume : string
            path to real space crystal volume / tomogram in .mrc format
        coords : numpy.ndarray, shape (n_refls, 3), dtype int
            predicted or refined reflection coordinates
        hkl : numpy.ndarray, shape (n_refls, 3), dtype int
            Miller indices, ordered as coords
        box_size : int
            dimensions of box to extract

        Returns
        -------
        ivols : dictionary
            Miller index: subvolume array of intensities
        pvols : dictionary
            Miller index: subvolume array of phases in degrees
        """
        
        hb = int(box_size/2)
        pvols, ivols = dict(), dict()
        
        # Fourier transform real space volume
        volume = mrcfile.open(volume).data.copy().astype(np.float32)
        volume = self.eliminate_phase_splitting(volume)
        ftI, ftp = phases_utils.ft_to_I_phase(phases_utils.compute_ft(volume), deg=True)
        
        # extract phase and intensity subvolumes around each predicted peak
        for i,miller in enumerate(hkl):
            c = coords[i]
            ivols[tuple(miller)] = ftI[c[0]-hb:c[0]+hb+1,c[1]-hb:c[1]+hb+1,c[2]-hb:c[2]+hb+1]
            pvols[tuple(miller)] = ftp[c[0]-hb:c[0]+hb+1,c[1]-hb:c[1]+hb+1,c[2]-hb:c[2]+hb+1]
            
        return ivols, pvols
    
    def fit_peak(self, ivol, pvol, isigma, psigma, weighted=True, tm_boxsize=None):
        """
        Fit the peak in the intensity/phase subvolumes. First, high-intensity pixels 
        are selected and the set of contiguous, selected pixels nearest the subvolume
        center is chosen. Then, pixels from this selection are discarded until their 
        phase standard deviation is less than the psigma threhsold. 

        Parameters
        ----------
        ivol : numpy.ndarray, shape (N,N,N)
            subvolume of intensities
        pvol : numpy.ndarray, shape (N,N,N)
            subvolume of phases in degrees
        isigma : float
            threshold, pixel selected if its intensity > mean + isigma * std dev
        psigma : float
            threshold in degrees, pixels discarded while phase std dev exceeds this value
        weighted : bool
            whether to intensity-weight the mean phase calculation
        tm_boxsize : int or tuple of (int,int,int)
            dimensions of central region of subvolume to consider for peak-fitting.
            The mask will default to its previously-used value if set before.
            
        Returns
        -------
        ival : float
            peak intensity, or 0 if no peak was found
        pval : float
            peak phase in degrees, or 0 if no peak was found
        mask : numpy.ndarray, shape (N,N,N)
            subvolume in which 1 indicates that voxel was retained during peak fitting
        """
        # generate a tight mask as requested
        if tm_boxsize is not None:
            self.tightmask = self.generate_tightmask(tm_boxsize)

        # identify high intensity pixels
        ithreshold = ivol.mean() + isigma*ivol.std()
        imask = np.zeros_like(ivol)
        imask[np.where((ivol>ithreshold) & (self.tightmask==1))] = 1
        indices = np.array(np.where(imask==1))
        
        # condition of no high intensity pixels in valid region
        if indices.size == 0:
            return 0, 0, np.zeros(self.radii.shape)

        # find set of contiguous pixels that are nearest to the center
        struct = scipy.ndimage.generate_binary_structure(3, 1)
        labeled, ncomponents = scipy.ndimage.measurements.label(imask, struct)
        d = {nc:np.mean(self.radii[labeled==nc]) for nc in range(1,ncomponents+1)}
        label = min(d, key=d.get)

        # select relevant intensity and phase values
        ivals = ivol[np.where(labeled==label)]
        pvals = pvol[np.where(labeled==label)]
        indices = np.array([np.where(labeled==label)])

        # discard pixels until psigma threshold is met
        if phases_utils.std_phases(pvals, weights=ivals) > psigma:
            sort_idx = np.argsort(ivals)
            ivals, pvals = ivals[sort_idx], pvals[sort_idx]

            for ni in range(ivals.shape[0]):
                if phases_utils.std_phases(pvals[ni:], weights=ivals[ni:]) < psigma:
                    break
            ivals, pvals = ivals[ni:], pvals[ni:]
            indices = indices.T[sort_idx][ni:]
        else:
            indices = indices.T
            
        # compute peak intensity and phase
        ival = np.mean(ivals)
        if weighted:
            pval = phases_utils.average_phases(pvals, weights=ivals)
        else:
            pval = phases_utils.average_phases(pvals)
        print("Phases:", pvals)

        # generate a mask to spatially track retained pixels
        mask = np.zeros(self.radii.shape)
        mask[indices[:,0], indices[:,1], indices[:,2]] = 1
        
        return ival, pval, mask
    
    def fit_all_peaks(self, isigma, psigma, weighted=True, tm_boxsize=None):
        """
        Fit the Bragg peak intensity and phase in each subvolume.
        
        Parameters
        ----------
        isigma : float
            threshold, pixel selected if its intensity > mean + isigma * std dev
        psigma : float
            threshold in degrees, pixels discarded while phase std dev exceeds this value
        weighted : bool
            whether to intensity-weight the mean phase calculation
        tm_boxsize : int or tuple of (int,int,int)
            dimensions of central region of subvolume to consider for peak-fitting
            
        Returns
        -------
        hklIp : numpy.ndarray, shape (N, 5)
            data array of [h,k,l,intensity,phase]; phase is in degrees and from peak-fitting
        """
        # set up storage array and dictionaries
        hklIp = np.zeros((len(self.ivols.keys()), 5))
        self.masks = dict()
        
        # generate a tight mask as requested
        if tm_boxsize is not None:
            self.tightmask = self.generate_tightmask(tm_boxsize)
        
        # fit all peaks
        for i,miller in enumerate(self.ivols.keys()):
            print(f"Fitting reflection {miller}")
            peakI, peakp, self.masks[miller] = self.fit_peak(self.ivols[miller],
                                                             self.pvols[miller],
                                                             isigma,
                                                             psigma,
                                                             weighted=weighted)
            if peakI == 0:
                print(f"Warning: no peak found for reflection {miller}")
            hklIp[i] = np.array(miller + (peakI, peakp))
            
        # remove any peaks that couldn't be fit
        hklIp = hklIp[hklIp[:,3]!=0]
        
        return hklIp
        
    def visualize_peak(self, miller, sl=None, use_mask=True):
        """
        Generate a figure that visualizes cross-sections through the peak's intensity
        and phase values.
        
        Parameters
        ----------
        miller : tuple, shape 3, optional 
            Miller indices for reflection, if provided use as plot title
        sl : int, optional
            index for slicing through subvolumes; if None, show central slices
        """
        mvol = None
        if use_mask:
            mvol = self.masks[miller]
            
        visualize.visualize_peak(self.ivols[miller],
                                 self.pvols[miller],
                                 mvol=mvol, 
                                 miller=miller,
                                 sl=sl)
        return 
