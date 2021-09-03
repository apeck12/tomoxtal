import numpy as np
import itertools
import pathos.pools as pp
from tomoxtal.utils import phases as phases_utils

class MergeCrystals:

    """
    Class for merging reflections from different crystals by positioning them
    on a common phase origin. Intensities are also merged.
    """

    def __init__(self, n_processes=1):
        self.hkl = None 
        self.I = None 
        self.phases = None 
        self.cell = None 
        self.n_processes = n_processes
        self.n_crystals = 0
        
    def retrieve_common_indices(self, hkl_next, I_next, phases_next):
        """ 
        Determine the indices of the reflections common between the current
        dataset and the one to be added.
        
        Parameters
        ----------
        hkl_next : numpy.ndarray, shape (3, n_refl)
            Miller indices from crystal to be added
        I_next : numpy.ndarray, shape (3, n_refl)
            intensities from crystal to be added, ordered as hkl_next
        phases_next : numpy.ndarray, shape (3, n_refl)
            phases from crystal to be added in degrees, ordered as hkl_next
        
        Returns
        -------
        indices : numpy.ndarray, shape (2, n_refl)
            common reflections in self data and data to be added are given by 
            indices[0] and indices[1], respectively
        """
        indices = np.where(np.prod(np.swapaxes(self.hkl[:,:,None],1,2) == hkl_next,axis=2).astype(bool))
        return indices
    
    def retrieve_common_data(self, hkl_next, I_next, phases_next):
        """
        Retrieve common data between self.variables and crystal to be added.

        Parameters
        ----------
        hkl_next : numpy.ndarray, shape (3, n_refl)
            Miller indices from crystal to be added
        I_next : numpy.ndarray, shape (3, n_refl)
            intensities from crystal to be added, ordered as hkl_next
        phases_next : numpy.ndarray, shape (3, n_refl)
            phases from crystal to be added in degrees, ordered as hkl_next
        
        Returns
        -------
        hkl1 : numpy.ndarray, shape (3, n_refl_common)
            Miller indices in common from self.hkl
        I1 : numpy.ndarray, shape (3, n_refl_common)
            intensities in common from self.I
        p1 : numpy.ndarray, shape (n_refl_common)
            subset of phases in common from self.phases, in degrees
        hkl2 : numpy.ndarray, shape (3, n_refl_common)
            Miller indices in common from crystal to be added
        I2 : numpy.ndarray, shape (3, n_refl_common)
            intensities in common from crystal to be added
        p2 : numpy.ndarray, shape (3, n_refl_common)
            phases in common from crystal to be added, in degrees
        """
        indices = self.retrieve_common_indices(hkl_next, I_next, phases_next)
        hkl1, I1, p1 = self.hkl[indices[0]], self.I[indices[0]], self.phases[indices[0]]
        hkl2, I2, p2 = hkl_next[indices[1]], I_next[indices[1]], phases_next[indices[1]]
        return hkl1, I1, p1, hkl2, I2, p2
        
    def shift_phases(self, hkl, phases, fshifts):
        """
        Shift all phases to new origin based on fractional shifts.

        Parameters
        ----------
        hkl : numpy.ndarray, shape (n_reflections,3)
            Miller indices [h,k,l]
        phases : numpy.ndarray, shape (n_reflections,)
            phases in degrees, ordered as hkl
        fshifts : tuple, length 3
            fractional shifts along unit cell axes (a,b,c)

        Returns
        -------
        shifted_phases : numpy.ndarray, shape (n_reflections,)
            shifted phases in degrees
        """
        shifted_phases = phases - 360.0 * np.dot(hkl, fshifts)
        return phases_utils.wrap_phases(shifted_phases)
 
    def compute_residual(self, fshifts, p1, hkl2, p2):
        """
        Compute the phase residual between shared reflections when data
        from the crystal to be added are shifted to a candidate origin.
        
        Parameters
        ----------
        fshifts : numpy.ndarray, shape (3,)
            fractional shifts applied to p2 along (a,b,c)
        p1 : numpy.ndarray, shape (n_refl_common)
            subset of phases in common from self.phases, in degrees
        hkl2 : numpy.ndarray, shape (3, n_refl_common)
            Miller indices in common from crystal to be added
        p2 : numpy.ndarray, shape (3, n_refl_common)
            phases in common from crystal to be added, in degrees
            
        Returns
        -------
        residuals : numpy.ndarray, shape (n_refl_common)
            mean phase residual for reflections in common between datasets
        """
        shifted_phases = self.shift_phases(hkl2, p2, fshifts)
        return np.mean(np.abs(phases_utils.wrap_phases(p1 - shifted_phases)))
            
    def wrap_compute_residual(self, args):
        """
        Wrapper for enabling pickling of arguments for pathos multiprocessing.
        """
        return self.compute_residual(*args)

    def rank_common_origins(self, hkl_next, I_next, phases_next, fshifts_list):
        """
        Rank the origins in fshifts_list based on the phase residual of 
        shared reflections.
        
        Parameters
        ----------
        hkl_next : numpy.ndarray, shape (3, n_refl)
            Miller indices from crystal to be added
        I_next : numpy.ndarray, shape (3, n_refl)
            intensities from crystal to be added, ordered as hkl_next
        phases_next : numpy.ndarray, shape (3, n_refl)
            phases from crystal to be added in degrees, ordered as hkl_next
        fshifts_list : numpy.ndarray of shape (n_origins,3)
            candidate origins in fractional unit cell space
            
        Returns
        -------
        fshifts_list : numpy.ndarray of shape (n_origins,3)
            candidate origins ranked in decreasing likelihood
        scores : numpy.ndarray of shape (n_origins,)
            phase residuals associated with candidate origins
        """
        # set up for using pathos multiprocessing
        pool = pp.ProcessPool(self.n_processes)
        num = len(fshifts_list)
        
        # get data in common and evaluate origins
        hkl1, I1, p1, hkl2, I2, p2 = self.retrieve_common_data(hkl_next, I_next, phases_next)
        args_eval = zip(fshifts_list, num*[p1], num*[hkl2], num*[p2])
        metrics = np.array(pool.map(self.wrap_compute_residual, args_eval))
               
        # reorder in decreasing likelihood of being the crystallographic origin
        ordering = np.argsort(metrics)
        return fshifts_list[ordering], metrics[ordering]
    
    def generate_candidate_origins(self, cell, grid_spacing):
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
    
    def merge_phases(self, hklIp, cell, grid_spacing=None, fshifts_list=None):
        """
        Merge phases from new crystal by finding a common phase origin and then
        updating the relevant class variables. If grid_spacing is provided, the
        unit cell is uniformly discretized and each node tested; otherwise, the
        origins in fshifts_list are evaluated. If both variables are provided,
        fshifts_list overrides grid_spacing.
        
        Parameters
        ----------
        hklIp : numpy.ndarray, shape (N, 5)
            data array of [h,k,l,intensity,phase]; phase is in degrees 
        cell : tuple, length 6
            unit cell parameters (a,b,c,alpha,beta,gamma)
        grid_spacing : float, optional
            sampling frequency in Angstrom
        fshifts_list : numpy.ndarray, shape (n_origins, 3)
            fractional unit cell shifts to test as candidate merge origins
            
        Returns
        -------
        fshifts : numpy.ndarray, shape (3,)
            fractional unit cell shift used for merging
        score : float
            phase residual associated with fshifts in degrees
        """
        # find best candidate origin and shift all phases to it
        hkl_next, I_next, phases_next = hklIp[:,:3], hklIp[:,3], hklIp[:,4]
        if fshifts_list is None:
            fshifts_list = self.generate_candidate_origins(cell, grid_spacing)
        fshifts_list, scores = self.rank_common_origins(hkl_next, I_next, phases_next, fshifts_list)
        p_shifted = self.shift_phases(hkl_next, phases_next, fshifts_list[0])
        print(f"Merging based on shift {fshifts_list[0]} with phase residual of {scores[0]} degrees")
        
        # retrieve both shared and unique indices
        indices = self.retrieve_common_indices(hkl_next, I_next, phases_next)
        unique_hkl = np.ones(hklIp.shape[0])
        unique_hkl[indices[1]] = 0

        # append new Miller indices
        self.hkl = np.vstack((self.hkl, hkl_next[unique_hkl==1]))
        
        # combine phase data
        p_combined = 360*np.ones((self.hkl.shape[0], self.n_crystals+1))
        p_combined[:len(self.phases),:self.n_crystals] = self.phases_all # original phases
        p_combined[indices[0],self.n_crystals] = p_shifted[indices[1]] # phases in common
        p_combined[len(self.phases):,self.n_crystals] = p_shifted[unique_hkl==1] # phases unique to new dataset
        
        # update class variables
        self.phases_all = np.ma.masked_values(p_combined, 360)
        self.phases = np.mean(phases_utils.wrap_phases(self.phases_all), axis=1).data
        
        return fshifts_list[0], scores[0]
    
    def merge_intensities(self, hklIp):
        """
        Merge intensities from new crystal -- for now, just by taking the mean.
        Note that this should be done after merging the phase data.
        
        Parameters
        ----------
        hklIp : numpy.ndarray, shape (N, 5)
            data array of [h,k,l,intensity,phase]; phase is in degrees 
        """
        
        # map new data Millers to class variables' ordering
        hkl_next, I_next, phases_next = hklIp[:,:3], hklIp[:,3], hklIp[:,4]
        indices = self.retrieve_common_indices(hkl_next, I_next, phases_next)
        
        # combine intensity data
        I_combined = np.zeros((self.hkl.shape[0], self.n_crystals+1))
        I_combined[:len(self.I),:self.n_crystals] = self.I_all
        I_combined[indices[0],self.n_crystals] = I_next[indices[1]]
        
        # update class variables
        self.I_all = np.ma.masked_values(I_combined, 0)
        self.I = np.mean(self.I_all, axis=1).data
        
        return

    def compile_data(self):
        """
        Compile the Miller indices, associated intensities, and phases into a
        numpy.array.
        
        Returns
        -------
        hklIp : numpy.ndarray, shape (n_refl, 5)
            data array of [h,k,l,intensity,phase]; phase is in degrees 
        """
        return np.hstack((self.hkl, self.I[:,np.newaxis], self.phases[:,np.newaxis]))

    def add_crystal(self, hklIp, cell, grid_spacing=None, fshifts_list=None):
        """
        Add data from new crystal. The origin is dictated by the first crystal; 
        all subsequent crystals are merged to this origin. Except for the first 
        crystal, either grid_spacing or fshifts_list has to be provided; if the
        former, the unit cell is discretized and each node tested as an origin.
        
        Parameters
        ----------
        hklIp : numpy.ndarray, shape (n_refl, 5)
            data array of [h,k,l,intensity,phase]; phase is in degrees 
        cell : tuple, length 6
            unit cell parameters (a,b,c,alpha,beta,gamma)
        grid_spacing : float, optional
            sampling frequency in Angstrom
        fshifts_list : numpy.ndarray, shape (n_origins, 3)
            fractional unit cell shifts to test as candidate merge origins
        """
        if self.hkl is None:
            self.hkl = hklIp[:,:3]
            self.I, self.phases = hklIp[:,3], hklIp[:,4]
            self.I_all, self.phases_all = hklIp[:,3][:,np.newaxis], hklIp[:,4][:,np.newaxis]
            self.cell = np.array(cell)
            
        else:
            self.cell = np.vstack((self.cell, cell))
            self.merge_phases(hklIp, cell, grid_spacing=grid_spacing, fshifts_list=fshifts_list)
            self.merge_intensities(hklIp)
            
        self.n_crystals += 1
