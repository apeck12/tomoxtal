import numpy as np
import itertools
import pathos.pools as pp

from cctbx import miller, sgtbx, crystal
from cctbx.array_family import flex
from tomoxtal.utils import phases as phases_utils
from tomoxtal.utils import cctbx_tools


class LocateXtalOrigin:

    """
    Class for identifying the crystallographic phase origin based on phase residuals,
    both for symmetry-equivalent and centric reflections.
    """
    
    def __init__(self, data, sg_symbol, cell):
        """
        Initialize class.
        
        Parameters
        ----------
        data : numpy.ndarray, shape (N, 5)
            data array of [h,k,l,intensity,phase]; phase is in degrees and from peak-fitting
        sg_symbol : string
            space group symbol in Hermann-Mauguin notation, e.g. 'P 43 21 2'
        cell : tuple, length 6
            unit cell parameters (a,b,c,alpha,beta,gamma)
        """
        self.hkl = data[:,:3].astype(np.int32)
        self.intensities, self.phases = np.ascontiguousarray(data[:,3]), np.ascontiguousarray(data[:,4])
        self.crystal_symmetry = crystal.symmetry(unit_cell=cell,
                                                 space_group_symbol=sg_symbol)
        self.space_group = self.crystal_symmetry.space_group()
        self.sym_ops = cctbx_tools.get_sym_ops(spacegroup, friedels=True)
        self.get_phase_restrictions()
        self.get_symmetry_mappings()


    def generate_miller_array(self, data):
        """
        Generate a Miller array from the input data.

        Parameters
        ----------
        data : numpy.ndarray, shape (n_refl)
            data array of variable type, i.e. intensities or phases

        Returns
        -------
        ma : cctbx.miller.array object
            data reformatted as a cctbx Miller array of structure factors
        """
        ma = miller.array(miller_set=miller.set(self.crystal_symmetry, 
                                                flex.miller_index(self.hkl),
                                                anomalous_flag=False), data=flex.double(data))
        return ma


    def get_phase_restrictions(self):
        """
        Get phase restrictions (modulo pi) for centric reflections. Store
        centric Millers and their indices as class variables, along with 
        an array of the associated phase restrictions in degrees.
        """
        ma_p = self.generate_miller_array(self.phases)
        self.ind_centric = np.array(list(ma_p.centric_flags().data()))
        self.hkl_centric = self.hkl[self.ind_centric]
        
        p_restrictions = np.array([self.space_group.phase_restriction(tuple(m.astype(np.int32))).ht_angle() for m in self.hkl_centric])
        p_restrictions = np.rad2deg(p_restrictions)
        self.p_restrictions = np.vstack((p_restrictions - 180, p_restrictions, p_restrictions + 180))
        return
    
    
    def centric_residual(self, phases):
        """
        Compute phase residuals between centric reflections and their expected
        values.
        
        Parameters
        ----------
        phases : numpy.ndarray, shape (n_refl,)
            phases in degrees, ordered as self.hkl

        Returns
        -------
        c_residuals : numpy.ndarray, shape (n_centric,)
            centric residuals in degrees
        """
        c_residuals = np.min(np.abs(phases[self.ind_centric] - self.p_restrictions), axis=0)
        return c_residuals
    
    
    def get_symmetry_mappings(self):
        """
        Get the mappings between available symmetry-equivalent reflections and also
        store the information necessary for computing the symmetry phase residual.
        """
        # identify all reflections in the asymmetric unit
        ma_p = self.generate_miller_array(self.phases)
        hkl_asu = np.array(ma_p.merge_equivalents().array().indices())

        # generate symmetry-equivalents for each asu reflection
        num_ops = len(self.sym_ops.keys())
        hkl_sym = np.zeros((len(hkl_asu), 3, num_ops))
        for op in range(num_ops):
            hkl_sym[:,:,op] = np.matmul(hkl_asu, self.sym_ops[op][:3,:3].T)

        # translational operations for symmetry-equivalents
        T = np.zeros((1,3,num_ops))
        for op in range(num_ops):
            T[:,:,op] = np.array(self.sym_ops[op][:3,-1])
            
        # compute shifts to map symmetry-equivalent reflections to the asu
        self.sym_shifts = np.sum(hkl_sym * T, axis=1)

        # subtract observed reflections from all possible 
        hkl_reshape = np.moveaxis(hkl_sym.T, -1, 0)
        hkl_reshape1 = hkl_reshape.reshape(hkl_reshape.shape[0]*hkl_reshape.shape[1],3)
        diff = np.sum(np.abs(hkl_reshape1[:,np.newaxis] - self.hkl), axis=-1)
        diff = diff.reshape(hkl_reshape.shape[:2] + (self.hkl.shape[0],))

        # first and second axes correspond to position in self.hkl_sym; third axis is self.hkl index
        inds = np.array(np.where(diff==0))
        hkl_inds, map_inds = np.unique(inds[-1], return_index=True)
        self.sym_mapping = inds.T[map_inds]

        return
    
    
    def symmetry_residual(self, phases):
        """
        Compute phase residuals for symmetry-equivalent reflections, specifically
        the difference between every reflection and the average of its symmetry-
        equivalents.
        
        Parameters
        ----------
        phases : numpy.ndarray, shape (n_refl,)
            phases in degrees, ordered as self.hkl

        Returns
        -------
        s_residuals : numpy.ndarray, shape (n_refl,)
            residuals for symmetry-equivalent reflections in degrees
        """
        # map shifted phases to their asymmetric unit values; masked value is 360
        p_sym = 360*np.ones(self.sym_shifts.shape)
        p_sym[tuple(self.sym_mapping[:,:2].T)] = phases[self.sym_mapping[:,-1]]
        p_sym = np.ma.masked_values(p_sym, 360)
        p_shifted = phases_utils.wrap_phases(p_sym - 360 * self.sym_shifts)
        p_shifted[:,int(self.sym_shifts.shape[1]/2):] *= -1 # deal with Friedels
        
        # compute symmetry residuals, making sure to handle wrapping
        s_residuals = np.abs(phases_utils.wrap_phases(p_shifted - np.mean(p_shifted, axis=1)[:,np.newaxis]))
        s_residuals_pi = np.abs(phases_utils.wrap_phases(p_shifted + 180 - np.mean(p_shifted, axis=1)[:,np.newaxis]))
        s_residuals = s_residuals.data[~s_residuals.mask]
        s_residuals_pi = s_residuals_pi.data[~s_residuals_pi.mask]
        
        s_residuals = np.min(np.abs(np.vstack((s_residuals, s_residuals_pi))), axis=0)  
        return s_residuals
    
    
    def shift_phases(self, fshifts):
        """
        Shift all phases to new origin based on fractional shifts.
        
        Parameters
        ----------
        fshifts : numpy.ndarray, shape (3,)
            fractional shifts applied along (a,b,c)

        Returns
        -------
        shifted_phases : numpy.ndarray, shape (n_refls,)   
            phase values in degrees, ordered as self.hkl 
        """
        shifted_phases = self.phases - 360.0 * np.dot(self.hkl, fshifts)
        return phases_utils.wrap_phases(shifted_phases)
    
    
    def eval_origin(self, fshifts):
        """
        Compute residuals for candidate origin given by fshift.
        
        Parameters
        ----------
        fshifts : numpy.ndarray, shape (3,)
            fractional shifts applied along (a,b,c)

        Returns
        -------
        score : float
            mean of residuals (both centric and symmetry-equivalent) in degrees
        """
        phases = self.shift_phases(fshifts)
        c_residual = self.centric_residual(phases)
        s_residual = self.symmetry_residual(phases)
        scores = np.concatenate((c_residual,s_residual))
        
        return np.mean(scores)

    
    def scan_candidate_origins(self, grid_spacing, n_processes=1):
        """
        Perform a grid scan to assess each fractional origin shift as a candidate 
        crystallographic phase origin.
        
        Parameters
        ----------
        grid_spacing : float
            sampling frequency along each unit cell axis in Angstrom
        n_processes : int
            number of CPU processors over which to parallelize calculation 
        
        Returns
        -------
        fshifts : numpy.ndarray, shape (n_points, 3)
            fractional shifts along (a,b,c) that were evaluated
        scores : numpy.ndarray, shape (n_points,)
            mean phase residuals for fractional shifts that were evaluated
        """
        # set up list of fractional shifts
        cell = self.crystal_symmetry.unit_cell().parameters()[:3]
        xshifts, yshifts, zshifts = [np.arange(0, cell[i], grid_spacing) for i in range(3)]
        fshifts_list = np.array(list(itertools.product(xshifts/cell[0], 
                                                       yshifts/cell[1],
                                                       zshifts/cell[2]))) 
        print(f"Finding origin: {len(fshifts_list)} grid points to evaluate")
        
        # evaluate shifts using multiprocessing
        pool = pp.ProcessPool(n_processes)
        metrics = np.array(pool.map(self.eval_origin, fshifts_list))
        
        # reorder in decreasing likelihood of being the crystallographic origin
        ordering = np.argsort(metrics)
        return fshifts_list[ordering], metrics[ordering]

