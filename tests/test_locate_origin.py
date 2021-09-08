import numpy as np
from tomoxtal.utils import cctbx_tools
from tomoxtal.utils import phases as phases_utils
from tomoxtal.pipeline import LocateXtalOrigin

class TestLocateXtalOrigin:
    
    def setup_class(self):
        """
        Prepare a few simulated datasets for a P212121 crystal.
        """
        args = {'pdb_path':'/sdf/home/a/apeck/tomoxtal/examples/input/3j7b.pdb', 'resolution':9.0, 'size':500}

        # generate structure factors and retrieve associated cell information
        sf = cctbx_tools.reference_sf(args['pdb_path'], args['resolution'], expand_to_p1=True)
        data = cctbx_tools.reformat_sf(sf)
        keep_idx = np.unique(np.random.randint(0, high=data.shape[0], size=args['size']))
        self.data = data[keep_idx]
        self.sg_symbol, sg_no, self.cell, cs = cctbx_tools.unit_cell_info(args['pdb_path'])
        
    def test_residuals_unshifted(self):
        """
        Check that known origins for space group 19 yield a phase residual of zero.
        """
        eq_pos = np.array([0,0.5,1.0])
        eq_pos = np.array(np.meshgrid(eq_pos,eq_pos,eq_pos)).T.reshape(-1,3)
        
        fo = LocateXtalOrigin(self.data, self.sg_symbol, self.cell, weighted=False)
        for fs in eq_pos:
            assert np.isclose(fo.eval_origin(fs), 0, atol=1e-06)
        
    def test_scan_shifted(self):
        """
        Check that correct origin is identified when a random phase shift is applied 
        to data. Here use the scan_candidate_origins function with intensity-weighting.
        """
        # add random phase shifts
        self.data[:,-1], shifts = phases_utils.add_random_phase_shift(self.data[:,:3], self.data[:,-1])
        fshifts_list = np.random.uniform(size=(4,3))
        fshifts_list = np.vstack((fshifts_list, 1-shifts))
        
        fo = LocateXtalOrigin(self.data, self.sg_symbol, self.cell, weighted=True)
        pred_shifts, scores = fo.scan_candidate_origins(fshifts_list=fshifts_list, n_processes=1)
        
        assert np.allclose(pred_shifts[0] + shifts, 1, atol=1e-06) 
        assert np.isclose(scores[0], 0, atol=1e-06) 
        assert not any([np.isclose(s, 0, atol=1e-06) for s in scores[1:]])
        
