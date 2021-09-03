import numpy as np
from tomoxtal.utils import cctbx_tools
from tomoxtal.utils import phases as phases_utils
from tomoxtal.pipeline import MergeCrystals

class TestMergeCrystals:
    
    def setup_class(self):
        """
        Prepare a few simulated datasets.
        """
        args = {'pdb_path':'/sdf/home/a/apeck/tomoxtal/examples/input/193l.pdb', 'resolution':6.0, 'size':250}

        # generate structure factors and retrieve associated cell information
        sf = cctbx_tools.reference_sf(args['pdb_path'], args['resolution'], expand_to_p1=True)
        sf_data = cctbx_tools.reformat_sf(sf)
        sg_symbol, sg_no, self.cell, cs = cctbx_tools.unit_cell_info(args['pdb_path'])
        
        # add random phase shifts
        hklIp1, hklIp2, hklIp3 = sf_data.copy(), sf_data.copy(), sf_data.copy()
        hklIp2[:,-1], self.shifts2 = phases_utils.add_random_phase_shift(sf_data[:,:3], sf_data[:,-1])
        hklIp3[:,-1], self.shifts3 = phases_utils.add_random_phase_shift(sf_data[:,:3], sf_data[:,-1])

        # retain subset of Millers
        for data in [hklIp1,hklIp2,hklIp3]:
            keep_idx = np.unique(np.random.randint(0, high=data.shape[0], size=args['size']))
            data = data[keep_idx]
        
        self.data1, self.data2, self.data3 = hklIp1, hklIp2, hklIp3
        fshifts_list = np.random.uniform(size=(4,3))
        self.fshifts_list = np.vstack((fshifts_list, 1-self.shifts2, 1-self.shifts3))
        
    def test_common_origin_search(self):
        """
        Test that the correct common phase origin is found.
        """
        mc = MergeCrystals()
        mc.add_crystal(self.data1, self.cell)
        fs, score = mc.merge_phases(self.data2, self.cell, fshifts_list=self.fshifts_list)
        assert np.allclose(fs, 1-self.shifts2)
        
    def test_add_crystal(self):
        """
        Testing that phase values and intensity values match, which equires both that 
        1) the correct common origin has been found and 2) that the intensities and
        phases have been properly assembled.
        """
        mc = MergeCrystals()
        mc.add_crystal(self.data1, self.cell)
        mc.add_crystal(self.data2, self.cell, fshifts_list=self.fshifts_list)
        assert np.isclose(np.sum(phases_utils.wrap_phases(np.abs(np.diff(mc.phases_all,axis=1)))), 0, atol=1e-06)
        assert np.isclose(np.sum(np.abs(np.diff(mc.I_all,axis=1))), 0, atol=1e-06)
        
        mc.add_crystal(self.data3, self.cell, fshifts_list=self.fshifts_list)
        assert np.isclose(np.sum(phases_utils.wrap_phases(np.abs(np.diff(mc.phases_all,axis=1)))), 0, atol=1e-06)
        assert np.isclose(np.sum(np.abs(np.diff(mc.I_all,axis=1))), 0, atol=1e-06)
