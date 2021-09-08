import numpy as np
from cctbx.array_family import flex
from cctbx import miller, sgtbx, crystal
import iotbx.pdb


def reference_sf(pdb_path, resolution, expand_to_p1=False, table='electron'):
    """
    Compute structure factors from a coordinates file to the specified resolution.
    
    Parameters
    ----------
    pdb_path : string
        path to coordinates file in PDB format
    resolution : float
        high-resolution limit
    expand_to_p1 : bool
        if True, expand structure factors to P1
    table : string
        scattering factor type; use 'n_gaussian' for X-rays
    
    Returns
    -------
    sf : scitbx.array object
        structure factors in cctbx format
    """
    pdb_input = iotbx.pdb.input(file_name=pdb_path)
    xrs = pdb_input.xray_structure_simple(crystal_symmetry=pdb_input.crystal_symmetry_from_cryst1())
    xrs.scattering_type_registry(table=table)
    sf = xrs.structure_factors(d_min=resolution, anomalous_flag=True).f_calc()
    if expand_to_p1 is True:
        sf = sf.expand_to_p1()

    return sf


def reformat_sf(sf):
    """
    Reformat cctbx scitbx structure factors array into a numpy
    data array of reflection information.
    
    Parameters
    ----------
    sf : scitbx.array object 
        structure factors in cctbx format

    Returns
    -------
    data : numpy.ndarray, shape (n_refls, 5)
        structure factor data formatted as [h,k,l,I,p], with phases in degrees
    """
    hkl = np.array(sf.indices())
    I = np.array(sf.intensities().data())
    p = np.rad2deg(np.array(sf.phases().data()))
    data = np.hstack((hkl, I[:,np.newaxis], p[:,np.newaxis]))
    return data


def retain_millers(sf_data, hkl_sel):
    """
    Retain reflections and any corresponding data that are present in hkl_sel.
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_refls, 5) 
        structure factor data formatted as [h,k,l,I,p], with phases in degrees
    hkl_sel : list of tuples
        reflections to retain

    Returns
    -------
    sel_data : numpy.ndarray, shape (len(hkl_sel), 5)
        selection of structure factor data formatted as [h,k,l,I,p]
    """
    indices = np.array([idx for idx in range(sf_data.shape[0]) if tuple(sf_data[idx,:3]) in hkl_sel])
    return sf_data[indices]


def unit_cell_info(pdb_path):
    """
    Extract unit cell information from a reference PDB file.
   
    Parameters
    ----------
    pdb_path : string
        path to coordinates file in PDB format

    Returns
    -------
    sg_symbol : string
        space group symbol in Hermann-Mauguin notation, e.g. 'P 43 21 2'
    sg_no : int
        space group number
    cell : tuple
        unit cell parameters (a,b,c,alpha,beta,gamma)
    cs : crystal.symmetry object
        instance of cctbx's crystal.symmetry class
    """
    pdb_input = iotbx.pdb.input(file_name=pdb_path)
    cs = pdb_input.crystal_symmetry()

    cell = cs.unit_cell().parameters()
    sg_info = cs.space_group().info().symbol_and_number()
    sg_symbol, sg_no = sg_info.split("(")[0][:-1], int(sg_info.split(".")[1][1:-1])

    return sg_symbol, sg_no, cell, cs


def sym_ops_friedels(sym_ops):
    """
    Expand the input dictionary of symmetry operations with those for each of
    the Friedel mates. For each symmetry operation, the rotational element is
    multiplied by negative 1 while the translational component is the same as
    the starting symmetry relationship.
    
    Parameters
    ----------
    sym_ops : dictionary
        symmetry operations with numbered keys and matrix values
    
    Returns
    -------
    sym_ops_friedels : dictinoary
        symmetry operation dictionary expanded to include operations for friedels
    """
    sym_ops_friedels = dict()
    for i,key in enumerate(sym_ops.keys()):
        sym_ops_friedels[key] = sym_ops[key]
        sym_ops_friedels[max(sym_ops.keys())+1+i] = np.vstack((-1*sym_ops[key][:,:-1].T, 
                                                               sym_ops[key][:,-1])).T
    return sym_ops_friedels


def get_sym_ops(sg_symbol, friedels=True):
    """
    Retrieve the symmetry operations for given space group.
    
    Parameters
    ----------
    sg_symbol : string
        space group symbol in Hermann-Mauguin notation, e.g. 'P 43 21 2'
    friedels : boolean
        if True, include operations for Friedels
    
    Returns
    -------
    sym_ops : dictionary
        symmetry operations with numbered keys and matrix values
    """
    s = sgtbx.space_group_symbols(sg_symbol)
    g = sgtbx.space_group(s.hall())
    ops = g.all_ops()

    sym_ops = {num:np.array(op.as_4x4_rational()).reshape(4,4).astype(float) for (num,op) in enumerate(ops)}
    if friedels:
        sym_ops = sym_ops_friedels(sym_ops)
    
    return sym_ops


def generate_crystal_symmetry_object(cell, sg_symbol):
    """
    Generate a cctbx crystal symmetry object.
    
    Parameters
    ----------
    cell : tuple, shape (6,)
        unit cell parameters (a,b,c,alpha,beta,gamma)
    sg_symbol : string
        space group symbol in Hall notation
        
    Returns
    -------
    cs : crystal.symmetry object
        instance of cctbx's crystal.symmetry class
    """
    
    from cctbx import crystal
    return crystal.symmetry(unit_cell=cell,
                            space_group_symbol=sg_symbol)


def data_to_miller_array(hklIp, cs):
    """
    Convert a numpy array of crystallographic data to a cctbx-style
    Miller array.
    
    Parameters
    ----------
    hklIp : numpy.ndarray, shape (n_reflections,5)
        data in format [h,k,l,I,phi], with phases in degrees
    cs : crystal.symmetry object
        instance of cctbx's crystal.symmetry class
        
    Returns
    -------
    ma : cctbx.miller.array object
        data reformatted as a cctbx Miller array of structure factors
    """
    hkl = [tuple(hklIp[i][:3].astype(np.int32)) for i in range(hklIp.shape[0])]
    I, p = hklIp[:,3], np.deg2rad(hklIp[:,4])
    A, B = np.sqrt(I)*np.cos(p), np.sqrt(I)*np.sin(p)
    B[np.abs(B)<1e-12] = 0

    indices = flex.miller_index(hkl)
    sf_data = flex.complex_double(flex.double(A),flex.double(B))
    ma = miller.array(miller_set=miller.set(cs, indices, anomalous_flag=False), data=sf_data)

    return ma


def compute_map(ma, savename = None, grid_step = 0.3):
    """
    Compute CCP4 map from a cctbx Miller array and save if an output path
    is given. Default grid step is 0.3 Angstroms.
    
    Parameters
    ----------
    ma : cctbx.miller.array object
        data reformatted as a cctbx Miller array of structure factors
    savename : string, optional
        output path for saving .ccp4 file
    grid_step : float 
        grid step in Angstrom for computing real space map 
        
    Returns
    -------
    fft_map : cctbx.miller.fft_map object
        real space map
    """
    from cctbx import maptbx
    
    fft_map = ma.fft_map(grid_step = grid_step,
                         symmetry_flags = maptbx.use_space_group_symmetry)
    fft_map = fft_map.apply_volume_scaling()

    if savename is not None:
        fft_map.as_ccp4_map(savename)

    return fft_map


def generate_miller_array(crystal_symmetry, hkl, data):
    """
    Generate a Miller array from the input data.
    
    Parameters
    ----------
    crystal_symmetry : crystal.symmetry object
        instance of cctbx's crystal.symmetry class
    hkl : numpy.ndarray, shape (n_refl, 3)
        Miller indices, ordered as data
    data : numpy.ndarray, shape (n_refl)
        data array of variable type, i.e. intensities or phases
        
    Returns
    -------
    ma : cctbx.miller.array object
        data reformatted as a cctbx Miller array of structure factors
    """ 
    ma = miller.array(miller_set=miller.set(crystal_symmetry, 
                                            flex.miller_index(hkl.astype(np.int32)),
                                            anomalous_flag=False), 
                      data=flex.double(np.ascontiguousarray(data)))
    return ma
