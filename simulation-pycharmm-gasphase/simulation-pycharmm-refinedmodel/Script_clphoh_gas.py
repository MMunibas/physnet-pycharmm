# Test Script to import PhysNet as energy function in CHARMM via PyCHARMM

# Basics
import os
import sys
import ctypes
import pandas
import numpy as np

# ASE
from ase import Atoms
from ase import io
import ase.units as units

# PyCHARMM
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.lingo as stream
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.cons_fix as cons_fix
import pycharmm.cons_harm as cons_harm
from pycharmm.lib import charmm as libcharmm
import pycharmm.lib as lib

# Step 0: Load parameter files
#-----------------------------------------------------------

toppar_data_dir = 'toppar'

# Cl-PhOH and Water
rtf_fn = os.path.join(toppar_data_dir, 'clphoh.top')
read.rtf(rtf_fn)
prm_fn = os.path.join(toppar_data_dir, 'clphoh.par')
read.prm(prm_fn, flex=True)

settings.set_bomb_level(-2)
settings.set_warn_level(-1)

# Step 1: Read F-PhOH in water
#-----------------------------------------------------------

# Generate system
read.sequence_string('PLIG')
gen.new_segment(
    seg_name='SOLU',
    setup_ic=True)

write.psf_card("para_clphoh.psf", title="para-Chlorophenol")

# Read system
read.coor_card("para_clphoh.cor")

# Step 2: Set CHARMM Properties
#-----------------------------------------------------------

# Non-bonding parameter
dict_nbonds = {
    'atom': True,
    'vdw': True,
    'vswitch': True,
    'cutnb': 14,
    'ctofnb': 12,
    'ctonnb': 10,
    'cutim': 14,
    'lrc': True,
    'inbfrq': -1,
    'imgfrq': -1
    }

nbond = pycharmm.NonBondedScript(**dict_nbonds)
nbond.run()

# Energy
energy.show()

# Optimization with initial CGenFF
minimize.run_sd(**{
    'nstep': 50,
    'tolenr': 1e-5,
    'tolgrd': 1e-5})

# Write pdb file
write.coor_pdb("init_clphoh.pdb", title="Para F-PhOH in water")

# Energy
energy.show()

# Step 2: Define PhysNet energy function
#-----------------------------------------------------------

# Prepare PhysNet input parameter
selection = pycharmm.SelectAtoms(seg_id='SOLU')

# Hard coded atomic numbers
# 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'H1', 'H2', 'H3', 'H4', 'F', 'O', 'H5'
Z = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 17, 8, 1]

# Checkpoint files
checkpoint = "../../best-models-6748/6748_CPhOH_mp2_631g-1"

# PhysNet config file
config = "../../best-models-6748/run_clpho.mp2.inp"

# Model units are eV and Angstrom
econv = 1./(units.kcal/units.mol)
fconv = 1./(units.kcal/units.mol)

charge = 0

# Initialize PhysNet calculator
pycharmm.MLpot(
    selection,
    fq=True,
    Z=Z,
    checkpoint=checkpoint,
    config=config,
    charge=charge,
    econv=econv,
    fconv=fconv,
    v1=True)	# Model is trained by PhysNet using tensorflow 1.x

# Custom energy
energy.show()
write.psf_card("clphoh_nobonds.psf")

# Step 2.8: MINI - CHARMM, PhysNet
#-----------------------------------------------------------

# Optimization with PhysNet parameter
minimize.run_sd(**{
    'nstep': 50,
    'tolenr': 1e-5,
    'tolgrd': 1e-5})

# Write pdb file
write.coor_pdb("mini_clphoh.pdb", title="Minimized para Cl-PhOH")

# Step 3: Heating - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.0005	# 0.5 fs

    res_file = pycharmm.CharmmFile(
        file_name='heat.res', file_unit=2, formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='heat.dcd', file_unit=1, formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': True,
        'nstep': 50.*1./timestep,
        'nsavc': 0.01*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea':-1,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 1000,
        'ihtfrq': 200,
        'ieqfrq': 1000,
        'firstt': 100,
        'finalt': 300,
        'tbath': 300,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_heat.run()
    
    res_file.close()
    dcd_file.close()

# Step 4: NVE - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.0002	# 0.2 fs

    str_file = pycharmm.CharmmFile(
        file_name='heat.res', file_unit=3, formatted=True, read_only=False)
    res_file = pycharmm.CharmmFile(
        file_name='nve.res', file_unit=2, formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='nve.dcd', file_unit=1, formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': False,
        'restart': True,
        'nstep': 10.*1./timestep,
        'nsavc': 0.01*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': str_file.file_unit,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 0,
        'ihtfrq': 0,
        'ieqfrq': 0,
        'firstt': 300,
        'finalt': 300,
        'tbath': 300,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_nve = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_nve.run()
    
    str_file.close()
    res_file.close()
    dcd_file.close()

# Step 5: Equilibration - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
        
    timestep = 0.0002	# 0.2 fs
    
    pmass = int(np.sum(select.get_property('mass'))/50.0)
    tmass = int(pmass*10)

    str_file = pycharmm.CharmmFile(
        file_name='heat.res', file_unit=3, formatted=True, read_only=False)
    res_file = pycharmm.CharmmFile(
        file_name='equi.res', file_unit=2, formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='equi.dcd', file_unit=1, formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': True,
        'verlet': False,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': False,
        'restart': True,
        'nstep': 50.*1./timestep,
        'nsavc': 0.01*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': str_file.file_unit,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 1000,
        'ihtfrq': 200,
        'ieqfrq': 0,
        'firstt': 300,
        'finalt': 300,
        'tbath': 300,
        'pgamma': 5,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_equi = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_equi.run()
    
    str_file.close()
    res_file.close()
    dcd_file.close()


# Step 6: Production - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.0002	# 0.2 fs

    pmass = int(np.sum(select.get_property('mass'))/50.0)
    tmass = int(pmass*10)

    for ii in range(0, 50):
        
        if ii==0:

            str_file = pycharmm.CharmmFile(
                file_name='equi.res', 
                file_unit=3, formatted=True, read_only=False)
            res_file = pycharmm.CharmmFile(
                file_name='dyna.{:d}.res'.format(ii), 
                file_unit=2, formatted=True, read_only=False)
            dcd_file = pycharmm.CharmmFile(
                file_name='dyna.{:d}.dcd'.format(ii), 
                file_unit=1, formatted=False, read_only=False)
            
        else:
            
            str_file = pycharmm.CharmmFile(
                file_name='dyna.{:d}.res'.format(ii - 1), 
                file_unit=3, formatted=True, read_only=False)
            res_file = pycharmm.CharmmFile(
                file_name='dyna.{:d}.res'.format(ii), 
                file_unit=2, formatted=True, read_only=False)
            dcd_file = pycharmm.CharmmFile(
                file_name='dyna.{:d}.dcd'.format(ii), 
                file_unit=1, formatted=False, read_only=False)
            
        # Run some dynamics
        dynamics_dict = {
            'leap': True,
            'verlet': False,
            'cpt': False,
            'new': False,
            'langevin': False,
            'timestep': timestep,
            'start': False,
            'restart': True,
            'nstep': 100.0*1./timestep,
            'nsavc': 0.001*1./timestep,
            'nsavv': 0,
            'inbfrq':-1,
            'ihbfrq': 50,
            'ilbfrq': 50,
            'imgfrq': 50,
            'ixtfrq': 1000,
            'iunrea': str_file.file_unit,
            'iunwri': res_file.file_unit,
            'iuncrd': dcd_file.file_unit,
            'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
            'iunldm':-1,
            'ilap': -1,
            'ilaf': -1,
            'nprint': 100, # Frequency to write to output
            'iprfrq': 500, # Frequency to calculate averages
            'isvfrq': 1000, # Frequency to save restart file
            'ntrfrq': 1000,
            'ihtfrq': 200,
            'ieqfrq': 0,
            'firstt': 300,
            'finalt': 300,
            'tbath': 300,
            'pgamma': 5,
            'iasors': 0,
            'iasvel': 1,
            'ichecw': 0,
            'iscale': 0,  # scale velocities on a restart
            'scale': 1,  # scaling factor for velocity scaling
            'echeck':-1}

        dyn_prod = pycharmm.DynamicsScript(**dynamics_dict)
        dyn_prod.run()
        
        str_file.close()
        res_file.close()
        dcd_file.close()
