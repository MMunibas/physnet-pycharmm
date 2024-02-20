# Basics
import os
import sys
import numpy as np
from glob import glob

# MDAnalysis
import MDAnalysis
from MDAnalysis.analysis.distances import distance_array

# ASE - Basics
from ase import Atoms
from ase import io
from ase.calculators.physnet_v2 import PhysNet

# Statistics
from statsmodels.tsa.stattools import acovf

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt

# Miscellaneous
import ase.units as units

#-----------------------------
# Parameters
#-----------------------------

# Maximum number of dcd files to evaluate
Nmaxdcd = 13

# Requested jobs
request_reading = True 
request_reappending = True

recalc_IRspectra = True
recalc_PWspectra = True

# System tag
tag = "para_clphoh"

# ML atom indices
Nml = 13
ml_atoms = np.arange(Nml)

# Temperatures [K]
T = 300.0

# Main directory
maindir = os.getcwd()

# Result directory
rsltdir = 'results'

# Trajectory source
sys_fdcd = 'dyna.*.dcd'
sys_scrd = ['.', 1]

# System psf file
traj_psffile = 'para_clphoh.psf'

# Bond distance atom pairs
sys_bonds = [
    [0, 10], # C-Cl
    [11, 12] # O-H
    ]

# Bond distance atom pairs
sys_dihedrals = [
    [4, 3, 11, 12] # C-C-O-H
    ]
sys_dihedrals_labels = [
    "Ph-O-H"]

# Radial distribution function list
sys_rdfs = [
    [10, np.arange(13, 2654, 3)], # Cl-O(H2O)
    [11, np.arange(13, 2654, 3)], # O-O(H2O)
    [12, np.arange(13, 2654, 3)], # OH-O(H2O)
    ]
sys_rdfs_labels = [
    r"Cl$-$O$_{H_2O}$",
    r"O$_{OH}$$-$O$_{H_2O}$",
    r"O$_{OH}$$-$O$_{H_2O}$"]

# Distance binning density in 1/Angstrom
dst_step = 0.1

# Fixed Time step if needed
fixdt = None

# Conversion parameter
a02A = units.Bohr
kcalmol2Ha = units.kcal/units.mol/units.Hartree
kcalmol2J = units.kcal/units.mol/units.J
u2kg = units._amu
ms2Afs = 1e-5

u2au = units._amu/units._me
ms2au = units._me*units.Bohr*1e-10*2.0*np.pi/units._hplanck

# Time for speed of light in vacuum to travel 1 cm in fs
jiffy = 0.01/units._c*1e12

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 200

#-----------------------------
# Preparations
#-----------------------------

if not os.path.exists(os.path.join(maindir, rsltdir)):
    os.mkdir(os.path.join(maindir, rsltdir))

# Get dcd files
dcdfiles = np.array(glob(sys_fdcd))
iruns = np.array([
    int(dcdfile.split('/')[-1].split(sys_scrd[0])[sys_scrd[1]])
    for dcdfile in dcdfiles])

# Sort dcd files
dcdsort = np.argsort(iruns)
dcdfiles = dcdfiles[dcdsort]
iruns = iruns[dcdsort]

# Limit number of dcd files to evaluate
if len(iruns) > Nmaxdcd:
    iruns = iruns[:Nmaxdcd]

# Initialize PhysNet calculator
dcd = MDAnalysis.Universe(traj_psffile, dcdfiles[0])

# Sample system data
positions = dcd.trajectory[0]._pos[ml_atoms]
masses = dcd._topology.masses.values[ml_atoms]
atoms = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 17, 8, 1]
system = Atoms(atoms, positions=positions)
symbols = system.get_chemical_symbols()

# Checkpoint files
checkpoint = "../../best-models-3245/3245_CPhOH_mp2_631g-1"

# PhysNet config file
config = "../../best-models-3245/run_clpho.mp2.inp"

physnet = PhysNet(
    atoms=system,
    charge=0,
    qmmm=False,
    checkpoint=checkpoint,
    v1=True,
    config=config)

energy = physnet.get_potential_energy(system)
charges = physnet.get_charges(system)
#print(energy)
#print(charges)

#-----------------------------
# Read trajectories
#-----------------------------

# Initialize trajectory time counter in ps
traj_time_dcd = 0.0

# Initialize atom object
system_atoms = Atoms(atoms)

# Iterate over dcd files
for ir, idcd in enumerate(iruns):
    
    # ML atoms positions file
    rfile_i = os.path.join(
        rsltdir, 'mlpos_{:s}_{:d}.npy'.format(tag, idcd))
        
    # Dipole file
    dfile_i = os.path.join(
        rsltdir, 'dipos_{:s}_{:d}.npy'.format(tag, idcd))
    
    # Bond distances file
    afile_i = os.path.join(
        rsltdir, 'bonds_{:s}_{:d}.npy'.format(tag, idcd))
    
    # Dihedral angle file
    bfile_i = os.path.join(
        rsltdir, 'dihdr_{:s}_{:d}.npy'.format(tag, idcd))
    
    # Radial distribution file
    gfile_i = os.path.join(
        rsltdir, 'rdfs_{:s}_{:d}.npy'.format(tag, idcd))
    
    # Time file
    tfile_i = os.path.join(
        rsltdir, 'times_{:s}_{:d}.npy'.format(tag, idcd))
    
    # Open dcd file
    dcdfile = dcdfiles[ir]
    dcd = MDAnalysis.Universe(traj_psffile, dcdfile)
    
    # Get trajectory parameter
    Nframes = len(dcd.trajectory)
    Nskip = int(dcd.trajectory.skip_timestep)
    dt = np.round(
        float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
    if fixdt is not None:
        dt = fixdt
    
    if not os.path.exists(rfile_i) and request_reading:
        
        # ML atoms positions list
        rlst = []
        
        # Dipoles list
        dlst = []
        
        # Bond distances
        alst = []
        
        # Dihedral angle list
        blst = []
        
        # Radial distribution function list
        glst = []
        
        # Times list
        tlst = []
        
        # Iterate over frames
        for ii, frame in enumerate(dcd.trajectory):
            
            if not ii%100:
                print(ii, Nframes)
            
            # Get ML atom positions
            positions = frame._pos[ml_atoms]
            
            # Get cell information
            cell = frame._unitcell
            if ii == 0:
                dst_lim = np.min(cell[:3])/2.
                rdfs = [
                    np.zeros(int(dst_lim/dst_step), dtype=int)
                    for _ in sys_rdfs]
                
            
            # Set positions
            system_atoms.set_positions(positions)
            system_atoms.set_cell(cell)
            
            # Get energy and charges
            system.set_positions(positions)
            energy = physnet.get_potential_energy(system)
            charges = physnet.get_charges(system)
            
            # Calculate dipole moment
            dipole = np.dot(charges, positions)
            
            # Compute bond distances
            bdst = []
            for (aa, bb) in sys_bonds:
                bdst.append(system_atoms.get_distance(aa, bb))
            
            # Compute dihedral angles
            dang = []
            for (aa, bb, cc, dd) in sys_dihedrals:
                dang.append(system_atoms.get_dihedral(aa, bb, cc, dd))
            
            # Append results
            rlst.append(positions)
            dlst.append(dipole)
            alst.append(bdst)
            blst.append(dang)
            tlst.append(traj_time_dcd + ii*dt*Nskip)
    
        # Convert results to array
        rarr = np.array(rlst)
        darr = np.array(dlst)
        aarr = np.array(alst)
        barr = np.array(blst)
        tarr = np.array(tlst)
        
        # Save results to file
        np.save(rfile_i, rarr)
        np.save(dfile_i, darr)
        np.save(afile_i, aarr)
        np.save(bfile_i, barr)
        np.save(tfile_i, tarr)
        
    #else:
        
        #rarr = np.load(rfile_i)
        #darr = np.load(dfile_i)
        #aarr = np.load(afile_i)
        #barr = np.load(bfile_i)
        #rdfs = np.load(gfile_i)
        #tarr = np.load(tfile_i)
        
    # Increment trajectory time
    traj_time_dcd = idcd*Nframes*dt*Nskip
    
# Append results:
# ML atoms positions file
rfile = os.path.join(
    rsltdir, 'mlpos_{:s}.npy'.format(tag))
    
# Dipole file
dfile = os.path.join(
    rsltdir, 'dipos_{:s}.npy'.format(tag))

# Bond distances file
afile = os.path.join(
    rsltdir, 'bonds_{:s}.npy'.format(tag))

# Dihedral angle file
bfile = os.path.join(
    rsltdir, 'dihdr_{:s}.npy'.format(tag))

# Radial distribution file
gfile = os.path.join(
    rsltdir, 'rdfs_{:s}.npy'.format(tag))

# Time file
tfile = os.path.join(
    rsltdir, 'times_{:s}.npy'.format(tag))

if not os.path.exists(rfile) or request_reappending:
    
    # Positions list
    tmplst = []
    for ir, idcd in enumerate(iruns):
        rfile_i = os.path.join(
            rsltdir, 'mlpos_{:s}_{:d}.npy'.format(tag, idcd))
        if os.path.exists(rfile_i):
            tmplst.append(np.load(rfile_i))
    if len(tmplst):
        np.save(rfile, np.concatenate(tmplst))
    
    # Dipole list
    tmplst = []
    for ir, idcd in enumerate(iruns):
        dfile_i = os.path.join(
            rsltdir, 'dipos_{:s}_{:d}.npy'.format(tag, idcd))
        if os.path.exists(dfile_i):
            tmplst.append(np.load(dfile_i))
    if len(tmplst):
        np.save(dfile, np.concatenate(tmplst))
    
    # Bond distances list
    tmplst = []
    for ir, idcd in enumerate(iruns):
        afile_i = os.path.join(
            rsltdir, 'bonds_{:s}_{:d}.npy'.format(tag, idcd))
        if os.path.exists(afile_i):
            tmplst.append(np.load(afile_i))
    if len(tmplst):
        np.save(afile, np.concatenate(tmplst))
        
    # Dihedral angle list
    tmplst = []
    for ir, idcd in enumerate(iruns):
        bfile_i = os.path.join(
            rsltdir, 'dihdr_{:s}_{:d}.npy'.format(tag, idcd))
        if os.path.exists(bfile_i):
            tmplst.append(np.load(bfile_i))
    if len(tmplst):
        np.save(bfile, np.concatenate(tmplst))
    
    # Positions list
    tmplst = []
    for ir, idcd in enumerate(iruns):
        tfile_i = os.path.join(
            rsltdir, 'times_{:s}_{:d}.npy'.format(tag, idcd))
        if os.path.exists(tfile_i):
            tmplst.append(np.load(tfile_i))
    if len(tmplst):
        np.save(tfile, np.concatenate(tmplst))
    
#-----------------------------
# Plot IR
#-----------------------------

avgfreq = 5.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')

# Figure arrangement
figsize = (8, 6)
left = 0.15
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.70, 0.15])

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
axs1 = fig.add_axes([left, bottom, column[0], row[0]])

# IR spectra file
ifile = os.path.join(
    rsltdir, 'irspc_{:s}.npy'.format(tag))

# IR frequency file
ffile = os.path.join(
    rsltdir, 'irfrq_{:s}.npy'.format(tag))

# Dipole file
dfile = os.path.join(
    rsltdir, 'dipos_{:s}.npy'.format(tag))

# Time file
tfile = os.path.join(
    rsltdir, 'times_{:s}.npy'.format(tag))

# Number of frames and frequency points
tlst = np.load(tfile)
Nframes = len(tlst)
Nfreq = int(Nframes/2) + 1

# Frequency array
dtime = tlst[1] - tlst[0]
freq = np.arange(Nfreq)/float(Nframes)/dtime*jiffy

# Compute or load IR spectra
if not os.path.exists(ifile) or recalc_IRspectra:
    
    # Load results
    dlst = np.load(dfile)
   
    # Weighting constant
    beta = 1.0/3.1668114e-6/float(T)
    hbar = 1.0
    cminvtoau = 1.0/2.1947e5
    const = beta*cminvtoau*hbar
    
    # Compute IR spectra
    
    acvx = acovf(dlst[:, 0], fft=True)
    acvy = acovf(dlst[:, 1], fft=True)
    acvz = acovf(dlst[:, 2], fft=True)
    acv = acvx + acvy + acvz
    
    acv = acv*np.blackman(Nframes)
    spec = np.abs(np.fft.rfftn(acv))*np.tanh(const*freq/2.)
    
    # Save spectra and frequency range
    np.save(ifile, spec)
    np.save(ffile, freq)
    
else:
    
    # Load results
    spec = np.load(ifile)
    tlst = np.load(tfile)

# Apply moving average
Nave = int(avgfreq/(freq[1] - freq[0]))
#avgspec = moving_average(spec, Nave)
avgspec = spec

# Scale avgspec
select = np.logical_and(
    freq > 350, freq < 4000)
avgspec /= np.max(avgspec[select])

axs1.plot(
    freq[select], spec[select], '-', 
    label=tag)

axs1.set_title("IR spectra of '{:s}'".format(tag))
axs1.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontweight='bold')
axs1.get_xaxis().set_label_coords(0.5, -0.1)
axs1.set_ylabel('Intensity', fontweight='bold')
axs1.get_yaxis().set_label_coords(-0.15, 0.50)

axs1.set_xlim(freq[select][0], freq[select][-1])
axs1.set_ylim([0, 1.1])
axs1.set_yticklabels([])

axs1.legend(loc='upper right')

#plt.show()
plt.savefig(
    os.path.join(
        rsltdir, 'IR_spec_{:s}.png'.format(tag)),
    format='png', dpi=dpi)
plt.close()


#-----------------------------
# Plot Power spectrum
#-----------------------------

avgfreq = 5.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')

# Figure arrangement
figsize = (8, 6)
left = 0.15
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.70, 0.15])

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
axs1 = fig.add_axes([left, bottom, column[0], row[0]])

# Power spectra file
pfile = os.path.join(
    rsltdir, 'pwspc_{:s}.npy'.format(tag))

# Power spectra frequency file
ffile = os.path.join(
    rsltdir, 'pwfrq_{:s}.npy'.format(tag))

# Bond distance file
afile = os.path.join(
    rsltdir, 'bonds_{:s}.npy'.format(tag))

# Time file
tfile = os.path.join(
    rsltdir, 'times_{:s}.npy'.format(tag))

# Number of frames and frequency points
tlst = np.load(tfile)
Nframes = len(tlst)
Nfreq = int(Nframes/2) + 1

# Frequency array
dtime = tlst[1] - tlst[0]
freq = np.arange(Nfreq)/float(Nframes)/dtime*jiffy

# Compute or load Power spectra
if not os.path.exists(ifile) or recalc_PWspectra:
    
    # Load results
    alst = np.load(afile)
    
    # Power spectra
    spec = np.zeros((Nfreq, len(sys_bonds)), dtype=float)
    
    # Iterate over bonds
    for ii, (aa, bb) in enumerate(sys_bonds):
        acv = acovf(alst[:, ii], fft=True)
        acv = acv*np.blackman(Nframes)
        spec[:, ii] = np.abs(np.fft.rfftn(acv))*np.tanh(const*freq/2.)
    
    # Save spectra and frequency range
    np.save(pfile, spec)
    np.save(ffile, freq)
    
else:
    
    # Load results
    spec = np.load(pfile)
    tlst = np.load(tfile)
    
# Iterate over bonds
for ii, (aa, bb) in enumerate(sys_bonds):
    
    # Apply moving average
    Nave = int(avgfreq/(freq[1] - freq[0]))
    #avgspec = moving_average(spec[:, ii], Nave)
    avgspec = spec[:, ii]

    # Scale avgspec
    select = np.logical_and(
        freq > 350, freq < 4000)
    avgspec /= np.max(avgspec[select])

    label = "{:s}-{:s}".format(symbols[aa], symbols[bb])
    axs1.plot(
        freq[select], spec[select], '-', 
        label=label)

axs1.set_title("Power spectra of '{:s}'".format(tag))
axs1.set_xlabel(r'Frequency (cm$^{-1}$)', fontweight='bold')
axs1.get_xaxis().set_label_coords(0.5, -0.1)
axs1.set_ylabel('Intensity', fontweight='bold')
axs1.get_yaxis().set_label_coords(-0.15, 0.50)

axs1.set_xlim(freq[select][0], freq[select][-1])
axs1.set_ylim([0, 1.1])
axs1.set_yticklabels([])

axs1.legend(loc='upper right')

#plt.show()
plt.savefig(
    os.path.join(
        rsltdir, 'PW_spec_{:s}.png'.format(tag)),
    format='png', dpi=dpi)
plt.close()


#----------------------------------
# Plot Dihedral Angle Distribution
#----------------------------------

# Figure arrangement
figsize = (8, 6)
left = 0.15
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.70, 0.15])

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
axs1 = fig.add_axes(
    [left, bottom, column[0], row[0]],
    projection='polar')

# Dihedral angle file
bfile = os.path.join(
    rsltdir, 'dihdr_{:s}.npy'.format(tag))

# Load results
blst = np.load(bfile)

# Get and plot histogram
dih_rnge = [0.0, 360.0]
dih_step = 1.0
dih_bins = np.arange(0.0, dih_rnge[-1] + dih_step/2., dih_step)
dih_cntr = dih_bins[:-1] + dih_step/2.

for jj, (aa, bb, cc, dd) in enumerate(sys_dihedrals):
    
    dhst, _ = np.histogram(blst[:, jj], bins=dih_bins)
    
    axs1.plot(
        dih_cntr/180.*np.pi, dhst/np.max(dhst), '-')
    
axs1.set_title("{:s} Dihedral angle distribution in '{:s}'".format(
    sys_dihedrals_labels[jj], tag))
axs1.set_xlabel(r'Dihedral angle ($^\circ$)', fontweight='bold')
axs1.get_xaxis().set_label_coords(0.5, -0.1)
#axs1.set_ylabel('g(r)', fontweight='bold')
#axs1.get_yaxis().set_label_coords(-0.15, 0.50)

axs1.legend(loc='upper right')    

#plt.show()
plt.savefig(
    os.path.join(
        rsltdir, 'dihd_{:s}.png'.format(tag)),
    format='png', dpi=dpi)
plt.close()
