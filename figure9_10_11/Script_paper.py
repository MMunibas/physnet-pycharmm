# Basics
import os
import sys
import numpy as np
from glob import glob

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

#-----------------------------
# Parameters
#-----------------------------

# System directories
sysdirs_env = [
    [
        "../simulation-pycharmm-gasphase/simulation-pycharmm-basemodel",
        "../simulation-pycharmm-gasphase/simulation-pycharmm-refined"
    ],
    [
        "../simulation-pycharmm-basemodel",
        "../simulation-pycharmm-refined"
    ]
]

sysdirs_rdf = [
    "../simulation-pycharmm-basemodel",
    "../simulation-pycharmm-refined"]

# System spectra and frequencies
sysspcs = [
    "results/irspc_para_clphoh.npy",
    "results/irspc_para_clphoh.npy"]
sysfrqs = [
    "results/irfrq_para_clphoh.npy",
    "results/irfrq_para_clphoh.npy"]

# System radial distribution function
sysrdfs = [
    "results/rdfs_para_clphoh.npy",
    "results/rdfs_para_clphoh.npy"]
dst_step = 0.1
sys_rdfs_labels = [
    r"Cl$-$O$_{H_2O}$",
    r"O$_{OH}$$-$O$_{H_2O}$",
    r"O$_{OH}$$-$O$_{H_2O}$"]

# System Dihedral angle Distribution
sysdihs = [
    "results/dihdr_para_clphoh.npy",
    "results/dihdr_para_clphoh.npy"]
dih_step = 1.0

# System tags
systags = [
    "Gas Phase",
    "Solution",
    "Exp."]

# System title
systitle = r"Para-Chlorophenole at 300K"

# Experimental reference spectra and frequencies
refdirs = [
    "reference"]
refdats = [
    "p_clphoh_10perc_ccl4.csv"]
refdat2 = "reference/ir_data_michakska.dat"

# Reference tags
reftags = [
    "exp."]

# Reference title
reftitle = r"p-ClPhOH (10wt%) in CCl$_4$"

# Plotstyle

figtag = 'p_clphoh'

# Fontsize
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Graphical output format type
gout_type = 'png'
dpi = 200

# Moving average definition
avgfreq = 5.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')

# Prepare second experimental spectra
dat2 = np.loadtxt(refdat2, delimiter=',')
argsort = np.argsort(dat2[:, 0])
dat2 = dat2[argsort]

# Apply moving average
refspec2 = moving_average(dat2[:, 1], 500)

# Reduce range
selection = np.logical_and(dat2[:, 0] > 821.5, dat2[:, 0] < 1645.0)
reffreq2 = dat2[:, 0][selection]
refspec2 = refspec2[selection]

# Invert spectra
refspec2 /= np.max(refspec2)
refspec2 = 1.0 - refspec2

#-----------------------------
# Plot IR
#-----------------------------

# Figure size
figsize = (8, 8)

# Figure arrangement
left = 0.15
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.80, 0.15])

# Frequency range
freq_range = [350, 4000]

# Plot colors
colors = [['blue', 'orange'], ['red', 'green'], 'black', 'magenta']

# Figure labels
axslabels = ['A', 'B', 'C']

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
Naxs = len(sysspcs) + len(refdats)
axs_all = [
    fig.add_axes(
        [left, bottom + (Naxs - ir - 1)*np.sum(row)/Naxs, column[0], row[0]/Naxs])
    for ir in range(Naxs)]
    
freq_lim = None

# Iterate over simulation spectra
for ir, sysdir_env in enumerate(sysdirs_env):
    
    linestyle = ['-', '--']
    
    for ii, sdir in enumerate(sysdir_env):
    
        # Load spectra data
        spec = np.load(os.path.join(sdir, sysspcs[ir]))
        freq = np.load(os.path.join(sdir, sysfrqs[ir]))
        
        # Frequency range
        select = np.logical_and(freq > freq_range[0], freq < freq_range[1])
        
        # Apply moving average
        Nave = int(avgfreq/(freq[1] - freq[0]))
        spec = moving_average(spec, Nave)
        spec /= np.max(spec[select])
        
        # Subfrequency range
        subfreq = 1725.
        subscale = 30.
        subselect = np.logical_and(freq > subfreq, freq < freq_range[1])
        spec[subselect] = spec[subselect]*subscale
        
        # Plot spectra
        axs_all[2].plot(
            freq[select], spec[select], ls=linestyle[ii], 
            color=colors[ir][ii], label=systags[ir])

        # Set axis range
        axs_all[2].set_xlim(freq[select][0], freq[select][-1])
        axs_all[2].set_ylim([0.0, 1.1])
        
        if freq_lim is None:
            freq_lim = [freq[select][0], freq[select][-1]]
        
        # Set legend
        #axs_all[ir].legend(loc='upper right')
        tbox = TextArea(
            systags[ir], textprops=dict(color='k', fontsize=MEDIUM_SIZE))
        anchored_tbox = AnchoredOffsetbox(
            loc='upper right', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.98, 0.95),
            bbox_transform=axs_all[ir].transAxes, borderpad=0.)
        axs_all[2].add_artist(anchored_tbox)
        
        # Mark intensity scaling
        axs_all[2].plot([subfreq, subfreq], [0, 1.1], '--k')
        axs_all[2].text(
            subfreq + 50., 1.0, 'x{:.0f}'.format(subscale),
            fontdict={
                'family' : 'monospace',
                'style'  : 'italic',
                'weight' : 'light',
                'size'   : 'small'})
        
        # Plot label
        tbox = TextArea(
            axslabels[ir], textprops=dict(color='k', fontsize=BIGGER_SIZE))
        anchored_tbox = AnchoredOffsetbox(
            loc='upper left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.95),
            bbox_transform=axs_all[ir].transAxes, borderpad=0.)
        axs_all[ir].add_artist(anchored_tbox)
    
## Iterate over reference spectra
#for ir, refdir in enumerate(refdirs):
    
    ## Load spectra data
    #data = np.loadtxt(
        #os.path.join(refdir, refdats[ir]),
        #delimiter=',')
    #spec = 1.0 - data[:, 1]
    #freq = data[:, 0]
    
    ## Frequency range
    #select = np.logical_and(freq > freq_range[0], freq < freq_range[1])
    
    ## Increment by number of simulation spectra
    #ia = ir + len(sysspcs)
    
    ## Plot spectra
    #axs_all[ia].plot(
        #freq[select], spec[select], '-', 
        #color=colors[ia], label=reftitle)#reftags[ir])

    ## Set axis range
    #axs_all[ia].set_xlim(freq_lim[0], freq_lim[1])
    #axs_all[ia].set_ylim([0.0, 1.1])
    
    ## Set legend
    ##axs_all[ia].legend(loc='upper right')
    #tbox = TextArea(
        #systags[ia], textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    #anchored_tbox = AnchoredOffsetbox(
        #loc='upper right', child=tbox, pad=0., frameon=False,
        #bbox_to_anchor=(0.98, 0.95),
        #bbox_transform=axs_all[ia].transAxes, borderpad=0.)
    #axs_all[ia].add_artist(anchored_tbox)
    
    ## Plot label
    #tbox = TextArea(
        #axslabels[ia], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    #anchored_tbox = AnchoredOffsetbox(
        #loc='upper left', child=tbox, pad=0., frameon=False,
        #bbox_to_anchor=(0.02, 0.95),
        #bbox_transform=axs_all[ia].transAxes, borderpad=0.)
    #axs_all[ia].add_artist(anchored_tbox)
    
## Second reference spectra
#axs_all[ia].plot(reffreq2, refspec2, '--', color=colors[ia + 1])


# Set axis tick labels
for axsi in axs_all[:-1]:
    axsi.set_xticklabels([])
    axsi.set_yticklabels([])
axs_all[-1].set_yticklabels([])

# Set axis labels
axs_all[-1].set_xlabel(r'Frequency $\nu$ (cm$^{-1}$)', fontweight='bold')
axs_all[-1].get_xaxis().set_label_coords(0.5, -0.1*Naxs)
axs_all[Naxs//2].set_ylabel(
    'scaled Absorption (arbitrary units)', fontweight='bold')
axs_all[Naxs//2].get_yaxis().set_label_coords(-0.08, (Naxs/2)%1)

plt.savefig(
    "ir_spectra_rev_{:s}.png".format(figtag),
    format='png', dpi=dpi)

plt.show()
exit()




#-----------------------------
# Plot IR II
#-----------------------------

# System directories
sysdirs_env = [
    "GasPhase/para_ClPhOH_kaisheng_basemodel",
    "GasPhase/para_ClPhOH_kaisheng_adaptive1",
    "para_ClPhOH_kaisheng_adaptive1"
    ]

# System spectra and frequencies
sysspcs = [
    "results/irspc_para_clphoh.npy",
    "results/irspc_para_clphoh.npy",
    "results/irspc_para_clphoh.npy"]
sysfrqs = [
    "results/irfrq_para_clphoh.npy",
    "results/irfrq_para_clphoh.npy",
    "results/irfrq_para_clphoh.npy"]

# System tags
systags = [
    "Gas Phase\nbase",
    "Gas Phase\nrefined",
    "Exp.",
    "Solution\nrefined"]

ir_list = [0, 1, 3]
ii4exp = 2

# Figure size
figsize = (8, 8)

# Figure arrangement
left = 0.15
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.80, 0.15])

# Frequency range
freq_range = [350, 4000]

# Plot colors
colors = ['blue', 'red', ['black', 'magenta'], 'green']

# Figure labels
axslabels = ['A', 'B', 'C', 'D']

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
Naxs = len(sysspcs) + len(refdats)
axs_all = [
    fig.add_axes(
        [left, bottom + (Naxs - ir - 1)*np.sum(row)/Naxs, column[0], row[0]/Naxs])
    for ir in range(Naxs)]
    
freq_lim = None

# Iterate over simulation spectra
for ii, sdir in enumerate(sysdirs_env):
    
    ir = ir_list[ii]
    
    # Load spectra data
    spec = np.load(os.path.join(sdir, sysspcs[ii]))
    freq = np.load(os.path.join(sdir, sysfrqs[ii]))
    
    # Frequency range
    select = np.logical_and(freq > freq_range[0], freq < freq_range[1])
    
    # Apply moving average
    Nave = int(avgfreq/(freq[1] - freq[0]))
    spec = moving_average(spec, Nave)
    spec /= np.max(spec[select])
    
    # Subfrequency range
    subfreq = 1725.
    subscale = 30.
    subselect = np.logical_and(freq > subfreq, freq < freq_range[1])
    spec[subselect] = spec[subselect]*subscale
    
    # Plot spectra
    axs_all[ir].plot(
        freq[select], spec[select], ls='-', 
        color=colors[ir], label=systags[ir])

    # Set axis range
    axs_all[ir].set_xlim(freq[select][0], freq[select][-1])
    axs_all[ir].set_ylim([0.0, 1.1])
    
    if freq_lim is None:
        freq_lim = [freq[select][0], freq[select][-1]]
    
    # Set legend
    #axs_all[ir].legend(loc='upper right')
    tbox = TextArea(
        systags[ir], textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper right', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.98, 0.95),
        bbox_transform=axs_all[ir].transAxes, borderpad=0.)
    axs_all[ir].add_artist(anchored_tbox)
    
    # Mark intensity scaling
    axs_all[ir].plot([subfreq, subfreq], [0, 1.1], '--k')
    axs_all[ir].text(
        subfreq + 50., 1.0, 'x{:.0f}'.format(subscale),
        fontdict={
            'family' : 'monospace',
            'style'  : 'italic',
            'weight' : 'light',
            'size'   : 'small'})
    
    # Plot label
    tbox = TextArea(
        axslabels[ir], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.02, 0.95),
        bbox_transform=axs_all[ir].transAxes, borderpad=0.)
    axs_all[ir].add_artist(anchored_tbox)
    

# Iterate over reference spectra
for ir, refdir in enumerate(refdirs):
    
    # Load spectra data
    data = np.loadtxt(
        os.path.join(refdir, refdats[ir]),
        delimiter=',')
    spec = 1.0 - data[:, 1]
    freq = data[:, 0]
    
    # Frequency range
    select = np.logical_and(freq > freq_range[0], freq < freq_range[1])
    
    # Increment by number of simulation spectra
    ia = ir + ii4exp
    
    # Plot spectra
    axs_all[ia].plot(
        freq[select], spec[select], '-', 
        color=colors[ia][0], label=reftitle)#reftags[ir])

    # Set axis range
    axs_all[ia].set_xlim(freq_lim[0], freq_lim[1])
    axs_all[ia].set_ylim([0.0, 1.1])
    
    # Set legend
    #axs_all[ia].legend(loc='upper right')
    tbox = TextArea(
        systags[ia], textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper right', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.98, 0.95),
        bbox_transform=axs_all[ia].transAxes, borderpad=0.)
    axs_all[ia].add_artist(anchored_tbox)
    
    # Plot label
    tbox = TextArea(
        axslabels[ia], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.02, 0.95),
        bbox_transform=axs_all[ia].transAxes, borderpad=0.)
    axs_all[ia].add_artist(anchored_tbox)
    
# Second reference spectra
axs_all[ia].plot(reffreq2, refspec2, '--', color=colors[ia][1])


# Set axis tick labels
for axsi in axs_all[:-1]:
    axsi.set_xticklabels([])
    axsi.set_yticklabels([])
axs_all[-1].set_yticklabels([])

# Set axis labels
axs_all[-1].set_xlabel(r'Frequency $\nu$ (cm$^{-1}$)', fontweight='bold')
axs_all[-1].get_xaxis().set_label_coords(0.5, -0.1*Naxs)
axs_all[Naxs//2].set_ylabel(
    'scaled Absorption (arbitrary units)', fontweight='bold')
axs_all[Naxs//2].get_yaxis().set_label_coords(-0.08, -(Naxs - 1)%2)

plt.savefig(
    "ir_spectra_rev2_{:s}.png".format(figtag),
    format='png', dpi=dpi)

#plt.show()
exit()


#-----------------------------
# Plot RDFs
#-----------------------------

# Figure size
figsize = (6, 6)

# Figure arrangement
left = 0.15
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.80, 0.15])

# Plot colors
colors = ['blue', 'red', 'green']

# Figure labels
axslabels = ['A', 'B', 'C']

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
Naxs = len(sysrdfs)
axs_all = [
    fig.add_axes(
        [left, bottom + (Naxs - ir - 1)*row[0]/Naxs, column[0], row[0]/Naxs])
    for ir in range(Naxs)]
    

# Iterate over simulation spectra
for ir, sysdir in enumerate(sysdirs_rdf):
    
    # Load g(r)
    rdfs_data = np.load(os.path.join(sysdir, sysrdfs[ir]))
    rdfs = np.zeros((3, len(rdfs_data[0])), dtype=float)
    Ndiv = len(rdfs_data)//3
    for ii, glsti in enumerate(rdfs_data):
        rdfs[ii%3, :] += glsti/Ndiv
    
    # Get distance bins and centers
    dst_rnge = [0.0, rdfs.shape[1]*dst_step]
    dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
    dst_cntr = dst_bins[:-1] + dst_step/2.
    
    # Plot RDF
    ls = ['-', '--', ':']
    for jj, label in enumerate(sys_rdfs_labels):
        
        axs_all[ir].plot(
            dst_cntr, rdfs[jj, :], ls=ls[jj], lw=3.0, 
            label=label)
    
    # Set axis range
    axs_all[ir].set_xlim(0.0, 10.0)
    axs_all[ir].set_ylim(-0.1, 1.9)
    
    # Plot label
    tbox = TextArea(
        axslabels[ir], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.02, 0.95),
        bbox_transform=axs_all[ir].transAxes, borderpad=0.)
    axs_all[ir].add_artist(anchored_tbox)
    
# Set axis tick labels
for axsi in axs_all[:-1]:
    axsi.set_xticklabels([])
    axsi.set_yticks(np.arange(0.0, 2.0, 0.5))
axs_all[-1].set_yticks(np.arange(0.0, 2.0, 0.5))

# Set axis labels
axs_all[-1].set_xlabel(r'Distance $r$ ($\mathrm{\AA}$)', fontweight='bold')
axs_all[-1].get_xaxis().set_label_coords(0.5, -0.1*Naxs)
axs_all[Naxs//2].set_ylabel(r'$g(r)$', fontweight='bold')
axs_all[Naxs//2].get_yaxis().set_label_coords(-0.12, 1.0 - (Naxs/2)%1)

axs_all[0].legend(loc='lower right')

plt.savefig(
    "rdfs_{:s}.png".format(figtag),
    format='png', dpi=dpi)




#----------------------------------
# Plot Dihedral Angle Distribution
#----------------------------------

# Figure size
figsize = (6, 4)

# Figure arrangement
left = 0.15
bottom = 0.20
row = np.array([0.65, 0.25])
column = np.array([0.40, 0.15])

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
axs1 = fig.add_axes(
    [left, bottom, column[0], row[0]],
    projection='polar')

axsl = fig.add_axes([
    left + np.sum(column), bottom + row[0]/2.5, 
    column[0]/2., row[0]/1.5])

# Get and plot histogram
dih_rnge = [0.0, 360.0]
dih_bins = np.arange(0.0, dih_rnge[-1] + dih_step/2., dih_step)
dih_cntr = dih_bins[:-1] + dih_step/2.

dhst_max = 0.0

# Free energy profile
dG_dih = np.zeros((len(sysdirs_rdf), len(dih_cntr)), dtype=float)
kb_kcalmolK = 1.987204259E-3
T_K = 300.0

# Plot colors
colors = ['blue', 'red', 'green']

# Panel labels
axslabels = ['A', 'B', 'C']
axslinestyles = ['-', '--', ':']

# Iterate over simulation
for ir, sysdir in enumerate(sysdirs_rdf):
    
    # Load data
    dihs = np.load(os.path.join(sysdir, sysdihs[ir]))
    
    # Get histogram
    dhst, _ = np.histogram(dihs[:, 0], bins=dih_bins)
    
    # Symmetrize
    dhst = dhst + np.roll(dhst, len(dhst)//2 - 1)
    dhst = dhst/np.sum(dhst)*(np.pi/180.)
    
    # Compute free energy profile
    dG_dih[ir] = -np.log(dhst)*kb_kcalmolK*T_K
    
    axs1.plot(
        dih_cntr/180.*np.pi, dhst, ls=axslinestyles[ir], color=colors[ir], lw=2)
    
    if np.max(dhst) > dhst_max:
        dhst_max = np.max(dhst)
        
    # Plot label
    tbox = TextArea(
        axslabels[0], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(-0.10, 1.19),
        bbox_transform=axs1.transAxes, borderpad=0.)
    axs1.add_artist(anchored_tbox)
    
    
axs1.set_xticks(np.array([0, 90, 180, 270])*np.pi/180.0)
axs1.set_ylim(0.0, dhst_max)
axs1.set_yticklabels([])

axs1.set_xlabel(
    #r'C-C-O-H Dihedral angle $\theta$ ($^\circ$)',
    r'$\theta$ ($^\circ$)', 
    fontweight='bold')
axs1.get_xaxis().set_label_coords(0.5, -0.2)
axs1.set_ylabel(
    r'Probability Distribution $P(\theta)$', 
    fontweight='bold')
axs1.get_yaxis().set_label_coords(-0.25, 0.5)

# Iterate over simulation
for ir, sysdir in enumerate(sysdirs_rdf):
    
    dih_cntr_half = dih_cntr[dih_cntr <= 180.]
    dG_dih_half = (dG_dih[ir] - np.nanmin(dG_dih[ir]))[dih_cntr <= 180.]
    axsl.plot(dih_cntr_half, dG_dih_half, color=colors[ir])
    

axsl.set_xlabel(r'$\theta$ ($^\circ$)', fontweight='bold')
axsl.set_ylabel(
    r'$\Delta G^{\mathrm{300K}}_\theta$ (kcal/mol)', 
    fontweight='bold')
axsl.get_yaxis().set_label_coords(1.35, 0.5)

axsl.set_xlim(0.0, 180.0)
axsl.set_xticks(np.array([0, 90, 180]))

# Plot label
tbox = TextArea(
    axslabels[1], textprops=dict(color='k', fontsize=BIGGER_SIZE))
anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(-0.40, 1.10),
    bbox_transform=axsl.transAxes, borderpad=0.)
axsl.add_artist(anchored_tbox)


plt.savefig(
    "dihs_{:s}.png".format(figtag),
    format='png', dpi=dpi)

