import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

data = np.loadtxt(sys.argv[1])
x = data[:, 0]
y1 = data[:, 1]
#y2 = data[:, 2]
delta1 = y1 - x
#delta2 = y2 - x

fig,axis=plt.subplots(figsize=(8,4))

axis.axhline(y=0.0,color='r',linestyle='--')
axis.scatter(x, delta1,marker='o',s=60,color='black',label='TL model')
#axis.scatter(x, delta2,marker='s',s=60,facecolors='none',edgecolors='r',lw=1.2,label='TL model2')
axis.set_xlim([min(x)+50, max(x)+50])
axis.set_ylim([-100,115])
axis.set_xlabel('$\u03C9_{MP2}$ (cm⁻¹)', fontsize=12, fontweight='bold')
axis.set_ylabel('$Δ\u03C9$ (cm⁻¹)', fontsize=12, fontweight='bold')
axis.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
axis.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#axis.tick_params(axis='both', labelsize=12, direction='in', width=1, length=6)
axis.legend(loc='upper right')

plt.savefig('freq-tf.pdf', dpi=100,bbox_inches='tight')

plt.show()

