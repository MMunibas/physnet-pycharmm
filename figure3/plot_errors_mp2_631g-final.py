import matplotlib.pyplot as plt
import numpy as np

font_axis = {'family':'sans-serif','color':'black','size':12,'weight':'demibold'}

file_list =[
"CPhOH-3245-1-energy-training.dat",
"CPhOH-3245-1-energy-test.dat",
"CPhOH-3245-1-force-training.dat",
"CPhOH-3245-1-force-test.dat",
"CPhOH-6748-energy-training.dat",
"CPhOH-6758-energy-test.dat",
"CPhOH-6748-force-training.dat",
"CPhOH-6758-force-test.dat"]

fig = plt.figure(figsize=(16,4))

outer_grid = fig.add_gridspec(1,2,wspace=0.13)
left_panel = outer_grid[0].subgridspec(2,2,wspace=0.0,hspace=0.0)

data1 = np.loadtxt(file_list[0])
x1 = data1[:,0]
y1 = data1[:,1]
delta1 = y1 - x1

ax1=fig.add_subplot(left_panel[0])
ax1.scatter(x1,delta1, s=5, color='black')
ax1.set_xticks([])
ax1.set_xlim([-1430,-1099])
ax1.set_ylim([-4,3])
fig.add_subplot(ax1)

data2 = np.loadtxt(file_list[1])
x2 = data2[:,0]
y2 = data2[:,1]
delta2 = y2 - x2

ax2=fig.add_subplot(left_panel[1])
ax2.scatter(x2,delta2, s=5, color='black')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim([-1430,-1099])
ax2.set_ylim([-4,3])
fig.add_subplot(ax2)

data3 = np.loadtxt(file_list[4])
x3 = data3[:,0]
y3 = data3[:,1]
delta3 = y3 - x3

ax3=fig.add_subplot(left_panel[2])
ax3.scatter(x3,delta3, s=5, color='r')
ax3.set_xlim([-1430,-1099])
ax3.set_ylim([-4,3])
fig.add_subplot(ax3)
plt.xticks(rotation=45)

data4 = np.loadtxt(file_list[5])
x4 = data4[:,0]
y4 = data4[:,1]
delta4 = y4 - x4

ax4=fig.add_subplot(left_panel[3])
ax4.scatter(x4,delta4, s=5, color='r')
ax4.set_yticks([])
ax4.set_xlim([-1430,-1099])
ax4.set_ylim([-4,3])
fig.add_subplot(ax4)
plt.xticks(rotation=45)

right_panel = outer_grid[1].subgridspec(2,2,wspace=0.0,hspace=0.0)

data5 = np.loadtxt(file_list[2])
x5 = data5[:,0]
y5 = data5[:,1]
delta5 = y5 - x5

ax5=fig.add_subplot(right_panel[0])
ax5.scatter(x5,delta5, s=5, color='black')
ax5.set_xticks([])
ax5.set_xlim([-999,1190])
ax5.set_ylim([-99,100])
fig.add_subplot(ax5)

data6 = np.loadtxt(file_list[3])
x6 = data6[:,0]
y6 = data6[:,1]
delta6 = y6 - x6

ax6=fig.add_subplot(right_panel[1])
ax6.scatter(x6,delta6, s=5, color='black')
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_xlim([-999,1190])
ax6.set_ylim([-99,100])
fig.add_subplot(ax6)


data7 = np.loadtxt(file_list[6])
x7 = data7[:,0]
y7 = data7[:,1]
delta7 = y7 - x7

ax7=fig.add_subplot(right_panel[2])
ax7.scatter(x7,delta7, s=5, color='r')
ax7.set_xlim([-999,1190])
ax7.set_ylim([-99,100])
fig.add_subplot(ax7)
plt.xticks(rotation=45)


data8 = np.loadtxt(file_list[7])
x8 = data8[:,0]
y8 = data8[:,1]
delta8 = y8 - x8

ax8=fig.add_subplot(right_panel[3])
ax8.scatter(x8,delta8, s=5, color='r')
ax8.set_yticks([])
ax8.set_xlim([-999,1190])
ax8.set_ylim([-99,100])
fig.add_subplot(ax8)
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.19)
fig.text(0.505,0.55,u'\u0394F (kcal/(mol$\cdot$\u00c5))',va='center',ha='center',rotation='vertical',fontdict=font_axis)
fig.text(0.1,0.55,u'\u0394E (kcal/mol)',va='center',ha='center',rotation='vertical',fontdict=font_axis)
fig.text(0.32,0.02,'Ab initio Energy (kcal/mol)',va='center',ha='center',fontdict=font_axis)
fig.text(0.73,0.02,'Ab initio Force (kcal/(mol$\cdot$\u00c5))',va='center',ha='center',fontdict=font_axis)

#save
#plt.savefig("errors.png", dpi=100,bbox_inches='tight')

#show

plt.show()
