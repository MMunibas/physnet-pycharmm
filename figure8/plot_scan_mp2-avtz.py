import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
# 读取数据
data1 = np.loadtxt('traj-tf-dihedral-energy.dat')#699-tf-model
data2= np.loadtxt('mp2-avtz-rot.dat') #mp2/avtz

# 分离数据
x = data1[:,0]
y1 = data1[:,1]*23.0605 
y2_t = data2[:]  # 每隔10行取一次第三列数据

ene_min=min(y1)

y1 = [i-y1[0] for i in y1]

dev2=y2_t[0]-ene_min

y2 = []
for tp in y2_t:
    if tp != 0.0:
        tp = tp - ene_min -dev2
    y2.append(tp)
        
# 绘制图形
fig, ax = plt.subplots()

plt.rc('legend',fontsize=11)


# 绘制折线图
ax.plot(x, y1, '-', color='black', linewidth=3, label='TL model')
ax.set_xlabel(u"\u03B8 (\u00b0)", fontname="sans-serif", fontsize=12, fontweight="bold")
ax.set_ylabel("Energy (kcal/mol)", fontname="sans-serif", fontsize=12, fontweight="bold")
#ax.tick_params('y', colors='blue')


# 绘制散点图
ax.scatter(x[::10], y2[::10], facecolors='none',edgecolors='red',linewidth=2, label='MP2/aVTZ',zorder=4)  # 每隔10行绘制一次散点图
#ax.tick_params('y', colors='red')

x_major_locator=MultipleLocator(30)
ax.xaxis.set_major_locator(x_major_locator)

ax.set_xlim([-3,183])
ax.set_ylim([-0.2,3.5])

# 设置标题和标签
ax.legend(loc='lower center')
ax.legend(loc='lower center')


plt.savefig('scan-rot-avtz.pdf',dpi=100,bbox_inches='tight')

# 显示图形
plt.show()

