import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
# 读取数据
data1 = np.loadtxt('traj-6748-rot.dat')#6748-model
data2= np.loadtxt('traj-3245-rot.dat') #3245-model
data3= np.loadtxt('energy-rot-mp2.dat') #mp2/631g

# 分离数据
x = data1[:, 0]
y1_t = data1[:, 1]*23.0605 
y2_t = data2[:]*23.0605 
y3 = data3[:]  # 每隔10行取一次第三列数据

ene_min=min(y3[:])
dev2=y2_t[0]-ene_min

y2 = []
for i in y2_t:
    i = i - dev2
    y2.append(i)

dev1=y1_t[0]-ene_min
y1 = [i-dev1 for i in y1_t]

y1 = [i-ene_min for i in y1]
y2 = [i-ene_min for i in y2]
y3 = [i-ene_min for i in y3]

print('The barrier height of reference calculation is:'+str(y3[90])+'kcal/mol\n')
print('The barrier height of base model is :'+str(y2[90])+'kcal/mol\n')
print('The barrier height of refined model is :'+str(y1[90])+'kcal/mol\n')
print('The differnece of barrier height between the reference and base model is:'+str(y2[90]-y3[90])+'kcal/mol\n')
print('The differnece of barrier height between the reference and refined model is:'+str(y1[90]-y3[90])+'kcal/mol\n')

# 绘制图形
fig, ax = plt.subplots()

plt.rc('legend',fontsize=11)

ax.plot(x,y2, '-', color='black', linewidth=3,label='base model')

# 绘制折线图
ax.plot(x, y1, '--', color='red',linewidth=3, label='refined model')
ax.set_xlabel(u"\u03B8 (\u00b0)", fontname="sans-serif", fontsize=12, fontweight="bold")
ax.set_ylabel("Energy (kcal/mol)", fontname="sans-serif", fontsize=12, fontweight="bold")
#ax.tick_params('y', colors='blue')


# 绘制散点图
ax.scatter(x[::10], y3[::10], facecolors='none',edgecolors='limegreen',linewidth=2, label='MP2/6-31G(d,p)',zorder=4)  # 每隔10行绘制一次散点图
#ax.tick_params('y', colors='red')

x_major_locator=MultipleLocator(30)
ax.xaxis.set_major_locator(x_major_locator)

ax.set_xlim([-3,183])

# 设置标题和标签
ax.legend(loc='lower center')
ax.legend(loc='lower center')
ax.legend(loc='lower center')


plt.savefig('scan-rot-1.pdf',dpi=100,bbox_inches='tight')

# 显示图形
plt.show()

