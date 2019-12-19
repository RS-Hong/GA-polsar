# import matplotlib.pyplot as plt
#
# from pylab import mpl
#
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文，不然会乱码
#
# x = range(30)
# l1 = plt.plot(x, x, 'ro')
# l2 = plt.plot(x, [y ** 2 for y in x], 'bs')
# plt.title('不同线性测试')
# plt.xlabel('x坐标轴标签')
# plt.ylabel('y轴坐标标签')
# plt.legend((l1[0], l2[0]), ('1', '2'))
# plt.show()
import numpy as np
import os
print(os.path.join(os.getcwd(),'cfx_mx.txt'))