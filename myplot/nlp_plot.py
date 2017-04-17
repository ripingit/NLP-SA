import matplotlib.pyplot as plt
import numpy as np

#a = np.loadtxt(r"E:\Strange\SRTP.11\NLP\code\myplot\fig3.txt")  
'''
x = a[0,:]
KNN = a[1,:]
NB = a[2,:]
SVM = a[3,:]
plt.plot(x, KNN, 'r-x', label='KNN')
plt.plot(x, NB , 'g-^', label='NB')
plt.plot(x, SVM ,'b-o', label='SVM')
plt.legend() # 展示图例
plt.xlabel(u'样本数') # 给 x 轴添加标签
plt.ylabel('F-score') # 给 y 轴添加标签
#plt.title('Sin and Cos Waves') # 添加图形标题
plt.ylim(0.55,0.78)
plt.xlim(900,6100)
plt.show()
'''
n_groups = 3
onehot = (46.796,0.520,78.156)
vec = (5.233,0.056,7.149)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4
rects1 = plt.bar(index, onehot, bar_width,alpha=opacity, color='b',label=    'one-hot')
rects2 = plt.bar(index + bar_width, vec, bar_width,alpha=opacity,color='r',label='词向量')
plt.xlabel('分类器')
plt.ylabel('运行时间(t)')
plt.xticks(index + bar_width, ('KNN', 'NB', 'SVM'))
plt.ylim(-1,80)
plt.legend()
plt.tight_layout()
plt.show()